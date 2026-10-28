"""Run the real README workload under NVTX; invoked by profile_examples.py."""

import argparse
import functools
import gc
import importlib
import inspect
import json
import os
import runpy
import sys
import time
from contextlib import ExitStack
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOMAIN = "newton-solver"


def phase(name, module="", owner=""):
    """Label Python orchestration only; never wrap Warp kernels or scalar math."""
    if name in {
        "backward",
        "_backward",
        "_backward_frame",
        "loss_gradient",
        "pressure_step",
    }:
        return "backward"
    if name in {"forward", "_forward", "_forward_frame"}:
        return "forward"
    if name in {"train", "optimize_step", "optimize_velocity"}:
        return "optimization"
    if name in {"cut", "refresh_topology", "update_topology"}:
        return "cut_topology"
    if name in {
        "collide",
        "_contact_response",
        "eval_particle_ground_contact_forces",
        "eval_soft_contact_forces",
    }:
        return "contact"
    if name.startswith(("_assemble", "assemble_")) or name in {
        "assemble",
        "_build_constant_matrix",
        "_write_element_blocks",
        "_rebuild_element_blocks",
        "refresh_tangent",
    }:
        return "assembly"
    if name in {
        "solve",
        "_solve",
        "factor",
        "refactor",
        "_factor",
        "_factor_ldlt",
        "_newton_factorize",
    }:
        if "contact" in owner.lower() or module.endswith("response"):
            return "contact_solve"
        return (
            "linear_solve" if any(word in (module + owner).lower() for word in ("linear", "sparse", "pcg")) else "solve"
        )
    if name in {
        "step",
        "substep",
        "graph_step",
        "_step_single",
        "_step_batched",
        "implicit_integration",
    }:
        return "solver"
    if name in {
        "_evaluate_forces_single",
        "_evaluate_forces_batched",
        "_accumulate_forces",
    }:
        return "forces"
    if name in {"_residual", "_dynamic_residual"}:
        return "residual"
    return None


class Instrumentation:
    """Temporary host annotations, with no changes to ordinary solver execution."""

    def __init__(self, package, nvtx):
        self.package, self.nvtx = package, nvtx
        self.originals = []
        self.labels = []

    def wrap(self, owner, name, function, label, descriptor=None):
        if getattr(function, "_nsys_wrapped", False):
            return

        @functools.wraps(function)
        def annotated(*args, **kwargs):
            with self.nvtx.annotate(label, domain=DOMAIN):
                return function(*args, **kwargs)

        annotated._nsys_wrapped = True
        self.originals.append((owner, name, vars(owner)[name]))
        setattr(owner, name, descriptor(annotated) if descriptor else annotated)
        self.labels.append(label)

    def install(self):
        # Modules may load lazily during construction, so call again before frames.
        classes = set()
        for module in tuple(sys.modules.values()):
            if module is None or not getattr(module, "__name__", "").startswith(self.package):
                continue
            for name, value in tuple(vars(module).items()):
                if inspect.isfunction(value) and value.__module__.startswith(self.package):
                    category = phase(name, value.__module__)
                    if category:
                        self.wrap(module, name, value, f"{category}:{value.__module__}.{name}")
                elif inspect.isclass(value) and value.__module__.startswith(self.package):
                    classes.add(value)
        for cls in classes:
            for name, value in tuple(vars(cls).items()):
                descriptor = type(value) if isinstance(value, (staticmethod, classmethod)) else None
                function = value.__func__ if descriptor else value
                if not inspect.isfunction(function):
                    continue
                category = phase(name, cls.__module__, cls.__name__)
                if category:
                    self.wrap(
                        cls,
                        name,
                        function,
                        f"{category}:{cls.__name__}.{name}",
                        descriptor,
                    )

    def restore(self):
        for owner, name, original in reversed(self.originals):
            setattr(owner, name, original)
        self.originals.clear()


def run_frames(
    example,
    args,
    name,
    frames,
    hooks,
    nvtx,
    synchronize,
    *,
    result=None,
    final_test=True,
):
    """Time host submission; synchronize only at the measured interval boundaries."""
    synchronize()
    start = time.perf_counter()
    timings = []
    result = {} if result is None else result
    result.update(frames_completed=0, frame_host_timings=timings)
    for frame in range(frames):
        row = {"frame": frame}
        with nvtx.annotate(f"frame:{frame}", domain=DOMAIN):
            for label, callback in (
                (
                    "scenario",
                    lambda frame=frame: hooks.get("before_frame", lambda *_: None)(name, example, frame),
                ),
                ("step", example.step),
                (
                    "test",
                    lambda: example.test_post_step() if args.test and hasattr(example, "test_post_step") else None,
                ),
                ("render", example.render),
            ):
                if callback is None:
                    continue
                tick = time.perf_counter()
                with nvtx.annotate(label, domain=DOMAIN):
                    callback()
                row[label + "_host_seconds"] = time.perf_counter() - tick
        timings.append(row)
        result["frames_completed"] = len(timings)
    with nvtx.annotate("final_device_sync", domain=DOMAIN):
        synchronize()
    elapsed = time.perf_counter() - start
    with nvtx.annotate("test_final", domain=DOMAIN):
        if final_test and args.test and hasattr(example, "test_final"):
            example.test_final()
    result["loop_wall_seconds"] = elapsed
    return result


def run_visual(
    module,
    arguments,
    name,
    options,
    hooks,
    nvtx,
    synchronize,
    instrumentation,
    *,
    warmup,
    result=None,
):
    import newton.examples
    import newton.viewer

    frames = options.warmup if warmup else options.frames
    result = {} if result is None else result
    viewers = []
    ran = False
    original_init, original_run = newton.examples.init, newton.examples.run
    original_gl = newton.viewer.ViewerGL
    initial = ExitStack()
    initial.enter_context(nvtx.annotate("initialization", domain=DOMAIN))
    start = time.perf_counter()

    class ProfileViewer(original_gl):
        def __init__(self, **kwargs):
            kwargs.update(width=options.width, height=options.height, vsync=False)
            super().__init__(**kwargs)

    def initialize(*args, **kwargs):
        instrumentation.install()
        viewer, parsed = original_init(*args, **kwargs)
        viewers.append(viewer)
        return viewer, parsed

    def measured_run(example, args):
        nonlocal ran
        ran = True
        instrumentation.install()
        configure = hooks.get("configure")
        if configure:
            with nvtx.annotate("scenario_configuration", domain=DOMAIN):
                configure(name, example)
        synchronize()
        result["initialization_wall_seconds"] = time.perf_counter() - start
        initial.close()
        run_frames(
            example,
            args,
            name,
            frames,
            hooks,
            nvtx,
            synchronize,
            result=result,
            final_test=not warmup,
        )

    newton.examples.init, newton.examples.run = initialize, measured_run
    newton.viewer.ViewerGL = ProfileViewer
    previous = sys.argv
    sys.argv = [
        module,
        *arguments,
        "--viewer",
        options.viewer,
        "--headless",
        "--num-frames",
        str(frames),
    ]
    try:
        runpy.run_module(module, run_name="__main__")
        if not ran:
            raise RuntimeError("Example did not call newton.examples.run; add a profiling adapter before capturing it")
        return result
    finally:
        initial.close()
        for viewer in viewers:
            viewer.close()
        newton.examples.init, newton.examples.run = original_init, original_run
        newton.viewer.ViewerGL = original_gl
        sys.argv = previous


def run_terminal(module, arguments, nvtx, synchronize):
    """Execute optimization itself; terminal videos contain no simulation frames."""
    previous = sys.argv
    sys.argv = [module, *arguments]
    synchronize()
    start = time.perf_counter()
    try:
        with nvtx.annotate("optimization", domain=DOMAIN):
            runpy.run_module(module, run_name="__main__")
        synchronize()
        return {
            "optimization_wall_seconds": time.perf_counter() - start,
            "frames_completed": None,
        }
    finally:
        sys.argv = previous


def profile(options):
    sys.path.insert(0, str(ROOT / "tools"))
    from build_readme import project

    _, package, examples = project(ROOT)
    if options.example not in examples:
        raise ValueError(f"Unknown example: {options.example}")
    _, terminal, relative_module = examples[options.example]
    profiles = ROOT / "docs" / "videos.json"
    video = json.loads(profiles.read_text()).get(options.example, {}) if profiles.is_file() else {}
    if terminal and options.frames != video.get("frames", 240):
        raise ValueError(
            "--frames does not apply to terminal optimization; keep the video frame count and use -- --iterations N"
        )
    if any(arg.startswith("--benchmark") for arg in options.arguments):
        raise ValueError("Newton --benchmark adds frame synchronization; use the profiler's boundary timing instead")
    arguments = [*video.get("args", []), *options.arguments, "--device", options.device]
    module = f"{package}.examples.{relative_module}"
    import nvtx
    import warp as wp

    device = wp.get_device(options.device)
    if not device.is_cuda:
        raise ValueError("Nsight CUDA profiling requires a CUDA device")

    def synchronize():
        wp.synchronize_device(device)

    importlib.import_module(package)
    instrumentation = Instrumentation(package, nvtx)
    instrumentation.install()
    instrumentation.wrap(wp.Tape, "backward", wp.Tape.backward, "backward:warp.Tape.backward")
    # Public Warp reusable Krylov state is also used directly by FEM.
    from warp.optim.linear import LinearSolverState

    instrumentation.wrap(LinearSolverState, "__call__", LinearSolverState.__call__, "linear_solve:warp.Krylov")
    hooks_path = ROOT / "tools" / "video_scenarios.py"
    hooks = runpy.run_path(str(hooks_path)) if hooks_path.exists() and not terminal else {}
    result = {
        "status": "running",
        "example": options.example,
        "module": module,
        "arguments": arguments,
        "device": options.device,
        "viewer": options.viewer,
        "render_note": "Every example.render call runs, including visualization preparation used by scenario checks. NullViewer omits OpenGL drawing; gl includes drawing without recording or readback.",
        "capture_includes_initialization": True,
        "workload": "terminal_optimization" if terminal else "live_simulation",
        "frames_requested": None if terminal else options.frames,
        "frames_completed": None if terminal else 0,
        "presentation_frames": video.get("frames") if terminal else None,
        "warmup_requested": options.warmup,
        "warmup_unit": "one complete independent optimization if nonzero"
        if terminal
        else "frames in a separate instance",
        "timing_note": "Host phase times are asynchronous submission durations, not GPU durations. Loop wall time includes the final device synchronization. NVTX instrumentation and Nsight add overhead. CUDA graph replay does not re-enter Python method ranges; inspect CUDA graph nodes for kernels.",
        "scenario_note": "Fresh initialization after warmup preserves video frame-zero state and every scripted interaction. First use of later cutting/contact branches can still compile kernels during capture.",
    }
    options.output_dir.mkdir(parents=True, exist_ok=True)
    try:
        if options.warmup:
            with nvtx.annotate("warmup", domain=DOMAIN):
                result["warmup"] = (
                    run_terminal(module, arguments, nvtx, synchronize)
                    if terminal
                    else run_visual(
                        module,
                        arguments,
                        options.example,
                        options,
                        hooks,
                        nvtx,
                        synchronize,
                        instrumentation,
                        warmup=True,
                    )
                )
        # runpy module/class cycles may otherwise keep warmup GPU buffers alive.
        gc.collect()
        synchronize()
        with nvtx.annotate("profile", domain=DOMAIN):
            captured = (
                run_terminal(module, arguments, nvtx, synchronize)
                if terminal
                else run_visual(
                    module,
                    arguments,
                    options.example,
                    options,
                    hooks,
                    nvtx,
                    synchronize,
                    instrumentation,
                    warmup=False,
                    result=result,
                )
            )
            result.update(captured)
            synchronize()
        result["status"] = "completed"
    except BaseException as error:
        result["status"] = "failed"
        result["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        result["instrumented_host_methods"] = sorted(set(instrumentation.labels))
        instrumentation.restore()
        (options.output_dir / "workload.json").write_text(json.dumps(result, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("example")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--frames", type=int, required=True)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--viewer", choices=("null", "gl"), default="null")
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    options, arguments = parser.parse_known_args()
    options.arguments = arguments[1:] if arguments[:1] == ["--"] else arguments
    if options.frames < 1 or options.warmup < 0 or min(options.width, options.height) < 1:
        parser.error("frames and dimensions must be positive; warmup must be nonnegative")
    if not options.device.startswith("cuda"):
        parser.error("Nsight CUDA profiling requires a CUDA device, e.g. cuda:0")
    options.output_dir = options.output_dir.resolve()
    os.chdir(ROOT)
    profile(options)


if __name__ == "__main__":
    main()
