"""Profile the live workloads demonstrated in this repository's README videos."""

import argparse
import hashlib
import json
import os
import platform
import re
import runpy
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from datetime import UTC, datetime
from importlib import metadata as packages
from pathlib import Path

from report import (
    REPORTS,
    compare,
    export_graph_summary,
    inspect_graph_coverage,
    read_reports,
    validate_graph_coverage,
    write_summary,
)

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def scene_arguments(arguments):
    """Remove flags the video recorder appends; the profiler owns these controls."""
    result = []
    index = 0
    while index < len(arguments):
        argument = arguments[index]
        key = argument.split("=", 1)[0]
        if key in {"--device", "--viewer", "--num-frames"}:
            index += 1 if "=" in argument else 2
        elif key == "--headless":
            index += 1
        else:
            result.append(argument)
            index += 1
    return result


def input_fingerprints(arguments, root=ROOT):
    fingerprints = {}
    for argument in arguments:
        if argument.startswith("--") and "=" in argument:
            argument = argument.split("=", 1)[1]
        elif argument.startswith("-"):
            continue
        path = root / Path(argument).expanduser()
        try:
            is_file = path.is_file()
        except OSError:
            # An inline JSON value or other long option value is not a path.
            continue
        if is_file:
            fingerprints[argument] = digest(path)
    return fingerprints


def video_examples(root=ROOT):
    project = runpy.run_path(str(root / "tools/build_readme.py"))["project"]
    _, package, examples = project(root)
    config_path = root / "docs/videos.json"
    configs = json.loads(config_path.read_text()) if config_path.is_file() else {}
    profiling_path = root / "nsys/config.json"
    profiling = json.loads(profiling_path.read_text()) if profiling_path.is_file() else {}
    selected = {}
    for name, (description, terminal, module) in examples.items():
        media = root / "docs/media"
        # Exactly the condition build_readme uses to display a playable preview.
        if not (media / f"{name}.mp4").is_file() or not any(
            (media / f"{name}{ext}").is_file() for ext in (".gif", ".jpg")
        ):
            continue
        sidecar = media / f"{name}.json"
        recorded = json.loads(sidecar.read_text()) if sidecar.is_file() else {}
        config = configs.get(name, {})
        # The sidecar describes the clip actually displayed; config also holds
        # current scripted controls and is checked against it before running.
        arguments = scene_arguments(recorded.get("arguments", config.get("args", [])))
        if recorded and arguments != scene_arguments(config.get("args", [])):
            raise ValueError(f"{name}: video arguments differ from docs/videos.json; refresh the video metadata first")
        frames = recorded.get("frames", config.get("frames", 240))
        if recorded and frames != config.get("frames", frames):
            raise ValueError(f"{name}: video frame count differs from docs/videos.json")
        selected[name] = {
            "description": description,
            "module": f"{package}.examples.{module}",
            "terminal": terminal,
            "frames": frames,
            "arguments": arguments,
            "config": config,
            "sidecar": recorded,
            "cuda_graph_trace": profiling.get(name, {}).get("cuda_graph_trace", "node"),
            "minimum_nsys_version": profiling.get(name, {}).get("minimum_nsys_version"),
            "nsys_config_directives": profiling.get(name, {}).get("nsys_config_directives", ""),
        }
        if selected[name]["cuda_graph_trace"] not in {"node", "graph"}:
            raise ValueError(f"{name}: nsys/config.json cuda_graph_trace must be node or graph")
    return package, selected


def query(command, cwd=ROOT):
    try:
        result = subprocess.run(command, cwd=cwd, capture_output=True, text=True, timeout=20)
        return result.stdout.strip() if result.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        return None


def check_profiler_version(version, examples):
    """Reject known incompatible collectors before starting expensive workloads."""
    match = re.search(r"\b(\d{4})\.(\d+)\.(\d+)\b", version or "")
    actual = tuple(map(int, match.groups())) if match else ()
    for name, entry in examples.items():
        minimum = entry.get("minimum_nsys_version")
        if minimum and actual < tuple(map(int, minimum.split("."))):
            raise ValueError(
                f"{name} requires Nsight Systems >= {minimum} for profiling this workload; "
                f"found {version!r}. Install a current CLI and select it with NSYS_ROOT (Docker) "
                "or NSYS_PATH (local)."
            )


def source_state(root=ROOT):
    commit = query(["git", "rev-parse", "HEAD"], root)
    status = query(["git", "status", "--porcelain"], root)
    files = query(["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"], root)
    checksum = hashlib.sha256()
    if files is None:
        raise ValueError("Git metadata is unavailable; profile from a Git checkout with git installed")
    for name in sorted(set(files.split("\0")) - {""}):
        path = root / name
        checksum.update(name.encode())
        if path.is_symlink():
            checksum.update(str(path.readlink()).encode())
        elif path.is_file():
            checksum.update(bytes.fromhex(digest(path)))
        else:
            checksum.update(b"deleted")
    return {"commit": commit, "dirty": bool(status)}, checksum.hexdigest()


def profile_command(nsys, name, entry, options, output):
    command = [
        nsys,
        "profile",
        "--trace="
        + getattr(options, "cuda_trace", "cuda")
        + ",nvtx,osrt,cusparse,cusolver"
        + (",opengl" if options.viewer == "gl" else ""),
        "--sample=process-tree" if options.cpu_sampling else "--sample=none",
        "--cpuctxsw=process-tree" if options.cpu_sampling else "--cpuctxsw=none",
        "--python-sampling=true",
        "--python-sampling-frequency=200",
        "--cuda-graph-trace=" + (options.cuda_graph_trace or entry.get("cuda_graph_trace", "node")),
        "--cuda-memory-usage=true",
        "--force-overwrite=false",
        "--output",
        str(output / "capture"),
    ]
    if not options.include_startup:
        command += ["--capture-range=nvtx", "--nvtx-capture=profile@newton-solver", "--capture-range-end=stop"]
    command += [
        sys.executable,
        "-X",
        "faulthandler",
        str(ROOT / "nsys/runner.py"),
        name,
        "--output-dir",
        str(output),
        "--device",
        options.device,
        "--frames",
        str(options.frames or entry["frames"]),
        "--warmup",
        str(options.warmup),
        "--viewer",
        options.viewer,
        "--width",
        str(options.width),
        "--height",
        str(options.height),
    ]
    if options.arguments:
        command += ["--", *options.arguments]
    return command


def collection_settings(command):
    """Normalize Nsight flags independently of executable and output locations."""
    arguments = iter(command[2 : command.index(sys.executable, 2)])
    settings = {}
    for argument in arguments:
        name, separator, value = argument.partition("=")
        if not separator:
            value = next(arguments)
        if name != "--output":
            settings[name] = value
    return settings


def run_logged(command, log_path, timeout, *, env=None):
    # Leave the example's full output in the run folder, without buffering an
    # unbounded training log in RAM. A process group also cleans up on timeout.
    with log_path.open("w") as stream:
        process = subprocess.Popen(
            command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True, env=env
        )
        try:
            return process.wait(timeout=timeout)
        except (subprocess.TimeoutExpired, KeyboardInterrupt):
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            raise


def capture(name, entry, package, options, nsys):
    options.output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
    output = Path(tempfile.mkdtemp(prefix=f"{stamp}_{name}_", dir=options.output_dir))
    git, source_hash = source_state()
    hardware = query(["nvidia-smi", "--query-gpu=index,name,driver_version,memory.total", "--format=csv,noheader"])
    command = profile_command(nsys, name, entry, options, output)
    dependencies = {}
    for dependency in (package, "newton", "warp-lang", "numpy", "nvtx"):
        try:
            dependencies[dependency] = packages.version(dependency)
        except packages.PackageNotFoundError:
            dependencies[dependency] = None
    input_hashes = input_fingerprints((*entry["arguments"], *options.arguments))
    environment = os.environ.copy()
    directives = environment.get("NSYS_CONFIG_DIRECTIVES", entry.get("nsys_config_directives", ""))
    if directives:
        environment["NSYS_CONFIG_DIRECTIVES"] = directives
    settings = {
        key: getattr(options, key)
        for key in ("device", "viewer", "width", "height", "warmup", "include_startup", "cpu_sampling", "arguments")
    }
    settings["frames"] = options.frames or entry["frames"]
    settings["cuda_graph_trace"] = options.cuda_graph_trace or entry.get("cuda_graph_trace", "node")
    scenario_path = ROOT / "tools/video_scenarios.py"
    profiling = {
        "runner_sha256": digest(ROOT / "nsys/runner.py"),
        "collection_settings": collection_settings(command),
        "nsys_config_directives": directives,
    }
    comparable = {
        "example": name,
        "settings": settings,
        "profiling": profiling,
        "video_config": entry["config"],
        "scenario_sha256": digest(scenario_path) if scenario_path.is_file() else None,
        "input_sha256": input_hashes,
        "gpu": hardware,
        "nsys": query([nsys, "--version"]),
        "dependencies": dependencies,
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    metadata = {
        "schema_version": 1,
        "example": name,
        "module": entry["module"],
        "started_utc": stamp,
        "status": "running",
        "label": options.label,
        "git": git,
        "source_sha256": source_hash,
        "options": settings,
        "profiling": profiling,
        "video": entry,
        "input_sha256": input_hashes,
        "command": command,
        "gpu": hardware,
        "nsys_version": comparable["nsys"],
        "python": sys.version,
        "platform": platform.platform(),
        "dependencies": dependencies,
        "comparison_key": hashlib.sha256(json.dumps(comparable, sort_keys=True).encode()).hexdigest(),
    }
    manifest = output / "run.json"
    manifest.write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Profiling {name}: {output}\n  Full output: {output / 'capture.log'}", flush=True)
    start = time.perf_counter()
    interrupted = False
    try:
        code = run_logged(command, output / "capture.log", options.timeout, env=environment)
        metadata["capture_exit_code"] = code
        if code:
            raise RuntimeError(f"Nsight/example exited with code {code}; see capture.log")
        report = output / "capture.nsys-rep"
        if not report.is_file():
            raise RuntimeError("No Nsight report produced; the capture range may not have been reached")
        workload = output / "workload.json"
        if not workload.is_file():
            raise RuntimeError("The example did not write its workload completion metadata")
        completed = json.loads(workload.read_text())
        if completed.get("status") != "completed":
            raise RuntimeError("The workload failed or was interrupted; see workload.json")
        if not entry["terminal"] and completed.get("frames_completed") != settings["frames"]:
            raise RuntimeError("The workload did not complete the requested simulation frames")
        # Let Nsight maintain its own evolving SQLite schema and CSV analysis.
        stats_command = [
            nsys,
            "stats",
            "--force-export=true",
            "--format=csv",
            "--report",
            ",".join(REPORTS),
            "--output",
            str(output / "stats"),
            str(report),
        ]
        metadata["stats_command"] = stats_command
        metadata["stats_exit_code"] = run_logged(stats_command, output / "stats.log", options.timeout)
        if metadata["stats_exit_code"]:
            raise RuntimeError("Nsight statistics failed; see stats.log (the raw capture is retained)")
        coverage = inspect_graph_coverage(output)
        metadata["profiling"]["graph_coverage"] = coverage
        validate_graph_coverage(coverage, settings["cuda_graph_trace"])
        export_graph_summary(output)
        tables = read_reports(output)
        if not (tables.get("cuda_gpu_kern_sum") or tables.get("cuda_graph_sum")):
            raise RuntimeError(
                "No CUDA kernel or graph events were recorded; this is not a valid GPU performance result"
            )
        metadata["status"] = "complete"
    except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired, KeyboardInterrupt) as error:
        metadata["status"] = "failed"
        metadata["error"] = str(error) or type(error).__name__
        interrupted = isinstance(error, KeyboardInterrupt)
    finally:
        metadata["wall_seconds_including_profiler_and_analysis"] = time.perf_counter() - start
        metadata["finished_utc"] = datetime.now(UTC).isoformat()
        manifest.write_text(json.dumps(metadata, indent=2) + "\n")
        write_summary(output, metadata)
        with (options.output_dir / "history.jsonl").open("a") as stream:
            stream.write(
                json.dumps(
                    {
                        "directory": output.name,
                        "example": name,
                        "status": metadata["status"],
                        "commit": git["commit"],
                        "source_sha256": source_hash,
                        "started_utc": stamp,
                        "label": options.label,
                    }
                )
                + "\n"
            )
    if interrupted:
        raise KeyboardInterrupt
    if options.compare and metadata["status"] == "complete":
        compare(output, options.compare)
    print(f"{name}: {metadata['status']}; {output / 'summary.md'}", flush=True)
    return metadata["status"] == "complete"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("example", nargs="?")
    parser.add_argument("--all", action="store_true", help="Profile every example with a README video, sequentially.")
    parser.add_argument("--list", action="store_true", help="List only examples with playable README previews.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without requiring CUDA, nvtx or nsys.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--frames", type=int, help="Override video frame count; a short run may omit scenario events.")
    parser.add_argument(
        "--warmup", type=int, default=5, help="Independent warmup frames before a fresh scenario (default 5)."
    )
    parser.add_argument(
        "--viewer", choices=("null", "gl"), default="null", help="null: physics; gl: physics and rendering."
    )
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument(
        "--include-startup", action="store_true", help="Also trace imports and the independent warmup run."
    )
    parser.add_argument(
        "--cpu-sampling",
        action="store_true",
        help="Add native CPU sampling/context switches; requires perf permissions.",
    )
    parser.add_argument(
        "--cuda-graph-trace",
        choices=("node", "graph"),
        help="Trace individual graph nodes or whole graphs; overrides nsys/config.json (otherwise node).",
    )
    parser.add_argument("--label", default="", help="A note attached to this capture in local history.")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "nsys/results")
    parser.add_argument("--timeout", type=float, default=1800, help="Maximum seconds per capture/statistics command.")
    parser.add_argument(
        "--compare", type=Path, help="Compare against a previous complete run with matching workload/settings."
    )
    args = sys.argv[1:]
    separator = args.index("--") if "--" in args else len(args)
    options = parser.parse_args(args[:separator])
    options.arguments = args[separator + 1 :]
    try:
        package, examples = video_examples()
        if options.list:
            for name, entry in examples.items():
                unit = "presentation frames; profile actual optimization" if entry["terminal"] else "simulation frames"
                print(f"{name}: {entry['frames']} {unit} — {entry['description']}")
            return
        if (options.all and options.example) or (not options.all and options.example not in examples):
            parser.error("choose a video example from --list, or --all")
        if options.compare and options.all:
            parser.error("--compare applies to one example at a time")
        if not options.device.startswith("cuda"):
            parser.error("Nsight GPU profiling requires a CUDA device")
        if (
            options.warmup < 0
            or (options.frames is not None and options.frames < 1)
            or min(options.width, options.height, options.timeout) <= 0
        ):
            parser.error("frames, dimensions and timeout must be positive; warmup must be nonnegative")
        selected = list(examples) if options.all else [options.example]
        if not selected:
            parser.error("no README videos exist yet; record examples with tools/record_examples.py first")
        options.output_dir = options.output_dir.expanduser().resolve()
        nsys = os.environ.get("NSYS_PATH") or shutil.which("nsys")
        if options.dry_run:
            for name in selected:
                command = profile_command(
                    nsys or "nsys", name, examples[name], options, options.output_dir / f"RUN_{name}"
                )
                print(
                    json.dumps(
                        {"example": name, "video_arguments": examples[name]["arguments"], "command": command}, indent=2
                    )
                )
            return
        if not nsys:
            parser.error(
                "nsys is not installed; install NVIDIA Nsight Systems or use docker/run.sh --nsys with NSYS_ROOT"
            )
        version = query([nsys, "--version"])
        if not version:
            parser.error("Cannot execute nsys; check NSYS_PATH")
        check_profiler_version(version, {name: examples[name] for name in selected})
        # Recent Nsight releases default to hardware tracing on Blackwell. Use
        # software tracing explicitly: HES can omit graph kernels on GB10/R580.
        # Older releases use software tracing for the existing 'cuda' selector.
        help_text = query([nsys, "profile", "--help"]) or ""
        options.cuda_trace = "cuda-sw" if "cuda-sw" in help_text else "cuda"
        try:
            packages.version("nvtx")
        except packages.PackageNotFoundError:
            parser.error("Install optional profiling dependencies: uv sync --locked --extra examples --extra profiling")
        success = True
        for name in selected:
            success = capture(name, examples[name], package, options, nsys) and success
        if not success:
            raise SystemExit(1)
    except (OSError, ValueError) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()
