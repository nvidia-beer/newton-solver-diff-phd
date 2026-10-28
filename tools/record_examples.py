"""Record one representative clip per discovered example, using its real simulation."""

import argparse
import json
import os
import runpy
import shutil
import subprocess
import sys
import time
from importlib.metadata import version
from pathlib import Path

from build_readme import ROOT, project


class Clip:
    def __init__(self, path, fps):
        self.path, self.fps = path, fps
        self.process = None
        self.count = 0
        self.last_image = None

    def write(self, pixels):
        import numpy as np

        pixels = np.ascontiguousarray(pixels, dtype=np.uint8)
        if pixels.ndim != 3 or pixels.shape[2] != 3:
            raise ValueError("Expected an RGB frame")
        if self.process is None:
            height, width, _ = pixels.shape
            self.process = subprocess.Popen(
                [
                    "ffmpeg",
                    "-y",
                    "-loglevel",
                    "error",
                    "-f",
                    "rawvideo",
                    "-pixel_format",
                    "rgb24",
                    "-video_size",
                    f"{width}x{height}",
                    "-framerate",
                    str(self.fps),
                    "-i",
                    "-",
                    "-an",
                    "-c:v",
                    "libx264",
                    "-threads",
                    "2",
                    "-crf",
                    "22",
                    "-pix_fmt",
                    "yuv420p",
                    "-movflags",
                    "+faststart",
                    str(self.path),
                ],
                stdin=subprocess.PIPE,
            )
        self.process.stdin.write(pixels.tobytes())
        self.count += 1
        self.last_image = pixels

    def close(self):
        if self.process is None:
            raise RuntimeError("The example produced no frames")
        self.process.stdin.close()
        if self.process.wait() != 0:
            raise RuntimeError("ffmpeg could not encode the clip")


def record(name, options):
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont

    _, package, examples = project()
    description, headless, relative_module = examples[name]
    profiles = ROOT / "docs" / "videos.json"
    profile = json.loads(profiles.read_text()).get(name, {}) if profiles.exists() else {}
    frames = options.frames or profile.get("frames", 240)
    fps = options.fps or profile.get("fps", 60)
    output = options.output_dir
    output.mkdir(parents=True, exist_ok=True)
    final = output / f"{name}.mp4"
    temporary = output / f".{name}.tmp.mp4"
    hooks_path = ROOT / "tools" / "video_scenarios.py"
    hooks = runpy.run_path(str(hooks_path)) if hooks_path.exists() else {}
    arguments = profile.get("args", []) + options.arguments
    caption = profile.get("caption", description)
    writer = Clip(temporary, fps)
    start = time.perf_counter()
    simulation_times = []

    def write_frame(pixels, label=""):
        picture = Image.fromarray(pixels)
        draw = ImageDraw.Draw(picture)
        draw.rectangle((0, 0, picture.width, 29), fill=(25, 28, 34))
        draw.text(
            (12, 8),
            name.replace("_", " ") + ("  |  " + label if label else ""),
            fill="white",
            font=ImageFont.load_default(size=16),
        )
        writer.write(np.asarray(picture))

    if headless:
        callback = hooks.get("record_terminal")
        if callback is None:
            raise ValueError(f"{name} is terminal-only; provide record_terminal in tools/video_scenarios.py")
        for pixels in callback(name, options.device, frames, options.width, options.height):
            write_frame(pixels, "optimization progress")
    else:
        import newton.examples
        import newton.viewer

        original_init, original_run = newton.examples.init, newton.examples.run
        original_gl = newton.viewer.ViewerGL

        class CaptureViewer(original_gl):
            def __init__(self, **kwargs):
                kwargs.update(width=options.width, height=options.height, vsync=False)
                super().__init__(**kwargs)

        def capture_run(example, args):
            configure = hooks.get("configure")
            if configure:
                configure(name, example)
            before_frame = hooks.get("before_frame")
            label = profile.get("label", "simulation playback")
            viewer = example.viewer
            try:
                for frame in range(frames):
                    if before_frame:
                        before_frame(name, example, frame)
                    example.step()
                    example.render()
                    frame_label = hooks.get("frame_label")
                    write_frame(viewer.get_frame().numpy(), frame_label(name, example) if frame_label else label)
                    if frame % max(1, frames // 8) == 0:
                        print(f"{name}: frame {frame + 1}/{frames}", flush=True)
                if args.test and hasattr(example, "test_final"):
                    example.test_final()
                for attribute in ("sim_time", "time", "t"):
                    value = getattr(example, attribute, None)
                    if isinstance(value, (int, float)):
                        simulation_times.append({"attribute": attribute, "end": value})
                        break
            finally:
                viewer.close()

        newton.viewer.ViewerGL = CaptureViewer
        newton.examples.run = capture_run
        module = f"{package}.examples.{relative_module}"
        sys.argv = [
            module,
            *arguments,
            "--device",
            options.device,
            "--viewer",
            "gl",
            "--headless",
            "--num-frames",
            str(frames),
        ]
        try:
            runpy.run_module(module, run_name="__main__")
        finally:
            newton.examples.init, newton.examples.run = original_init, original_run
            newton.viewer.ViewerGL = original_gl
    writer.close()
    temporary.replace(final)
    Image.fromarray(writer.last_image).save(output / f"{name}.jpg", quality=88)
    # GitHub README supports animated images; the preview links to the full MP4.
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-threads",
            "2",
            "-filter_complex_threads",
            "1",
            "-i",
            str(final),
            "-filter_complex",
            "fps=12,scale=480:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=128[p];[b][p]paletteuse=dither=bayer",
            str(output / f"{name}.gif"),
        ],
        check=True,
    )
    metadata = {
        "example": name,
        "module": f"{package}.examples.{relative_module}",
        "caption": caption,
        "arguments": arguments,
        "device": options.device,
        "frames": writer.count,
        "fps": fps,
        "width": options.width,
        "height": options.height,
        "recording_wall_seconds": time.perf_counter() - start,
        "playback_is_performance_benchmark": False,
        "simulation_time": simulation_times,
        "dependencies": {package: version(package), "newton": version("newton"), "warp-lang": version("warp-lang")},
    }
    (output / f"{name}.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Recorded {final}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("example", nargs="?")
    parser.add_argument("--all", action="store_true", help="Capture every automatically discovered example.")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--frames", type=int)
    parser.add_argument("--fps", type=int)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "docs" / "media")
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    options, arguments = parser.parse_known_args()
    options.arguments = arguments[1:] if arguments[:1] == ["--"] else arguments
    _, _, examples = project()
    if options.list:
        print("\n".join(examples))
        return
    if (not options.all and options.example not in examples) or (options.all and options.example):
        parser.error("Choose --all or one discovered example; use --list to see names")
    if any(value is not None and value < 1 for value in (options.frames, options.fps, options.width, options.height)):
        parser.error("Capture dimensions, frame count and fps must be positive")
    if options.width % 2 or options.height % 2:
        parser.error("MP4 dimensions must be even")
    if not shutil.which("ffmpeg"):
        parser.error("Install ffmpeg to encode example videos")
    if options.child:
        record(options.example, options)
        return
    failures = []
    for name in examples if options.all else [options.example]:
        if options.skip_existing and (options.output_dir / f"{name}.mp4").is_file():
            continue
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            name,
            "--child",
            "--device",
            options.device,
            "--width",
            str(options.width),
            "--height",
            str(options.height),
            "--output-dir",
            str(options.output_dir),
        ]
        for argument, value in (("--frames", options.frames), ("--fps", options.fps)):
            if value is not None:
                command.extend([argument, str(value)])
        command += ["--", *options.arguments] if options.arguments else []
        environment = os.environ.copy()
        environment["PYTHONUNBUFFERED"] = "1"
        result = subprocess.run(command, cwd=ROOT, env=environment)
        if result.returncode:
            failures.append(name)
    if failures:
        parser.exit(1, f"Failed recordings: {', '.join(failures)}\n")
    subprocess.run([sys.executable, str(ROOT / "tools" / "build_readme.py")], check=True)


if __name__ == "__main__":
    main()
