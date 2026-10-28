"""Profiling discovery, scope, reports and failure history without CUDA or Nsight."""

import argparse
import csv
import importlib.util
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "nsys"))
try:
    spec = importlib.util.spec_from_file_location("profile_examples", ROOT / "nsys/profile_examples.py")
    profiling = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(profiling)
    import report
finally:
    sys.path.pop(0)


def options(directory):
    return argparse.Namespace(
        device="cuda:0",
        frames=None,
        warmup=5,
        viewer="null",
        width=960,
        height=540,
        include_startup=False,
        cpu_sampling=False,
        cuda_graph_trace=None,
        arguments=[],
        output_dir=Path(directory),
        timeout=10,
        label="test",
        compare=None,
    )


class TestProfiling(unittest.TestCase):
    def test_input_changes_are_detected_with_either_argument_syntax(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "weights with spaces.json"
            source.write_text("first controller")
            before = profiling.input_fingerprints(["--load", source.name], root)
            self.assertEqual(before, profiling.input_fingerprints([f"--load={source.name}"], root))
            self.assertEqual(before[source.name], profiling.digest(source))
            source.write_text("different controller")
            self.assertNotEqual(before, profiling.input_fingerprints([f"--load={source.name}"], root))

    def test_only_discovered_video_examples_and_recorder_flags(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "tools").mkdir()
            shutil.copyfile(ROOT / "tools/build_readme.py", root / "tools/build_readme.py")
            package = "newton_solver_check"
            examples = root / "src" / package / "examples"
            examples.mkdir(parents=True)
            import tomllib

            actual = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["name"].replace("-", "_")
            shutil.copyfile(ROOT / "src" / actual / "examples/__main__.py", examples / "__main__.py")
            (root / "pyproject.toml").write_text('[project]\nname="newton-solver-check"\n')
            for name in ("recorded", "unrecorded"):
                (examples / f"example_{name}.py").write_text(
                    '"""A scene."""\nraise RuntimeError("discovery must not import scenes")\n'
                    'if __name__ == "__main__":\n    pass\n'
                )
            media = root / "docs/media"
            media.mkdir(parents=True)
            (media / "recorded.mp4").touch()
            (media / "recorded.gif").touch()
            (media / "orphan.mp4").touch()
            (media / "orphan.gif").touch()
            (media / "recorded.json").write_text(
                json.dumps(
                    {
                        "frames": 30,
                        "arguments": [
                            "--test",
                            "--device",
                            "cuda:0",
                            "--viewer=gl",
                            "--headless",
                            "--num-frames",
                            "30",
                        ],
                    }
                )
            )
            config = root / "docs/videos.json"
            config.write_text(json.dumps({"recorded": {"frames": 30, "args": ["--test"]}}))
            _, entries = profiling.video_examples(root)
            self.assertEqual(list(entries), ["recorded"])
            self.assertEqual(entries["recorded"]["arguments"], ["--test"])
            config.write_text(json.dumps({"recorded": {"frames": 30, "args": ["--different"]}}))
            with self.assertRaisesRegex(ValueError, "video arguments differ"):
                profiling.video_examples(root)

    def test_capture_scope_and_literal_example_arguments(self):
        opts = options("/tmp/not-created")
        opts.arguments = ["--load", "a path with spaces.json"]
        command = profiling.profile_command("/path to/nsys", "scene", {"frames": 540}, opts, opts.output_dir)
        self.assertIn("--cuda-graph-trace=node", command)
        self.assertIn("--capture-range-end=stop", command)
        self.assertIn("--sample=none", command)
        self.assertEqual(command[command.index("--frames") + 1], "540")
        self.assertEqual(command[-3:], ["--", *opts.arguments])
        command = profiling.profile_command(
            "nsys", "scene", {"frames": 540, "cuda_graph_trace": "graph"}, opts, opts.output_dir
        )
        self.assertIn("--cuda-graph-trace=graph", command)
        opts.cuda_graph_trace = "node"
        command = profiling.profile_command(
            "nsys", "scene", {"frames": 540, "cuda_graph_trace": "graph"}, opts, opts.output_dir
        )
        self.assertIn("--cuda-graph-trace=node", command)
        opts.include_startup, opts.viewer = True, "gl"
        command = profiling.profile_command("nsys", "scene", {"frames": 540}, opts, opts.output_dir)
        self.assertFalse(any(arg.startswith("--capture-range") for arg in command))
        self.assertTrue(any(arg.startswith("--trace=") and "opengl" in arg for arg in command))

    def test_failed_capture_retains_history_and_error(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fake = root / "nsys-fail"
            fake.write_text('#!/bin/sh\necho "capture failed"\nexit 7\n')
            fake.chmod(0o755)
            entry = {"frames": 1, "module": "scene", "terminal": False, "arguments": [], "config": {}}
            with (
                patch.object(profiling, "source_state", return_value=({"commit": "abc", "dirty": False}, "digest")),
                patch.object(profiling, "query", return_value="test"),
            ):
                self.assertFalse(
                    profiling.capture("scene", entry, "missing-package", options(root / "results"), str(fake))
                )
            history = json.loads((root / "results/history.jsonl").read_text())
            run = root / "results" / history["directory"]
            metadata = json.loads((run / "run.json").read_text())
            self.assertEqual(metadata["capture_exit_code"], 7)
            self.assertEqual(metadata["status"], "failed")
            self.assertIn("capture failed", (run / "capture.log").read_text())
            self.assertTrue((run / "summary.md").is_file())
            self.assertEqual(json.loads((run / "summary.json").read_text())["summed_event_ms"], {})

    def test_comparison_requires_same_runner_and_collection_but_allows_solver_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runner = root / "nsys/runner.py"
            runner.parent.mkdir()
            runner.write_text("initial instrumentation")
            opts = options(root / "results")
            entry = {"frames": 1, "module": "scene", "terminal": False, "arguments": [], "config": {}}
            command_builder = profiling.profile_command

            def complete_without_gpu(command, log_path, timeout):
                output = Path(command[command.index("--output") + 1]).parent
                if command[1] == "profile":
                    (output / "capture.nsys-rep").touch()
                    (output / "workload.json").write_text(json.dumps({"status": "completed", "frames_completed": 1}))
                else:
                    (output / "stats_cuda_gpu_kern_sum.csv").write_text("Total Time (ns),Name\n1000000,kernel\n")
                return 0

            def capture(revision, *, frequency="200"):
                def build_command(*args):
                    return [
                        f"--python-sampling-frequency={frequency}"
                        if item == "--python-sampling-frequency=200"
                        else item
                        for item in command_builder(*args)
                    ]

                with (
                    patch.object(profiling, "ROOT", root),
                    patch.object(
                        profiling, "source_state", return_value=({"commit": revision, "dirty": False}, revision)
                    ),
                    patch.object(profiling, "query", return_value="unchanged environment"),
                    patch.object(profiling, "profile_command", side_effect=build_command),
                    patch.object(profiling, "run_logged", side_effect=complete_without_gpu),
                ):
                    self.assertTrue(profiling.capture("scene", entry, "missing-package", opts, "nsys"))
                history = json.loads((opts.output_dir / "history.jsonl").read_text().splitlines()[-1])
                return opts.output_dir / history["directory"]

            baseline = capture("solver revision one")
            changed_solver = capture("solver revision two")
            report.compare(changed_solver, baseline)
            self.assertTrue((changed_solver / "comparison.md").is_file())
            runner.write_text("different instrumentation")
            changed_runner = capture("solver revision two")
            with self.assertRaisesRegex(ValueError, "not comparable"):
                report.compare(changed_runner, baseline)
            runner.write_text("initial instrumentation")
            changed_flags = capture("solver revision two", frequency="100")
            with self.assertRaisesRegex(ValueError, "not comparable"):
                report.compare(changed_flags, baseline)
            metadata = json.loads((changed_flags / "run.json").read_text())
            self.assertEqual(metadata["profiling"]["runner_sha256"], profiling.digest(runner))
            self.assertEqual(metadata["profiling"]["collection_settings"]["--python-sampling-frequency"], "100")

    def test_collection_settings_ignore_locations_order_and_option_syntax(self):
        first = [
            "nsys",
            "profile",
            "--trace=cuda,nvtx",
            "--output",
            "/first",
            "--sample=none",
            sys.executable,
            "runner",
        ]
        second = [
            "/other/nsys",
            "profile",
            "--sample",
            "none",
            "--output=/second",
            "--trace",
            "cuda,nvtx",
            sys.executable,
            "other-runner",
        ]
        self.assertEqual(profiling.collection_settings(first), profiling.collection_settings(second))

    def test_csv_units_summary_size_and_comparison_compatibility(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            before, after = root / "before", root / "after"
            for path, ns in ((before, 2000000), (after, 1000000)):
                path.mkdir()
                with (path / "stats_cuda_gpu_kern_sum.csv").open("w", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=["Time (%)", "Total Time (ns)", "Instances", "Name"])
                    writer.writeheader()
                    writer.writerow(
                        {"Time (%)": 100, "Total Time (ns)": ns, "Instances": 2, "Name": "kernel, with comma"}
                    )
                (path / "workload.json").write_text(
                    json.dumps(
                        {
                            "frames_completed": 960,
                            "frame_host_timings": [{"frame": i, "step_host_seconds": 0.01} for i in range(960)],
                        }
                    )
                )
                metadata = {
                    "example": "scene",
                    "status": "complete",
                    "git": {"commit": "abc"},
                    "source_sha256": "abc",
                    "options": {"viewer": "null"},
                    "comparison_key": "same",
                }
                report.write_summary(path, metadata)
                self.assertLess(len((path / "summary.md").read_text().splitlines()), 120)
            self.assertEqual(report.totals(report.read_reports(after))["cuda_gpu_kern_sum"], 1.0)
            report.compare(after, before)
            self.assertIn("-50.00", (after / "comparison.md").read_text())
            summary = json.loads((after / "summary.json").read_text())
            summary["run"]["comparison_key"] = "different"
            (after / "summary.json").write_text(json.dumps(summary))
            with self.assertRaisesRegex(ValueError, "not comparable"):
                report.compare(after, before)


if __name__ == "__main__":
    unittest.main()
