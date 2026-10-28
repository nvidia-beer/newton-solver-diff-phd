"""Check profiling isolation, frame fidelity and asynchronous timing boundaries."""

import importlib.util
import sys
import types
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location("nsys_runner", Path(__file__).resolve().parents[1] / "nsys" / "runner.py")
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


class Trace:
    def __init__(self):
        self.events = []

    @contextmanager
    def annotate(self, label, **kwargs):
        self.events.append(("begin", label))
        try:
            yield
        finally:
            self.events.append(("end", label))


class TestNsightRunner(unittest.TestCase):
    def test_wrappers_preserve_descriptors_return_values_and_restore(self):
        module = types.ModuleType("profiling_fixture")

        class SparseSolve:
            def solve(self, value, *, scale=2):
                return value * scale

            @staticmethod
            def factor(value):
                return value + 1

            @classmethod
            def refactor(cls, value):
                return cls.__name__, value

        SparseSolve.__module__ = module.__name__
        module.SparseSolve = SparseSolve
        original = dict(vars(module.SparseSolve))
        trace = Trace()
        instrument = RUNNER.Instrumentation(module.__name__, trace)
        with patch.dict(sys.modules, {module.__name__: module}):
            instrument.install()
            instrument.install()  # Re-discovery must not add duplicate wrappers.
            self.assertEqual(module.SparseSolve().solve(3, scale=4), 12)
            self.assertEqual(module.SparseSolve.factor(3), 4)
            self.assertEqual(module.SparseSolve.refactor(5), ("SparseSolve", 5))
            self.assertEqual(len(instrument.labels), 3)
            instrument.restore()
        for name in ("solve", "factor", "refactor"):
            self.assertIs(vars(module.SparseSolve)[name], original[name])
        self.assertIn(("begin", "linear_solve:SparseSolve.solve"), trace.events)

    def test_loop_keeps_video_frame_order_and_only_boundary_syncs(self):
        calls = []
        example = types.SimpleNamespace(
            step=lambda: calls.append("step"),
            test_post_step=lambda: calls.append("test"),
            render=lambda: calls.append("render"),
            test_final=lambda: calls.append("final"),
        )
        result = RUNNER.run_frames(
            example,
            types.SimpleNamespace(test=True),
            "demo",
            3,
            {"before_frame": lambda name, obj, frame: calls.append((name, frame))},
            Trace(),
            lambda: calls.append("sync"),
        )
        self.assertEqual(
            calls,
            [
                "sync",
                *(item for frame in range(3) for item in (("demo", frame), "step", "test", "render")),
                "sync",
                "final",
            ],
        )
        self.assertEqual(result["frames_completed"], 3)
        self.assertEqual([row["frame"] for row in result["frame_host_timings"]], [0, 1, 2])
        self.assertGreaterEqual(result["loop_wall_seconds"], 0.0)

    def test_partial_progress_survives_failure(self):
        count = 0

        def step():
            nonlocal count
            count += 1
            if count == 2:
                raise RuntimeError("simulation failed")

        result = {}
        with self.assertRaisesRegex(RuntimeError, "simulation failed"):
            RUNNER.run_frames(
                types.SimpleNamespace(step=step, render=lambda: None),
                types.SimpleNamespace(test=False),
                "demo",
                5,
                {},
                Trace(),
                lambda: None,
                result=result,
            )
        self.assertEqual(result["frames_completed"], 1)
        self.assertEqual(len(result["frame_host_timings"]), 1)

    def test_warmup_does_not_run_final_horizon_assertions(self):
        example = types.SimpleNamespace(
            step=lambda: None,
            render=lambda: None,
            test_final=lambda: self.fail("Final checks belong to the full measured workload"),
        )
        RUNNER.run_frames(
            example,
            types.SimpleNamespace(test=True),
            "demo",
            1,
            {},
            Trace(),
            lambda: None,
            final_test=False,
        )

    def test_terminal_runs_the_optimizer_once_without_video_playback(self):
        original_argv = sys.argv
        calls = []
        with patch.object(
            RUNNER.runpy,
            "run_module",
            side_effect=lambda *args, **kwargs: calls.append((args, kwargs, list(sys.argv))),
        ):
            result = RUNNER.run_terminal(
                "demo.optimize",
                ["--iterations", "20"],
                Trace(),
                lambda: calls.append("sync"),
            )
        self.assertEqual(calls[0], "sync")
        self.assertEqual(
            calls[1],
            (
                ("demo.optimize",),
                {"run_name": "__main__"},
                ["demo.optimize", "--iterations", "20"],
            ),
        )
        self.assertEqual(calls[2], "sync")
        self.assertIs(sys.argv, original_argv)
        self.assertIsNone(result["frames_completed"])


if __name__ == "__main__":
    unittest.main()
