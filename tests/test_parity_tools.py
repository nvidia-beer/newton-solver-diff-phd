"""The reference comparison CLI must not report failed or empty runs as success."""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from tools.parity import compare, port_cases


class TestParityTools(unittest.TestCase):
    def test_original_single_energy_heterogeneity_is_judged(self):
        fixture = {
            "mu": np.array([1.0, 100.0]),
            "lambda": np.array([2.0, 200.0]),
            "reference_material_encoding": "per_element",
        }
        self.assertEqual(compare._gradient_regime(fixture, False), "reference")
        self.assertEqual(compare._gradient_regime(fixture, True), "contact")
        fixture["reference_material_encoding"] = "grouped"
        self.assertEqual(compare._gradient_regime(fixture, False), "documented_deviation")

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.report = self.directory / "report.json"
        (self.directory / "port_environment.json").write_text('{"device": "cpu"}\n')
        self.arguments = [
            "compare.py",
            "--results",
            str(self.directory),
            "--reference-root",
            str(self.directory / "reference"),
            "--report",
            str(self.report),
        ]

    def fixture(self):
        np.savez(
            self.directory / "sample.fixture.npz",
            case="sample",
            port_revision="test-revision",
            port_source_sha256="test-source-hash",
            benchmark_steps=0,
        )
        np.savez(self.directory / "sample.port.npz")
        np.savez(self.directory / "sample.reference.npz")

    def run_comparison(self, *, passed=True, rejected=False):
        outcome = {"case": "sample", "pass": passed, "gradient_regime": "exact", "checks": []}
        with (
            patch("sys.argv", self.arguments),
            patch.object(compare, "_command", return_value=compare.REFERENCE_REVISION),
            patch.object(compare, "reject", return_value=["schema"] if rejected else []),
            patch.object(compare, "compare_case", return_value=outcome),
            contextlib.redirect_stdout(io.StringIO()),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            return compare.main()

    def test_failed_numerical_comparison_exits_unsuccessfully_and_keeps_report(self):
        self.fixture()
        self.assertEqual(self.run_comparison(passed=False), 1)
        self.assertEqual(json.loads(self.report.read_text())["summary"]["failed"], ["sample"])

    def test_rejected_fixture_exits_unsuccessfully(self):
        self.fixture()
        self.assertEqual(self.run_comparison(rejected=True), 1)
        self.assertEqual(json.loads(self.report.read_text())["rejected"][0]["reasons"], ["schema"])

    def test_passing_comparison_exits_successfully(self):
        self.fixture()
        self.assertEqual(self.run_comparison(), 0)

    def test_missing_fixtures_cannot_produce_a_passing_report(self):
        with self.assertRaises(SystemExit) as raised:
            self.run_comparison()
        self.assertEqual(raised.exception.code, 2)
        self.assertFalse(self.report.exists())

    def test_wrong_reference_revision_is_rejected_before_comparison(self):
        self.fixture()
        with (
            patch("sys.argv", self.arguments),
            patch.object(compare, "_command", return_value="unrelated-reference-revision"),
            contextlib.redirect_stderr(io.StringIO()),
            self.assertRaises(SystemExit) as raised,
        ):
            compare.main()
        self.assertEqual(raised.exception.code, 2)
        self.assertFalse(self.report.exists())

    def test_misspelled_case_does_not_silently_skip_validation(self):
        with (
            patch("sys.argv", ["port_cases.py", "--out", str(self.directory), "--cases", "homogeneous_fre"]),
            patch.object(port_cases, "case_fixtures", return_value=[{"case": "homogeneous_free"}]),
            contextlib.redirect_stderr(io.StringIO()),
            self.assertRaises(SystemExit) as raised,
        ):
            port_cases.main()
        self.assertEqual(raised.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
