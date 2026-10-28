"""Whole-graph GPU timing and explicit limits of graph-granularity captures."""

import importlib.util
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("graph_report", ROOT / "nsys/report.py")
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)


def graph_database(directory, events=()):
    with sqlite3.connect(Path(directory) / "capture.sqlite") as connection:
        connection.execute(
            "CREATE TABLE CUPTI_ACTIVITY_KIND_GRAPH_TRACE "
            "(start INTEGER, end INTEGER, globalPid INTEGER, deviceId INTEGER, contextId INTEGER, "
            "streamId INTEGER, graphId INTEGER, graphExecId INTEGER)"
        )
        connection.executemany("INSERT INTO CUPTI_ACTIVITY_KIND_GRAPH_TRACE VALUES (?,?,?,?,?,?,?,?)", events)


def metadata():
    return {
        "example": "bear",
        "status": "complete",
        "git": {"commit": "abc"},
        "source_sha256": "digest",
        "options": {"viewer": "null"},
        "comparison_key": "same-workload",
        "profiling": {"collection_settings": {"--cuda-graph-trace": "graph:host-only"}},
    }


class TestGraphReports(unittest.TestCase):
    def test_actual_gpu_graph_durations_are_grouped_without_merging_contexts(self):
        with tempfile.TemporaryDirectory() as directory:
            graph_database(
                directory,
                [
                    (100, 400, 10, 0, 1, 7, 3, 4),
                    (450, 550, 10, 0, 1, 7, 3, 4),
                    (600, 800, 10, 0, 2, 7, 3, 4),
                ],
            )
            rows = report.export_graph_summary(directory)
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["Total Time (ns)"], 400)
            self.assertEqual(rows[0]["Instances"], 2)
            self.assertEqual((rows[0]["Avg (ns)"], rows[0]["Min (ns)"], rows[0]["Max (ns)"]), (200, 100, 300))
            self.assertAlmostEqual(rows[0]["Time (%)"], 200 / 3)
            tables = report.read_reports(directory)
            self.assertEqual(len(tables["cuda_graph_sum"]), 2)
            self.assertAlmostEqual(report.totals(tables)["cuda_graph_sum"], 0.0006)

    def test_absent_or_empty_graph_events_have_no_fabricated_costs(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(report.export_graph_summary(directory), [])
            with sqlite3.connect(Path(directory) / "capture.sqlite"):
                pass
            self.assertEqual(report.export_graph_summary(directory), [])
            graph_database(directory)
            self.assertEqual(report.export_graph_summary(directory), [])
            self.assertFalse((Path(directory) / "stats_cuda_graph_sum.csv").exists())

    def test_unknown_schema_fails_instead_of_silently_reporting_zero(self):
        with tempfile.TemporaryDirectory() as directory:
            with sqlite3.connect(Path(directory) / "capture.sqlite") as connection:
                connection.execute("CREATE TABLE CUPTI_ACTIVITY_KIND_GRAPH_TRACE (start INTEGER, end INTEGER)")
            with self.assertRaisesRegex(ValueError, "Unsupported Nsight graph schema; missing columns"):
                report.export_graph_summary(directory)

    def test_invalid_duration_is_not_reported_as_gpu_work(self):
        with tempfile.TemporaryDirectory() as directory:
            graph_database(directory, [(50, 10, 1, 0, 1, 1, 1, 1)])
            with self.assertRaisesRegex(ValueError, "Invalid timestamps"):
                report.export_graph_summary(directory)

    def test_corrupt_sqlite_raises_a_capture_validation_error(self):
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "capture.sqlite").write_bytes(b"incomplete database export")
            with self.assertRaisesRegex(ValueError, "Cannot read Nsight GPU graph events from capture.sqlite"):
                report.export_graph_summary(directory)

    def test_graph_summary_marks_kernel_tables_partial_and_compares_graph_totals(self):
        with tempfile.TemporaryDirectory() as directory:
            previous, current = Path(directory) / "previous", Path(directory) / "current"
            for path, duration in ((previous, 2_000_000), (current, 1_000_000)):
                path.mkdir()
                graph_database(path, [(1, 1 + duration, 1, 0, 1, 1, 1, 1)])
                report.export_graph_summary(path)
                summary = report.write_summary(path, metadata())
                self.assertEqual(summary["summed_event_ms"]["cuda_graph_sum"], duration / 1e6)
                self.assertIn("only kernels outside captured graphs", summary["trace_detail"])
                text = (path / "summary.md").read_text()
                self.assertIn("Whole-graph GPU durations (no per-node detail)", text)
                self.assertIn("GPU kernels outside graphs", text)
                self.assertIn("empty standalone kernel table is valid", text)
                self.assertEqual(json.loads((path / "summary.json").read_text())["reports"], summary["reports"])
            report.compare(current, previous)
            self.assertIn("| cuda_graph_sum | 2.000 | 1.000 | -50.00 |", (current / "comparison.md").read_text())


if __name__ == "__main__":
    unittest.main()
