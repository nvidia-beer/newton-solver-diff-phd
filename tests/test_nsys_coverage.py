"""Validate actual CUDA graph kernel records and explicit profiler record loss."""

import importlib.util
import sqlite3
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("coverage_report", ROOT / "nsys/report.py")
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)


def database(directory, *, graph_ids=True, kernels=(), diagnostics=(), runtime_launches=1, driver_launches=0):
    with sqlite3.connect(Path(directory) / "capture.sqlite") as connection:
        column = ", graphId INTEGER" if graph_ids else ""
        connection.execute(f"CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (graphNodeId INTEGER{column})")
        connection.executemany(
            "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES " + ("(?,?)" if graph_ids else "(?)"), kernels
        )
        connection.execute("CREATE TABLE StringIds (id INTEGER, value TEXT)")
        connection.executemany(
            "INSERT INTO StringIds VALUES (?,?)",
            [(1, "cudaGraphLaunch_v10000"), (2, "cuGraphLaunch"), (3, "cudaGraphCreate")],
        )
        for kind, count, name_id in (("RUNTIME", runtime_launches, 1), ("DRIVER", driver_launches, 2)):
            connection.execute(f"CREATE TABLE CUPTI_ACTIVITY_KIND_{kind} (nameId INTEGER)")
            connection.executemany(f"INSERT INTO CUPTI_ACTIVITY_KIND_{kind} VALUES (?)", [(name_id,)] * count + [(3,)])
        connection.execute("CREATE TABLE DIAGNOSTIC_EVENT (severity INTEGER, text TEXT)")
        connection.executemany("INSERT INTO DIAGNOSTIC_EVENT VALUES (?,?)", diagnostics)


class TestGraphCoverage(unittest.TestCase):
    def test_new_schema_counts_graph_membership_without_double_counting(self):
        with tempfile.TemporaryDirectory() as directory:
            database(directory, kernels=[(None, None), (0, 0), (1, 7), (1, 7), (2, 7), (None, 8)], driver_launches=1)
            coverage = report.inspect_graph_coverage(directory)
            self.assertEqual(coverage["kernel_events"], 6)
            self.assertEqual(coverage["graph_kernel_events"], 4)
            self.assertEqual(coverage["unique_graph_nodes"], 2)
            self.assertEqual(coverage["unique_graphs"], 2)
            self.assertEqual(
                coverage["graph_launch_api_calls"], {"CUPTI_ACTIVITY_KIND_RUNTIME": 1, "CUPTI_ACTIVITY_KIND_DRIVER": 1}
            )
            report.validate_graph_coverage(coverage, "node")

    def test_old_schema_does_not_invent_missing_graph_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            database(directory, graph_ids=False, kernels=[(None,), (0,), (123,), (123,)])
            coverage = report.inspect_graph_coverage(directory)
            self.assertEqual(coverage["graph_kernel_events"], 2)
            self.assertEqual(coverage["unique_graph_nodes"], 1)
            self.assertIsNone(coverage["unique_graphs"])

    def test_node_mode_with_graph_launches_needs_graph_kernel_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            database(directory, kernels=[(None, None)])
            coverage = report.inspect_graph_coverage(directory)
            with self.assertRaisesRegex(ValueError, "graph-launch APIs.*no graph kernel"):
                report.validate_graph_coverage(coverage, "node")
            with self.assertRaisesRegex(ValueError, "graph-launch APIs.*no whole-graph duration"):
                report.validate_graph_coverage(coverage, "graph:host-only")
            coverage["graph_launch_api_calls"] = {}
            report.validate_graph_coverage(coverage, "node")
            report.validate_graph_coverage(coverage, "graph")

    def test_graph_mode_requires_actual_gpu_duration_events(self):
        with tempfile.TemporaryDirectory() as directory:
            database(directory, kernels=[(None, None)])
            with sqlite3.connect(Path(directory) / "capture.sqlite") as connection:
                connection.execute("CREATE TABLE CUPTI_ACTIVITY_KIND_GRAPH_TRACE (start INTEGER, end INTEGER)")
            coverage = report.inspect_graph_coverage(directory)
            self.assertEqual(coverage["whole_graph_events"], 0)
            with self.assertRaisesRegex(ValueError, "no whole-graph duration events"):
                report.validate_graph_coverage(coverage, "graph")
            with sqlite3.connect(Path(directory) / "capture.sqlite") as connection:
                connection.executemany(
                    "INSERT INTO CUPTI_ACTIVITY_KIND_GRAPH_TRACE VALUES (?,?)", [(100, 200), (220, 400)]
                )
            coverage = report.inspect_graph_coverage(directory)
            self.assertEqual(coverage["whole_graph_events"], 2)
            report.validate_graph_coverage(coverage, "graph:host-only")
            with self.assertRaisesRegex(ValueError, "no graph kernel events"):
                report.validate_graph_coverage(coverage, "node")

    def test_invalid_graph_records_are_not_accepted_as_duration_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            database(directory, kernels=[(None, None)])
            with sqlite3.connect(Path(directory) / "capture.sqlite") as connection:
                connection.execute("CREATE TABLE CUPTI_ACTIVITY_KIND_GRAPH_TRACE (graphId INTEGER)")
            with self.assertRaisesRegex(ValueError, "Unsupported Nsight graph schema"):
                report.inspect_graph_coverage(directory)
            with sqlite3.connect(Path(directory) / "capture.sqlite") as connection:
                connection.execute("DROP TABLE CUPTI_ACTIVITY_KIND_GRAPH_TRACE")
                connection.execute("CREATE TABLE CUPTI_ACTIVITY_KIND_GRAPH_TRACE (start INTEGER, end INTEGER)")
                connection.execute("INSERT INTO CUPTI_ACTIVITY_KIND_GRAPH_TRACE VALUES (100, 90)")
            with self.assertRaisesRegex(ValueError, "Invalid timestamps"):
                report.inspect_graph_coverage(directory)

    def test_actual_info_level_dropped_record_diagnostic_invalidates_capture(self):
        with tempfile.TemporaryDirectory() as directory:
            database(
                directory,
                kernels=[(1, 1)],
                diagnostics=[(1, "CUPTI dropped 744568 records. Some CUPTI events might be missing.")],
            )
            coverage = report.inspect_graph_coverage(directory)
            self.assertEqual(coverage["cupti_lost_records"], 744568)
            self.assertEqual(coverage["diagnostics"][0]["severity"], 1)
            with self.assertRaisesRegex(ValueError, "744568 dropped/incomplete records"):
                report.validate_graph_coverage(coverage, "node")
            with self.assertRaises(ValueError):
                report.validate_graph_coverage(coverage, "graph")

    def test_numeric_incomplete_records_are_not_a_generic_boundary_warning(self):
        with tempfile.TemporaryDirectory() as directory:
            database(
                directory,
                kernels=[(1, 1)],
                diagnostics=[
                    (1, "CUPTI reported 1,234 incomplete records."),
                    (2, "Not all CUDA events might have been collected."),
                ],
            )
            self.assertEqual(report.inspect_graph_coverage(directory)["cupti_lost_records"], 1234)

    def test_hardware_trace_incomplete_cupti_events_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            database(
                directory,
                kernels=[(1, 1)],
                diagnostics=[
                    (
                        1,
                        "Number of incomplete CUPTI events dropped: 17317. Incomplete events appear when tracing stops.",
                    ),
                ],
            )
            coverage = report.inspect_graph_coverage(directory)
            self.assertEqual(coverage["cupti_lost_records"], 17317)
            with self.assertRaisesRegex(ValueError, "17317 dropped/incomplete records"):
                report.validate_graph_coverage(coverage, "node")

    def test_generic_boundary_warning_is_preserved_without_fabricating_record_loss(self):
        with tempfile.TemporaryDirectory() as directory:
            database(
                directory,
                kernels=[(1, 1)],
                diagnostics=[(2, "Not all CUDA events might have been collected."), (1, "CUPTI dropped 0 records.")],
            )
            coverage = report.inspect_graph_coverage(directory)
            self.assertEqual(coverage["cupti_lost_records"], 0)
            self.assertEqual(len(coverage["diagnostics"]), 1)
            report.validate_graph_coverage(coverage, "node")

    def test_missing_and_corrupt_exports_fail_clearly(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "capture.sqlite is missing"):
                report.inspect_graph_coverage(directory)
            (Path(directory) / "capture.sqlite").write_bytes(b"corrupt database")
            with self.assertRaisesRegex(ValueError, "Cannot read Nsight CUDA graph coverage"):
                report.inspect_graph_coverage(directory)

    def test_unknown_kernel_schema_does_not_report_zero_graph_work(self):
        with tempfile.TemporaryDirectory() as directory:
            with sqlite3.connect(Path(directory) / "capture.sqlite") as connection:
                connection.execute("CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (start INTEGER, end INTEGER)")
            with self.assertRaisesRegex(ValueError, "Unsupported Nsight kernel schema"):
                report.inspect_graph_coverage(directory)

    def test_summary_preserves_observed_counts_and_diagnostics(self):
        with tempfile.TemporaryDirectory() as directory:
            database(directory, kernels=[(1, 7)], diagnostics=[(2, "Not all CUDA events might have been collected.")])
            coverage = report.inspect_graph_coverage(directory)
            metadata = {
                "example": "bear",
                "status": "complete",
                "git": {"commit": "abc"},
                "source_sha256": "digest",
                "options": {"viewer": "null"},
                "profiling": {"graph_coverage": coverage},
            }
            summary = report.write_summary(directory, metadata)
            self.assertEqual(summary["graph_coverage"], coverage)
            self.assertEqual(summary["run"]["profiling"]["graph_coverage"], coverage)
            markdown = (Path(directory) / "summary.md").read_text()
            self.assertIn("1 kernel events, including 1 attributed", markdown)
            self.assertIn("Not all CUDA events might have been collected.", markdown)
            self.assertIn("do not prove", markdown)


if __name__ == "__main__":
    unittest.main()
