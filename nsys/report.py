"""Summarize Nsight CSV reports and schema-checked whole-graph GPU events."""

import csv
import json
import re
import sqlite3
import statistics
from contextlib import closing
from pathlib import Path

REPORTS = (
    "cuda_gpu_kern_sum",
    "cuda_api_sum",
    "cuda_gpu_mem_time_sum",
    "cuda_gpu_mem_size_sum",
    "cuda_kern_exec_sum",
    "nvtx_pushpop_sum",
    "nvtx_gpu_proj_sum",
    "nvtx_kern_sum",
    "osrt_sum",
)


def inspect_graph_coverage(directory):
    """Count recorded graph kernels and explicit CUPTI losses, without inferring completeness."""
    database = Path(directory) / "capture.sqlite"
    if not database.is_file():
        raise ValueError("Cannot validate CUDA graph coverage: capture.sqlite is missing")
    coverage = {
        "kernel_events": 0,
        "graph_kernel_events": 0,
        "whole_graph_events": 0,
        "unique_graph_nodes": None,
        "unique_graphs": None,
        "graph_launch_api_calls": {},
        "cupti_lost_records": 0,
        "diagnostics": [],
    }
    try:
        with closing(sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True)) as connection:
            tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}

            def columns(table):
                return {row[1] for row in connection.execute(f"PRAGMA table_info({table})")}

            table = "CUPTI_ACTIVITY_KIND_KERNEL"
            if table in tables:
                available = columns(table)
                coverage["kernel_events"] = connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                markers = [name for name in ("graphNodeId", "graphId") if name in available]
                if not markers:
                    raise ValueError("Unsupported Nsight kernel schema: missing graphNodeId and graphId")
                condition = " OR ".join(f"{name} > 0" for name in markers)
                coverage["graph_kernel_events"] = connection.execute(
                    f"SELECT COUNT(*) FROM {table} WHERE {condition}"
                ).fetchone()[0]
                for column, label in (("graphNodeId", "unique_graph_nodes"), ("graphId", "unique_graphs")):
                    if column in available:
                        coverage[label] = connection.execute(
                            f"SELECT COUNT(DISTINCT {column}) FROM {table} WHERE {column} > 0"
                        ).fetchone()[0]
            table = "CUPTI_ACTIVITY_KIND_GRAPH_TRACE"
            if table in tables:
                if not {"start", "end"} <= columns(table):
                    raise ValueError("Unsupported Nsight graph schema: missing start or end")
                if connection.execute(
                    f"SELECT 1 FROM {table} WHERE start IS NULL OR end IS NULL OR end < start LIMIT 1"
                ).fetchone():
                    raise ValueError("Invalid timestamps in Nsight graph events")
                coverage["whole_graph_events"] = connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            for table in ("CUPTI_ACTIVITY_KIND_RUNTIME", "CUPTI_ACTIVITY_KIND_DRIVER"):
                if table not in tables:
                    continue
                if (
                    "nameId" not in columns(table)
                    or "StringIds" not in tables
                    or not {"id", "value"} <= columns("StringIds")
                ):
                    raise ValueError("Unsupported Nsight CUDA API schema: missing API name mapping")
                coverage["graph_launch_api_calls"][table] = connection.execute(
                    f"SELECT COUNT(*) FROM {table} a JOIN StringIds s ON a.nameId=s.id "
                    "WHERE s.value LIKE 'cudaGraphLaunch%' OR s.value LIKE 'cuGraphLaunch%'"
                ).fetchone()[0]
            if "DIAGNOSTIC_EVENT" in tables:
                if not {"severity", "text"} <= columns("DIAGNOSTIC_EVENT"):
                    raise ValueError("Unsupported Nsight diagnostic schema: missing severity or text")
                for severity, message, count in connection.execute(
                    "SELECT severity, text, COUNT(*) FROM DIAGNOSTIC_EVENT GROUP BY severity, text"
                ):
                    lost = 0
                    if message and "cupti" in message.lower():
                        patterns = (
                            r"\b(?:dropped|incomplete)\s+(?:CUPTI\s+)?(\d[\d,]*)\s+(?:records|events)\b",
                            r"\b(\d[\d,]*)\s+(?:dropped|incomplete)\s+(?:CUPTI\s+)?(?:records|events)\b",
                            r"\b(?:incomplete\s+)?CUPTI\s+(?:records|events)\s+dropped:\s*(\d[\d,]*)\b",
                        )
                        lost = sum(
                            int(value.replace(",", ""))
                            for pattern in patterns
                            for value in re.findall(pattern, message, re.I)
                        )
                    coverage["cupti_lost_records"] += lost * count
                    if (severity is not None and severity >= 2) or lost:
                        coverage["diagnostics"].append({"severity": severity, "message": message, "count": count})
    except sqlite3.Error as error:
        raise ValueError(f"Cannot read Nsight CUDA graph coverage: {error}") from error
    return coverage


def validate_graph_coverage(coverage, requested_mode):
    """Reject explicit record loss or missing events for the requested graph detail."""
    if coverage["cupti_lost_records"]:
        raise ValueError(
            f"Incomplete CUDA capture: CUPTI reports {coverage['cupti_lost_records']} dropped/incomplete records"
        )
    if (
        requested_mode.split(":")[0] == "node"
        and any(coverage["graph_launch_api_calls"].values())
        and not coverage["graph_kernel_events"]
    ):
        raise ValueError("Incomplete CUDA node capture: graph-launch APIs were recorded but no graph kernel events")
    if (
        requested_mode.split(":")[0] == "graph"
        and any(coverage["graph_launch_api_calls"].values())
        and not coverage["whole_graph_events"]
    ):
        raise ValueError(
            "Incomplete CUDA graph capture: graph-launch APIs were recorded but no whole-graph duration events"
        )


def export_graph_summary(directory):
    """Export GPU graph durations omitted by Nsight's stock kernel reports."""
    directory = Path(directory)
    database = directory / "capture.sqlite"
    if not database.is_file():
        return []
    try:
        with closing(sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True)) as connection:
            table = "CUPTI_ACTIVITY_KIND_GRAPH_TRACE"
            if not connection.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)).fetchone():
                return []
            groups = ("globalPid", "deviceId", "contextId", "streamId", "graphId", "graphExecId")
            columns = {row[1] for row in connection.execute(f"PRAGMA table_info({table})")}
            missing = {"start", "end", *groups} - columns
            if missing:
                raise ValueError(f"Unsupported Nsight graph schema; missing columns: {', '.join(sorted(missing))}")
            if connection.execute(
                f"SELECT 1 FROM {table} WHERE start IS NULL OR end IS NULL OR end < start LIMIT 1"
            ).fetchone():
                raise ValueError("Invalid timestamps in Nsight graph events")
            events = connection.execute(
                f"SELECT SUM(end-start), COUNT(*), AVG(end-start), MIN(end-start), MAX(end-start), "
                f"{', '.join(groups)} FROM {table} GROUP BY {', '.join(groups)} ORDER BY SUM(end-start) DESC"
            ).fetchall()
    except sqlite3.Error as error:
        raise ValueError(f"Cannot read Nsight GPU graph events from {database.name}: {error}") from error
    if not events:
        return []
    total = sum(row[0] for row in events)
    fields = (
        "Time (%)",
        "Total Time (ns)",
        "Instances",
        "Avg (ns)",
        "Min (ns)",
        "Max (ns)",
        "Global PID",
        "Device",
        "Context",
        "Stream",
        "Graph",
        "Graph Exec",
    )
    rows = [dict(zip(fields, (100 * row[0] / total if total else 0.0, *row), strict=True)) for row in events]
    with (directory / "stats_cuda_graph_sum.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def read_reports(directory):
    tables = {}
    for name in (*REPORTS, "cuda_graph_sum"):
        path = Path(directory) / f"stats_{name}.csv"
        if path.is_file():
            with path.open(newline="") as stream:
                rows = list(csv.DictReader(stream))
            if rows:
                tables[name] = rows
    return tables


def number(value):
    try:
        return float(str(value).replace(",", ""))
    except (TypeError, ValueError):
        return 0.0


def totals(tables):
    # These are summed event durations, never wall time or GPU utilization.
    return {
        name: sum(number(row.get("Total Time (ns)")) for row in tables.get(name, [])) / 1e6
        for name in ("cuda_gpu_kern_sum", "cuda_graph_sum", "cuda_api_sum", "cuda_gpu_mem_time_sum")
        if name in tables
    }


def table_markdown(rows, limit=12):
    if not rows:
        return "No events reported.\n"
    columns = list(rows[0])

    # Preserve Nsight's exact column names and units; CSV retains all rows.
    def cell(value):
        return str(value).replace("|", "\\|").replace("\n", " ").replace("`", "'")

    lines = ["| " + " | ".join(map(cell, columns)) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    lines.extend("| " + " | ".join(cell(row.get(column, "")) for column in columns) + " |" for row in rows[:limit])
    return "\n".join(lines) + "\n"


def write_summary(directory, metadata):
    directory = Path(directory)
    tables = read_reports(directory)
    workload_path = directory / "workload.json"
    workload = json.loads(workload_path.read_text()) if workload_path.is_file() else {}
    graph_mode = (
        metadata.get("profiling", {}).get("collection_settings", {}).get("--cuda-graph-trace", "").split(":")[0]
        == "graph"
    )
    detail_note = (
        "Whole-graph GPU durations exclude per-node detail. Kernel tables and percentages cover only kernels "
        "outside captured graphs; they do not rank the work inside graphs. "
        "NVTX GPU projection and kernel summaries are also derived from individual kernel events and may omit graph work."
        if graph_mode
        else "Kernel tables contain the recorded individual CUDA kernel events."
    )
    summary = {
        "run": metadata,
        "workload": workload,
        "trace_detail": detail_note,
        "summed_event_ms": totals(tables),
        "reports": tables,
    }
    coverage = metadata.get("profiling", {}).get("graph_coverage")
    if coverage is not None:
        summary["graph_coverage"] = coverage
    (directory / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    lines = [
        f"# {metadata['example']} — {metadata['status']}",
        "",
        f"Revision: `{metadata['git']['commit']}`; source digest: `{metadata['source_sha256']}`.",
        f"Viewer: `{metadata['options']['viewer']}`. Capture includes fresh initialization/training and the scenario.",
        "The default capture excludes the independent warmup run. `--include-startup` also records imports and warmup.",
        "These are instrumented measurements, not an unprofiled real-time benchmark.",
        detail_note,
        "",
        "Open `capture.nsys-rep` in Nsight Systems for the full CPU/GPU timeline. "
        "`run.json` records the command, settings, code and environment; `workload.json` records completed work. "
        "`summary.json` and `stats_*.csv` retain all table rows; this page shows the first twelve.",
        "",
    ]
    if metadata.get("error"):
        lines += [f"Run error: {metadata['error']}", "See `capture.log` and `stats.log`.", ""]
    if coverage is not None:
        lines += [
            "## CUDA graph coverage",
            "",
            f"Recorded {coverage['kernel_events']:,} kernel events, including "
            f"{coverage['graph_kernel_events']:,} attributed to CUDA graphs. "
            f"Whole-graph duration events: {coverage['whole_graph_events']:,}. "
            f"Explicit CUPTI dropped/incomplete records: {coverage['cupti_lost_records']:,}.",
            "Graph membership comes from the exported graphNodeId/graphId fields. "
            "These counts establish recorded coverage; they do not prove that every expected event was collected. "
            "Runtime and driver graph-launch API counts may describe the same launch and are kept separate in JSON.",
            "",
        ]
        for diagnostic in coverage["diagnostics"]:
            message = str(diagnostic["message"]).replace("\n", " ")
            lines.append(f"- Nsight diagnostic ({diagnostic['count']}): {message}")
        lines.append("")
    if workload:
        lines += ["## Work completed and host timings", ""]
        for key in (
            "status",
            "workload",
            "frames_requested",
            "frames_completed",
            "presentation_frames",
            "initialization_wall_seconds",
            "loop_wall_seconds",
            "optimization_wall_seconds",
        ):
            if key in workload:
                lines.append(f"- {key}: `{workload[key]}`")
        lines += ["", "| Host phase | Mean (ms) | p95 (ms) | Maximum (ms) |", "| --- | ---: | ---: | ---: |"]
        for phase in ("scenario", "step", "test", "render"):
            values = sorted(
                row[phase + "_host_seconds"] * 1000
                for row in workload.get("frame_host_timings", [])
                if phase + "_host_seconds" in row
            )
            if values:
                p95 = values[min(len(values) - 1, int(0.95 * len(values)))]
                lines.append(f"| {phase} | {statistics.mean(values):.3f} | {p95:.3f} | {max(values):.3f} |")
        lines += [
            "",
            workload.get("render_note", ""),
            "",
            "All per-frame samples and annotated method names are in `workload.json`.",
            "",
        ]
    sections = (
        ("Whole-graph GPU durations (no per-node detail)", "cuda_graph_sum"),
        ("GPU kernels outside graphs" if graph_mode else "Largest GPU kernel costs", "cuda_gpu_kern_sum"),
        ("CUDA calls on the CPU", "cuda_api_sum"),
        ("Memory transfer time", "cuda_gpu_mem_time_sum"),
        ("Memory transfer volume", "cuda_gpu_mem_size_sum"),
        ("Kernel launch, queue and execution", "cuda_kern_exec_sum"),
        ("Host phase ranges", "nvtx_pushpop_sum"),
        ("GPU spans associated with phases", "nvtx_gpu_proj_sum"),
        ("Kernels associated with phases", "nvtx_kern_sum"),
        ("Operating system calls", "osrt_sum"),
    )
    for title, report in sections:
        lines += [f"## {title}", "", table_markdown(tables.get(report, [])), ""]
    lines += [
        "## Reading the bottleneck",
        "",
        (
            "Start with whole-graph GPU durations and their host launch phases in the timeline. "
            if graph_mode
            else "Start with the largest GPU kernels, then inspect their parent phase in the timeline. "
        )
        + "Long host CUDA synchronization calls can be waits for previously submitted GPU work. "
        "Large transfer totals warrant checking host/device copies; repeated tiny launches warrant inspecting launch overhead. "
        "Python sampling stacks in the GUI help locate CPU work between launches.",
        "",
        "Kernel percentages refer to summed recorded kernel duration, not wall time or SM utilization. "
        "Whole-graph percentages refer to summed graph duration; graphs can contain concurrent work and gaps. "
        "GPU work can overlap. CUDA API times are host call durations. Queue time alone is not evidence of a problem. "
        "Nested NVTX durations overlap, and projected GPU spans include gaps; do not add them together. "
        "Per-phase host timings do not synchronize every step, so use the GPU timeline for execution costs. "
        "First-use compilation during a later cut or optimization event can remain even after warmup.",
        "",
        "Missing memory, OS or NVTX tables can simply mean no matching events were recorded. "
        + (
            "Graph-mode captures need GPU kernel or graph events; an empty standalone kernel table is valid when graphs ran. "
            "Missing graph events do not imply zero graph cost."
            if graph_mode
            else "A missing CUDA kernel table makes this GPU capture incomplete, not a zero-cost run."
        ),
        "",
        "[NVIDIA analysis guide](https://docs.nvidia.com/nsight-systems/AnalysisGuide/index.html)",
        "",
    ]
    (directory / "summary.md").write_text("\n".join(lines))
    return summary


def compare(directory, previous):
    directory, previous = Path(directory), Path(previous)
    current = json.loads((directory / "summary.json").read_text())
    old = json.loads((previous / "summary.json").read_text())
    if any(item["run"]["status"] != "complete" for item in (current, old)):
        raise ValueError("Only completed captures can be compared")
    if current["run"]["comparison_key"] != old["run"]["comparison_key"]:
        raise ValueError("Capture settings, scenario, arguments or GPU differ; these runs are not comparable")
    lines = [
        "# Capture comparison",
        "",
        f"Previous: `{previous}`",
        f"Current: `{directory}`",
        "",
        "Observed differences under profiling; repeated runs are needed to assess noise.",
        "Summed graph durations include the whole graph and exclude per-node detail; "
        "in graph mode, kernel totals cover only work outside graphs. Event sums are not wall time.",
        "",
        "| Summed event time | Previous (ms) | Current (ms) | Change (%) |",
        "| --- | ---: | ---: | ---: |",
    ]
    for name, value in current["summed_event_ms"].items():
        before = old["summed_event_ms"].get(name)
        if before is not None:
            delta = f"{100 * (value / before - 1):+.2f}" if before else "n/a"
            lines.append(f"| {name} | {before:.3f} | {value:.3f} | {delta} |")
    (directory / "comparison.md").write_text("\n".join(lines) + "\n")
