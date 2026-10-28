# Nsight Systems profiling

Profile the live workloads shown in this repository's README videos. The list is
built from discovered examples with an MP4 and preview in `docs/media/`; it uses
`docs/videos.json` and `tools/video_scenarios.py` for the recorded settings and
interactions. No separate example list is maintained.

## Run

From the repository root, after the normal Docker build:

```bash
./docker/run.sh --nsys --list
./docker/run.sh --nsys <example>
./docker/run.sh --nsys --all
```

Install [NVIDIA Nsight Systems](https://developer.nvidia.com/nsight-systems) on
the host. Docker discovers standard installation paths, or use
`NSYS_ROOT=/path/to/nsight-systems` (the directory containing `bin/nsys`). The
installation is mounted read-only for profiling. Use the host CPU architecture
and a CUDA-capable NVIDIA GPU. Rebuild an older solver image to install the new
optional profiling dependency.

For local development, put `nsys` on `PATH` or set `NSYS_PATH` to its executable:

```bash
uv sync --locked --extra examples --extra profiling
uv run --locked python nsys/profile_examples.py --list
uv run --locked python nsys/profile_examples.py <example> --label baseline
uv run --locked python nsys/profile_examples.py --all
```

`python3 nsys/profile_examples.py --all --dry-run` prints the capture commands
without requiring CUDA or Nsight. `--help` lists all options. Extra example
arguments follow `--`; they are recorded alongside the video defaults.

## What is measured

The default run warms five frames in a separate instance, then captures fresh
initialization and the complete video scenario from frame zero. This retains
scripted cuts, pressure changes, steering and learning events. Training performed
by an example's constructor is included in the `initialization` range, separately
from the frame loop. Warmup can still leave first-use compilation in later
branches; the trace makes those costs visible.

For terminal optimization examples, video frames only animate a result chart.
The profiler runs the actual optimization once, with one independent optimization
for warmup when `--warmup` is nonzero. Frame counts are marked inapplicable;
`--frames` cannot shorten such a workload. Use its optimization options after `--`.

| Option | Effect |
| --- | --- |
| `--viewer null` | Default. Runs physics and the example's visualization preparation, with no OpenGL drawing. |
| `--viewer gl` | Adds actual OpenGL drawing, at 960 × 540 by default; Docker needs the normal X11 access. |
| `--warmup N` | Changes independent warmup frames; zero skips warmup. |
| `--frames N` | Shortens or extends a visual workload. Short captures may omit the feature shown in the video. |
| `--include-startup` | Also captures initial imports and warmup. |
| `--cpu-sampling` | Adds native CPU sampling and context switches when host performance-counter permissions allow it. |
| `--cuda-graph-trace node\|graph` | Resolves graph kernels individually, or records whole-graph GPU durations. Overrides any example setting in `nsys/config.json`. |
| `--timeout SECONDS` | Limits each capture/statistics process; default 1800 seconds. |

The profiler runs live example code without video encoding or framebuffer
readback. Recorded targets displayed by an example remain part of its workload.
Regular example runs load no profiling wrappers. Only this profiling process
loads NVTX and temporarily annotates solver, assembly, linear solve, contact,
optimization and backward calls that are present in the implementation. Numerical
resolution, substeps and iteration budgets remain those of the chosen workload.

CUDA graphs replay without calling the Python methods again. Inspect graph nodes
and kernel names inside the frame/step range for those paths. Whole-graph mode
instead reports graph IDs, replay counts and GPU durations in `stats_cuda_graph_sum.csv`;
its kernel table covers only work outside graphs. `summary.md` marks this limit.
The optional `nsys/config.json` holds profiling overrides, not an example list;
discovery still selects only README video workloads. Host phase timings
measure asynchronous submission, including any synchronization already performed
by the example. The runner synchronizes at interval boundaries, not every step;
loop wall time includes the final device synchronization. Nsight and annotations
add overhead, so these captures do not establish unprofiled real-time throughput.

## Results and comparison

Each run creates a unique timestamped directory under `nsys/results/`:

| File | Contents |
| --- | --- |
| `summary.md` | Completed work, host timing summaries and largest reported costs. |
| `capture.nsys-rep` | Full CPU/GPU timeline for the Nsight Systems GUI. |
| `stats_*.csv`, `summary.json` | Full kernel, CUDA API, memory, NVTX and OS reports. |
| `run.json` | Command, workload settings, Git revision/source digest, versions, GPU information and completion status. |
| `workload.json` | Initialization/warmup durations, completed frames and individual host phase timings. |
| `capture.log`, `stats.log` | Complete application/profiler output and errors. |

`nsys/results/history.jsonl` indexes successful and failed runs. Results are
ignored by Git and stay local; copy a selected capture when sharing findings.
Failed captures retain diagnostics and are not accepted as performance results.
Start with the largest kernels in `summary.md`, then inspect their phase and
launching Python stack in the GUI. Kernel percentages and totals sum event
execution time; they are not wall time or GPU utilization. Nested NVTX ranges
and concurrent GPU work overlap, so their durations must not be added together.

Compare a new run with a previous completed capture:

```bash
uv run --locked python nsys/profile_examples.py <example> \
  --label candidate --compare nsys/results/<previous-run>
```

This writes `comparison.md` with changes in summed kernel, graph, CUDA API and
transfer times. Workload settings, scripted scenario, runner fingerprint,
collection options, versions, known file inputs and GPU metadata must match;
solver source revisions may differ. Repeat runs to assess noise.
Only file inputs named explicitly in the arguments are hashed. Keep any other
external datasets, device selection and machine load unchanged yourself; the
comparison key cannot establish that they are identical.

Examples requiring private data or trained checkpoints retain those requirements.
The profiler does not generate substitute inputs. See the repository's usage
notes and the capture log if a required asset is unavailable.

[NVIDIA analysis guide](https://docs.nvidia.com/nsight-systems/AnalysisGuide/index.html)
