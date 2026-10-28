# Numerical validation

The retained results below were collected on 2026-10-28 using the locked
Python 3.12/Newton 1.7.0/Warp 1.17.0 environment on ARM64 CPU and NVIDIA GB10.
They are recorded evidence, not fresh timing claims from a documentation build.

## Mechanics and derivatives

All 32 unit tests passed on CPU (88.1 s) and CUDA (12.015 s). Tests cover free
fall, independent Neo-Hookean residuals, heterogeneous pinned elements, plane
support, non-crossing soft bodies, rejection of inverted input, three-step
state/force/material finite differences, a dense finite-difference KKT oracle,
recorded-state ownership, analytic muscle-controller VJPs, optimization loss
decrease and taped/live inference agreement. Dense matrices are test oracles;
production mechanical solves remain sparse. The 120-frame forward fixture and
viewer smoke checks passed. A 15-update synthetic material fit recovered
[0.699536, 2.499644] for target scales [0.7, 2.5], with loss
21.4171 → 1.463e-6. Synthetic recovery does not identify measured foam properties.

## Bear

One full 100-step training update reduced loss 0.00840837275344 → 0.00415846192096,
with gradient norm 0.0367675733. Setup plus initial forward/backward took 11.257 s;
the accepted update took 9.459 s. The bundled twenty-update controller stores
historical losses 0.00840834884534513 → -0.08433614249329321; this is checkpoint
metadata, not a new twenty-update validation run.

With that bundled controller, 200 fresh live CUDA steps represented 1.6 simulated
seconds. Every state was finite, the minimum observed determinant was 0.760469,
and minimum height was 0.0000999990 m. Mass-centre travel was
(0.0195200, 0.00839995, -0.00712513) m. After ten warmup steps, synchronized physics
median/p95 was 73.652/100.624 ms and mean realtime factor 0.1095. Rendering,
diagnostic downloads and telemetry were excluded. This scene was not realtime.

Reports: `validation/bear-training-cuda.json` and
`validation/bear-inference-cuda.json`.

## Recorded die and UR5 probe

The complete 35-interval recorded trajectory reduced Chamfer loss
0.0277321302881 → 0.0103461284803 m² in six accepted placement updates within a
twelve-update budget. Final XY was (0.704074745, 0.085339532) m, maximum scaled
mechanics residual 9.94e-7 m, and nearest-vertex Chamfer RMS 17.1931 mm. The
170.1 s total includes cold compilation and diagnostics; it is not a live
physics benchmark. Reloaded live inference and continued deformation after the
last tool pose passed CLI checks.

**The full recorded-trajectory gradient gate did not pass:** central differences
at ±0.0001 m gave 0.784%/2.017% relative discrepancies for X/Y against a 1% gate.
The cause has not been isolated; contact/nearest-feature changes and solver
accuracy may contribute. The gate was not relaxed. Small smooth fixed-contact
alignment, material, compressibility and velocity-damping checks passed on CPU
and CUDA. Report: `validation/poke-recorded-acceptance-cuda.json`.

## Reproduce checks

```bash
uv sync --locked --extra examples
uv run --locked ruff check .
uv run --locked ruff format --check .
uv run --locked python -m unittest discover -s tests -v
NEWTON_TEST_DEVICE=cuda:0 uv run --locked python -m unittest discover -s tests -v
uv run --locked python -m newton_solver_diff_phd.examples diffsim_diff_phd_bear --trained --viewer null --num-frames 200 --test
```

UR5 checks need authorized local recordings; see [usage](usage.md). Wheels and
source archives exclude the private subtree. There are no custom C++/CUDA
extensions, sibling-solver dependencies or private Newton imports. Whole-step
capture is unsupported; the custom adjoint uses pinned Warp runtime hooks.
Linux ARM64 was measured; x86-64 execution needs separate hardware validation.

## Cleanup verification (2026-10-28)

The standalone suite ran with `python -m unittest discover -s tests -v`:

| Device | Tests | Result | Wall time |
| --- | --- | --- | --- |
| CPU | 42 | All passed | 100.870 s |
| CUDA (`NEWTON_TEST_DEVICE=cuda:0`) | 42 | All passed after the parity-test correction below | 13.160 s |

The initial CUDA run had one failure: taped/live inference velocities differed
by 1.7053e-12 m/s in a near-zero component against a 1e-12 absolute tolerance.
All positions were bitwise identical. Twelve additional live rollouts reproduced
up to 1.4381e-12 m/s of reduction noise; velocity output is float32, with a
5.9605e-8 m/s ULP at this trajectory's maximum velocity. Only that inference
comparison's absolute floor changed to 1e-11 m/s; its relative tolerance remains
zero. Exact position/control equality and every gradient gate are unchanged.
The initial suite ran 42 tests in 31.246 s; a targeted unchanged-test repeat also
passed before this correction. Solver module ASTs, excluding docstrings, match
the pre-cleanup implementation. The final CUDA run used
`WARP_CACHE_PATH=/tmp/newton-solvers-cleanup-warp`. Suite timings include setup
and compilation and are not solver performance measurements.

Lint and formatting passed for all 53 Python files. Wheel and source-archive
builds succeeded and excluded private recordings and local outputs. The Docker
image built and passed imports, example discovery and native USD loading in an
isolated container without network or host mounts. Its launcher also passed a
two-frame CPU headless example with `--test`.

The two GPU video captures passed their example `test_final` checks: 240 fresh
bear steps with the bundled controller and 119 fresh poke frames after the
initial display frame. Their 60/30 fps playback rates are video encoding
settings, not measured simulation throughput. The full recorded-trajectory
1% gradient limitation above remains unresolved.
