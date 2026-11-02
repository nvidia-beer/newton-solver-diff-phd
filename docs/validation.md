# Numerical validation

The retained validation is dated **2026-11-02**. `SolverDiffPhD` was replaced in
this checkout using the split implementation from the now-retired
`newton-solver-diff-phd-port`,
then its experimental contact and SVD branches were removed. `SolverDiffPD`
remains a separate forward solver for the DiffPD algorithm. The reference for
DiffPhD is [d319435373a8a94aac2035bfe41f16f49bc1d2a1](https://github.com/Akinesia112/DiffPhD/tree/d319435373a8a94aac2035bfe41f16f49bc1d2a1).

The current source restores the reference's mean-weight backward operator,
nonfinite scrubbing, clipping and cascade guards. The poke example also restores
the original training contact, probe material, root-only prescription and
physical-velocity semantics. Current checks and the recorded forward diagnostic
are separated below from retained historical measurements. The native-reference
rerun has a heterogeneous backward parity failure, and the prepared poke scene
still permits deep overlap. See
[reference compatibility](reference_compatibility.md) for their scope.

The [current Hex8 regression snapshot](validation/cantilever-tests-2026-11-02.json)
passes **217 tests** on each device: CPU has **211 passes/six skips**
(125.152 s); CUDA has **216 passes/one skip** (106.214 s). Ruff checks,
generated documentation, actual demonstration media, source and wheel builds
pass. Both complete cantilever motions and UI mode-switch restart behavior are
covered. These results do not fix the recorded UR5 contact failure.

The [preceding asset and tessellation audit](validation/tessellation-2026-11-02.json)
passes complete discovery on both devices. The final CPU suite has **206 tests**,
with **201 passes and five skips** (222.223 s); CUDA on GB10 has **205 passes and
one CPU-only skip** (124.531 s). These suites add original-asset round trips,
complete embedding containment, dependent contact constraints and independent
sparse-factor checks. The original recorded deep poke still
fails; these regressions do not establish full-motion nonpenetration.

The [preceding regression report](validation/solver-validation-2026-11-02.json)
uses identical source fingerprints for both devices. That snapshot passes
**197 tests**: CPU has **192 passes and five CUDA skips** (216.967 s); CUDA on
GB10 has **196 passes and one CPU-only skip** (118.562 s). Ruff check and format
check pass for 103 Python files. Source and wheel builds pass; the wheel includes
the new public geometry and excludes the private recording. These checks
include the complete synthetic cycle, independent contact/energy tests,
reference-mode regressions and explicit failure rollback. They do not turn the
recorded UR5 failure below into a successful full rollout.

## Original Hex8 cantilever validation

The [cantilever example](../src/newton_solver_diff_phd/examples/example_softbody_diff_phd_cantilever.py)
uses the original **756-node, 500-Hex8** volume with eight Gauss samples per
cell. The baked vertices and connectivity match the source generator bit for
bit. No volume tetrahedra are present. The native nodal mass, 150 stiff cells,
36 clamped vertices, 20 muscle energies and two tip-plane energies are retained.

The default axial mode reproduces the released `cantilever_3d.py` forward
problem: rest start, 25 steps at 0.01 s and 80*t/25 N applied to every tip
vertex. Against the unchanged native source, every iteration count agrees and
the largest full-trajectory position difference is **0.196 micrometres**.
Matching the initial float32 state reduces that difference to **18.2 nm**.
Minimum sampled det(F) is **0.96617**; clamps remain exact.

The optional twist mode uses the same Hex8 environment, an explicit 30-degree
initial twist, 100 steps at **0.001 s**, and relative tolerance **1e-6**.
The paper describes this motion but does not release the initial twist state.
It oscillates from +30° through **−23.09°**, with **55.89 mm** maximum nodal
motion and minimum sampled det(F) **0.95783**. CPU and CUDA trajectories agree
with the native Hex solver to about **34 nm** over the full episode. See the
[native reference audit](validation/cantilever-reference-2026-11-02.json)
for separate unchanged-source, matched-input and step-replay measurements.

The initial attempt to use the axial 0.01 s timestep for twist is not retained:
it heavily damps the visible release and exposes sensitivity in the reference
Anderson stopping rule. The original rule tests changes in accelerated iterates
and projection right-hand sides; those can pass while a raw fixed-point
correction is larger. Tightening only the tolerance did not establish full
trajectory agreement. The smaller twist timestep is an explicit temporal
resolution choice, validated against the same native problem; solver equations
and the original axial benchmark are unchanged.

The Motion dropdown switches and restarts both modes without replacing the
viewer model. Tests compare each switched trajectory with a fresh simulation,
including graphs rebuilt for the changed timestep/tolerance. The actual OpenGL
viewer was inspected at the initial pose and during the oscillation. Mesh edges
and tip angle identify the rotation; interactive twist begins with a 1.5-second
hold of its initial state. The bundled movie records the complete axial problem.

The [performance report](validation/cantilever-performance-2026-11-02.json)
contains Nsight measurements for the actual axial video workload. Playback
frame rate and the interactive initial hold are not solver throughput.
Hex8 is forward-only; no new adjoint or UR5 contact validation is claimed.

## Bunny reference diagnostic

The [Bunny audit](validation/bunny-reference-2026-11-02.json) reproduced the
released seed-211 launch on the original mesh recipe: 1,672 vertices and 5,950
tetrahedra. All 100 port steps are finite with minimum determinant ratio 0.828.
Six independent native comparisons, including impact and the last step, agree
within **26.65 nm** in position. The independently integrated native run was
interrupted after logged step 80; it is not a completed full-trajectory check.

This scene exposes an original contact approximation: only 66 of its 1,330
boundary vertices are registered. Registered vertices stay above the floor,
but other vertices reach **5.826 mm** penetration in the port; the native
step-40 log shows **5.866 mm**. Registering the complete surface causes inverted
tetrahedra in the high-speed launch, and a gentler 10 mm drop fails the
10,000-iteration gate during impact. Those trials were not adopted as runnable
examples. Their numerical results do not establish correct contact gradients
or a working optimizer. The simpler cantilever example isolates elastic
mechanics from these unresolved contact issues.

## Original assets and tetrahedralization

The optional `--reference` asset now reproduces the original preparation from
the same 36 local scans: frame-18 convex hull, 85% surface decimation, TetGen
volume, bottom-10% clamp and 24-section cylindrical probe. The die has **6,223
vertices / 27,129 tetrahedra**; the probe has **1,165 / 4,389**. Both volumes load
from the private USD asset by name. Preparation is offline and records source
hashes and dependency versions. This corrects the missing reference manifest
and makes the actual original geometry available; the default fine mesh remains
a separate prepared approximation.

The preparation also corrects two scan embeddings missed by the former nearest
128-cell search. A conservative complete containment fallback finds their actual
containing tetrahedra. All physical geometry, materials, clamps and tool commands
remain bitwise identical after rebaking; the public Docker startup and actual
rest-frame rendering were rechecked. Genuine scan protrusion beyond the
decimated collision hull remains explicit (maximum 0.523 mm at rest), and cannot
explain centimetre-scale penetration.

The [audit](validation/tessellation-2026-11-02.json) also tests a remesh with
5,353 die vertices / 29,022 tetrahedra. It preserves the outer surface and
volume and aligns cells with the clamp plane. The lowest-percentile shape
quality improves, but the worst shape is worse because of small boundary
segments. It is **not adopted as the default**. With the contact correction
below it accepts 481 steps and rejects step 482: six well-shaped rest elements
inside the cube collapse under the probe. They have no pinned or surface nodes,
rest mean-ratio qualities 0.880–0.977, and are about 200 mm above the clamp.
All six already invert in the unconstrained mechanical update, before contact
correction. Thus fixing the old clamp element does not establish a stable full
trajectory.

A second mesh grades resolution to roughly 6 mm along the complete probe sweep,
with **12,586 vertices / 67,725 tetrahedra**. Its minimum mean-ratio quality rises
from 0.00996 to 0.02061 and first-percentile quality from 0.177 to 0.343. Boundary
refinement changes the sampled surface by at most 0.233 mm and volume by
-0.00175%. Nevertheless, the final solver rejects **step 424** for inversion,
after 423 accepted steps. Six surface elements fail; their rest qualities are
0.722–0.964 and edge lengths 2.72–6.61 mm, with no pinned nodes. Their incoming
centroids are 0.785–1.379 mm from the actual probe surface. Five already invert
before contact correction; the sixth inverts during that correction, while the
contact and QP residuals pass. This candidate is also retained only as a
diagnostic; it does not replace the runtime mesh or establish a tessellation-only
fix. These audits use the original solver rest basis, not the preceding deformed
state.

The original asset with final production code accepts **479 / 1,050** steps and
rejects **step 480**. Its single inverted tetrahedron is at the loaded surface,
has no fixed vertices, and has rest mean-ratio quality 0.527. The incoming
determinant ratio is 0.00401; the contact-corrected candidate reaches -0.14004.
The contact residuals pass their unchanged tolerances. This is a different
failure location from the older fine-mesh clamp failure below.

Matched-input penalty-contact updates agree with compiled native AlgPhD at five
steps, including contact: maximum position difference **33.62 nm**, force
difference **4.86e-12 N**, and identical iteration counts. The native original
geometry also inverts during its full unrounded AlgPhD rollout, first at step
322. This is evidence about that configuration, not proof that every positive
equilibrium is impossible. The native Python default name `pd_eigen_alg4`
actually dispatches to generic PD in the tested revision; the audit distinguishes
that method from `pd_eigen_alg_phd`.

Sparse-factor preparation now uses vectorized updates while preserving arithmetic
order. Three alternating paired measurements of the complete 3,500-row setup
give median **32.35 s → 4.35 s** (7.44×). All six factor CSR arrays are bitwise
identical, also on the full original swept and penalty matrices. This measures
CPU setup, not GPU stepping or viewer FPS. The latest complete 90-frame Nsight
graph capture records 380 graph events and zero explicit dropped records, with
collection warnings retained. Per-node captures lost records and are rejected
as quantitative performance evidence. The audit includes their raw locations.

## Optional swept-contact validation

The synthetic swept-poke demo and its scene-specific reports have been removed.
The extension and independent solver tests remain because the UR5 diagnostic
still uses them. The larger recorded UR5 demonstration is **still incomplete**.

Independent checks exposed and corrected these errors in the extension:

- Anderson accelerated the unconstrained elastic map, whose residual contains
  a nonzero contact reaction even at equilibrium. It now accelerates the
  constrained map `F(x)=P(T(x))`.
- A small accelerated update could stop the solver without establishing
  equilibrium. Stopping now tests the unaccelerated `F(x)-x` residual and the
  material-projection residual, and returns the iterate actually checked.
- Frozen contact planes could exert force after their features separated.
  Accepted response now uses current closest points and normals, geometric
  complementarity, mechanical force consistency and a fresh physical sweep.
- Loose inner QP accuracy let almost parallel contact planes produce micrometre
  displacement errors despite nanometre gap errors. The inner solve now uses
  `1e-5` times the outer contact tolerance. An analytic two-plane test checks
  the unique metric projection and both contact forces.
- An overly large numerical pivot cutoff rejected a valid active-row exchange
  for nearly duplicate contacts and sent it to a slowly converging fallback.
  The normalized cutoff is now 1e-14; the physical KKT tolerance is unchanged.
  An independently solved three-row problem checks the constrained optimum,
  with a 67-row version exercising the other factorization path.
- Truly dependent entering rows could still stall the coordinate fallback.
  A feasible descent exchange now removes a blocking old row before solving
  again, without changing the quadratic or its tolerance. Independent 3-row
  and 67-row cases check both the primal optimum and all original constraints.

The [independent equilibrium report](validation/poke-swept-equilibrium-2026-11-02.json)
checks a broad-plate press and hold for both PD and Neo-Hookean material.
A separate NumPy/SciPy energy minimization uses exact plate inequalities;
it does not reuse the production contact response or material projection.
Across four cases, maximum nodal discrepancy is **3.69 nm**, force-stationarity
residual is at most **5.05 microNewtons**, and the minimum element determinant
ratio is **0.842**. The Neo-Hookean comparison uses the reference's proximal
energy envelope, which is the forward algorithm actually implemented; it is
not a comparison to an unmodified hyperelastic stress law.

The [Nsight measurement](validation/poke-swept-performance-2026-11-02.json)
uses ten repetitions of an identical loaded physical step after warmup.
Parallel entering-contact selection reduced profiled mean step time from
**381.7 ms to 256.1 ms**, with identical final float32 positions and 14 iterations.
The trace records 71,250 kernels, including 71,190 graph kernels, with no
explicitly lost records. Both captures retain CUDA/NVTX collection warnings,
so that does not establish complete trace collection. The matched measurements
used inner QP tolerance 1e-8, before the later tightening to 1e-12. This measures
that optimization snapshot and contact step, not current full-scene FPS.
The longer attempted capture reported dropped records and was rejected as a
performance measurement. Continuous queries remain the largest measured cost.

### Full recorded scene remains unresolved

The [earlier fine-mesh rerun](validation/poke-swept-current-2026-11-02.json),
before the dependent-row correction and original-asset preparation above,
accepted **472 of 1,050** substeps, then rejected step **473** for an inverted
tetrahedron beside the fixed clamp. Contact certification passed: nonlinear
residual 4.96e-8 against 1e-7, and inner QP residual 8.40e-19 against 1e-12.
No step reached its mechanical iteration cap; the maximum was 24 iterations.
The accepted prefix reached 59.50 mm maximum nodal displacement. Its saved
states had zero sampled overlap with both the tetrahedral and displayed
surfaces; this is not a completed full-motion nonpenetration result.

An earlier run with the same strict tolerances rejected step 438 because of
the premature numerical pivot cutoff. The corrected cutoff resolves that
saved QP to residual 4.80e-19 and gets past that failure, without relaxing
physical tolerances. The final clamp failure remains.

The last accepted determinant ratio was 0.00430. Three of the failing
element's four nodes are fixed, and its
free node receives no direct active contact force. The first unaccelerated
mechanical iteration of an independently replayed earlier clamp-failure state
already inverts it. Rejecting only Anderson extrapolations therefore does not
fix that case. In the final run, the unconstrained material update also already
inverts the same element before its small contact correction.

A diagnostic that keeps every numerical iterate positive stalls while its
unchanged equilibrium residual is roughly 919 times the stopping threshold.
That experiment was rejected and is not enabled in the solver. A positive
geometric configuration exists at the complete commanded probe position; these
results do **not** prove that the command requires inversion or that all possible
positive mechanical equilibria are absent. The remaining failure needs further
numerical investigation. Rejected steps restore positions and velocities and
do not advance the example's command clock.

The [earlier 477-step report](validation/poke-swept-contact-2026-11-02.json)
and [matched-frame blue/white comparison](validation/poke-swept-blue-white-prefix-2026-11-02.json)
precede these corrections. They are historical diagnostics, not validation of
the current implementation. The old matched-prefix RMS was 30.465 mm with swept
contact versus 30.649 mm with the reference penalty. This 0.60% reduction was
not an accepted fit: initial placement and prescribed robot motion were
unfitted. No complete recording fit or full-motion nonpenetration is claimed.

## Reference-mode checks

The [fresh native-reference audit](validation/adjoint-reference-audit-2026-11-02.json)
reran eight selected DiffPhD cases against the compiled original: all forward
comparisons and seven backward comparisons passed. The known heterogeneous
backward failure below reproduced unchanged. Five separate DiffPD forward
cases also passed against that build's `pd_eigen` implementation, with matching
iteration counts and maximum nodal error below 3.0e-15 m. This is not a new build
of `diff_pd_public` or a DiffPD adjoint validation.

The [earlier reference-training test report](validation/reference-training-tests-2026-11-02.json)
records its historical source snapshot, environment and command outcomes. Its CPU suite
passed: **152 tests, 148 passed and four CUDA skips**, in 116.492 s. The CUDA
suite passed on GB10: **152 tests, 151 passed and one CPU-only skip**, in 106.878 s.
Ruff checks, formatting of all 87 Python files, and package build also passed.
The full suites preceded the final zero-volume-weight material suppression
correction. Its focused regression subsequently passed once on CPU and once on
CUDA, and the native `zero_volume_weight` comparison passed with port status
`[0, 16]`. It preserves state derivatives while suppressing the whole material
gradient, as the original nonfinite aggregate guard does. The linked report
distinguishes these post-suite checks from the full-suite source snapshot.
Coverage includes independent endpoint targets, incoming physical contact
velocities, all-particle damping, root-only probe constraints, checkpoint scene
fingerprints, and forward-only setup without an optimizer or tape.

The current [native CPU parity report](validation/parity-reference-cpu.json)
has **16 fixtures: 15 passing case verdicts and one failed backward comparison**.
Forward checks pass in all 16; 13 backward comparisons pass, one fails, and the
two grouped-energy encodings are reported without a backward verdict. It
retains the failed `pinned_contrast_10x_per_element` backward comparison. Forward
states and iteration counts pass for that case. The reference and port backward
iterations both reach their 2,000-iteration budget; the maximum state/force
relative discrepancy is 4.35%, and a material-energy derivative differs by
17.46%. These exceed the unchanged tolerances and are **not accepted as parity**.
Mean-weight adjoints of heterogeneous forward weights are approximate and can
be unstable; matching equations does not guarantee matching a long unconverged
Anderson trajectory. Grouped reference material encodings use a different
energy representation, so their backward comparisons are reported but unjudged.

The [adjoint sensitivity diagnostic](validation/reference-adjoint-sensitivity-2026-11-02.json)
shows the numerical limit of the failed case: the two implementations agree
within 2.7e-8 relative error at the measured budgets through 50 backward iterations,
then diverge during the unconverged Anderson trajectory. In the compiled
reference alone, deterministic Gaussian perturbations of free-node output
positions scaled by 1e-15 m change the force gradient by 7.07%; scaling the same
perturbations by -1e-14 m activates the norm guard and zeros that gradient. This sensitivity
explains why the failure is retained instead of widening its tolerance. A
separate 10× contrast fixture at 1 ms converges and passes the existing bounds;
it does not replace the failed 4 ms fixture.

The actual main-checkout Docker launcher completed this command with exit zero,
including all 1,050 forward substeps on CUDA:

```bash
./docker/run.sh --headless softbody_diff_phd_ur5_poke --forward-only --num-frames 36 --test
```

This `--test` checks numerical output; it is not a nonpenetration test. The
geometric diagnostic below records the observed overlap separately.

## Current recorded forward diagnostic

The [original-training forward report](validation/poke-reference-training-2026-11-02.json)
runs the current `build_poke_scene` and untaped `Example._Forward` on the GB10.
Both placements completed all 1,050 substeps at 1/900 s, with solver status zero,
no iteration caps and no inverted die or probe tetrahedra at recorded frames.
The prepared scene has 3,663 die vertices, 18,624 die tetrahedra, and a 26-vertex,
36-tetrahedron probe with 13 prescribed root vertices.

**This run fails a nonpenetration requirement.** The original penalty law allows
the probe to pass deeply through the prepared die despite finite, converged
steps. These are measured outcomes, not a new accepted material fit:

| Placement | Maximum sampled probe depth inside die | Inside displayed surface | Maximum die displacement | Minimum die determinant |
| --- | ---: | ---: | ---: | ---: |
| Default initial XY (0.73, 0.115) m | 120.58 mm | 117.29 mm | 3.05 mm | 0.9421 |
| Former fitted XY, incompatible old physics | 97.00 mm | 93.58 mm | 2.39 mm | 0.9541 |

Each probe face is sampled at 45 barycentric locations against the oriented
tetrahedral boundary and embedded display surface; die boundary vertices are
also queried against the probe. These sampled distances detect the observed
overlap but do not prove global intersection bounds. Inversions and geometry
are checked at recorded frames, while status, iteration counts and contact
pairs are checked every substep. Maximum free-probe tracking error was 0.166 mm
at the default placement; prescribed-root error stayed below 0.000033 mm.

This establishes that the visible pass-through persists with the requested
original equations on the prepared geometry. It does not establish that the
original generated scene behaves identically. No new contact law, inversion
guard, material projection or clamp remeshing was used. Recorded durations
include synchronization and geometric diagnostics and are not throughput
benchmarks. The report records a concurrent adjoint-source change; the forward
solver and example source hashes did not change during this untaped run.

The [blue/white surface comparison](validation/poke-blue-white-comparison-2026-11-02.json)
evaluates the same saved default forward rollout against all 36 recorded scans.
No registration or per-frame alignment is applied. Across frames 19–53, the
symmetric vertex-to-triangle RMS is **31.216 mm**, compared with **31.227 mm**
when the initial white surface is held completely fixed: only **0.034%**
improvement. The original sampled training metric likewise gives 32.053 mm
versus 32.064 mm. A completed solver rollout therefore does not demonstrate
useful agreement with the recording.

The highest point of the blue surface falls by up to **27.789 mm** from frame
18; the white surface's highest point falls by only **1.901 mm**. These are
geometric extrema, not tracked material displacements. The default white cube
starts 30 mm in X and 26 mm in Y from the blue cube; both the simulation and the
frozen baseline include this placement error. The comparison uses every mesh
vertex against the opposite triangular surface, weights both directions and
frames equally, and does not weight by triangle area. It is separate from the
signed overlap diagnostic above.

The tool `tools/compare_poke_recording.py` accepts a prepared `--recording`,
a `--trajectory` NPZ, and `--positions-key` (default `positions`; training
exports use `optimized` or `initial`). It requires one state per recorded frame
including frame 18 and writes aggregate/per-frame metrics with `--output`.
Optional `--distances-output` saves private local arrays for inspection.
Recorded cube meshes are measured data, but the probe motion is the reference
script's hand-authored trajectory and its 30 Hz timing is assumed. No measured
probe poses or timestamps are present in the local recording. The comparison
does not validate those experiment inputs or repair the contact failure.

The [contact-shell audit](validation/poke-contact-shell-2026-11-02.json)
re-evaluates normal forces on the saved frame-end positions. At frames 38–45,
both vertex/triangle directions have zero active pairs despite deep overlap.
The source's unsigned 0.55 mm contact shell has no inside/outside recovery:
after crossing a face, a vertex can be pushed farther inside before leaving
the shell. The port matches that source formula and its opposite reactions.
This diagnostic does not reconstruct velocities, friction, or the original
start-of-substep force history.

## Retained checks before the original-training migration

| Check | Evidence and scope |
| --- | --- |
| CPU unit tests | **Passed:** 138 tests, 4 CUDA skips, 110.010 s after the training correction. The suite covers independent NumPy oracles, supported finite differences, API migration, contact regressions, and state-only/late-enabled parameter gradients. See [training audit](validation/training-performance-2026-11-02.json). |
| Ruff and package build | **Passed.** Build used the cached uv-build backend without isolation after network resolution failed; both solver families are packaged and private recordings are excluded. See [training audit](validation/training-performance-2026-11-02.json). |
| Executed C++ reference parity | [CPU report](validation/parity-reference-cpu-pre-reference-training.json): 13 forward cases; gradients compared in 11 cases and reported without acceptance in the two per-element heterogeneous cases. |
| CUDA unit tests | **Passed on NVIDIA GB10:** 138 tests, 1 CPU-only capture skip, 100.832 s, including first-use captured backward and late-enabled parameter gradients. The initial audit's CUDA error 100 reflected sandbox visibility; GPU execution works outside that restricted command sandbox. See [training audit](validation/training-performance-2026-11-02.json). |
| Recorded UR5 position fitting | **Executed:** complete initial forward/backward evaluation and one accepted optimizer update with default numerical settings. Loss decreased from 0.03069488 to 0.01967943 m². Some forward steps still reached the iteration cap. This is not a full training convergence or penetration acceptance test. |
| Bear training and recorded media | **Not revalidated for this revision.** Historical recordings and fits do not establish the corrected solver's behavior. |
| Paper equations | No new paper-fidelity claim. The numerical target here is executable reference code. |

The Python environment already contains Newton 1.7.0 and Warp 1.17.0. A locked
uv invocation attempted to fetch the Warp wheel and failed because DNS was
unavailable. Those checks therefore used `uv run --no-sync` with that existing
environment, without changing dependency pins. The report records the actual
Python, platform, device, package versions, source hash and repository revision.

## Reference parity

The linked 13-case report below is from before the restored mean-weight adjoint
and reference safeguards. Its fixture verdicts are preserved as historical
records; grouped-material and per-element encodings must be distinguished when
rerunning against the current mean-weight convention.

The reference runs in its own Python environment and calls the compiled
`TetDeformable.PyForward` and `PyBackward` with `pd_eigen_alg_phd`. The NumPy
oracle is a separate local transcription, not the source of these reference
outputs. The reference build uses the documented ARM64 build patch, with no
changes to solver sources. Its revision, working-tree changes and shared-library
hash are included in the [CPU report](validation/parity-reference-cpu-pre-reference-training.json).

The reference binary was built with CUDA support, but it also runs here without
a CUDA device: cuSPARSE initialization fails and the mechanical inverse uses
the reference's CPU fallback. These are CPU parity results; they do not validate
CUDA execution or establish a speedup.

| Fixtures | Accepted comparisons |
| --- | --- |
| `homogeneous_free`, `multistep_pinned` | Positions, velocities, convergence, iteration counts, state/force/material derivatives. |
| `pinned_contrast_{10,100}x_grouped` | Same checks with one scalar energy per material group in the reference. |
| `pinned_contrast_{10,100}x_per_element` | Forward checks only. Gradient differences are recorded and explicitly excluded from the passing verdict. |
| `neo_hookean_large_deformation` | Forward and reference-formula derivatives. Agreement with that formula does not make its material gradient a finite-difference derivative. |
| `muscle_activation` | Forward, state/force/material and contraction derivatives. |
| `plane_impact`, `sticking_friction`, `sliding_friction` | Three-step states, convergence, active sets, multipliers, planar gaps and reference backward outputs. |
| `mesh_contact_poke` | Three steps with the two-body penalty state force, including force differences; its derivative is held fixed on both sides. |
| `mesh_boundary_poke` | One step with independently evaluated mesh frictional boundary contact, active sets, multipliers and backward outputs. |

The position threshold is
`10 * (rel_tol * ||q_ref|| + abs_tol) + 10 * k * u32 * (||q_ref|| + dt * ||v_ref||)`,
where `k` counts earlier float32 state stores and `u32 = 2^-24`.
Velocity uses that bound divided by `dt`; iteration counts may differ by at
most one and convergence flags must agree. Free-particle gradients use relative
`1e-5`, or `1e-4` when contact is active. Multiplier tolerance accounts for the
reference trace's six significant digits. All thresholds predate this rerun and
are preserved in `tools/parity/compare.py` and the report.

All native-reference iteration counts matched exactly. Maximum position error
across these fixtures was `1.26e-7` m (the four-step pinned rollout). Maximum
accepted relative state/force/activation gradient error was `1.18e-7`; material
comparisons use the separate per-energy bounds recorded in the report.

The comparison command now fails for failed numerical checks, incompatible
fixtures, an empty run, or the wrong reference revision. Misspelled case names
are rejected. Six harness regressions cover these failure paths. A passing case
means its **judged** comparisons passed; the report separately lists cases whose
gradients were not judged.

## Corrections to the poke handoff

The scratchpad handoff's claim that `converged_contact=True` solves each contact
LCP exactly is incorrect. Its fixed budget of 16 block pivots can stop at a
non-complementary result: a reproduced positive-definite five-contact case left
a gap of approximately **−1.465** with a zero multiplier. The option and its
alternate solve path have been removed.

The experimental signed SVD was also inconsistent with the reference's unsigned
projection formulas: a reflected deformation produced NaNs in the volume
projection. The corrected path uses the reference unsigned SVD; a regression
steps reflected PD and Neo-Hookean tetrahedra and checks finite stationary
results.

Mesh candidates are again selected from the **start-of-step** surface, as in
the reference. The previous look-ahead to prescribed endpoint positions changed
the default algorithm even with experimental flags disabled. A regression moves
a prescribed block into the candidate window and checks that contacts are
admitted on the following step, matching the independent oracle and iteration
counts.

These corrections remove demonstrated discrepancies. They do not establish that
the recorded UR5 rollout has zero penetration, converges at every frame, or has
useful optimization gradients. The reference itself can reach its iteration
limit, and its contact derivatives have the restrictions below.

## Derivative limits

- The backward operator now uses the reference global mean weight per energy
  family. With heterogeneous per-element forward weights it is an approximate
  derivative and can become indefinite. The old 10×/100× comparison report
  predates this correction.
- Reference nonfinite scrubbing, the `1e6` adjoint norm clip, divergence
  suppression and cascade rescaling can alter or suppress derivatives. Passing
  finite differences on an unclipped branch does not validate clipped branches.
- State gradients use the reference's contact-free backbone. Woodbury contact
  projection affects material and activation gradients. This is not the exact
  derivative of a rollout that recreates and resolves contact after perturbation.
- `mesh_contact` is an explicit state force whose reference backward is zero.
- Neo-Hookean material gradients reproduce the reference hyperelastic-stress
  formula, which differs from the derivative of its proximal forward step.
- If every PD volume weight is zero, the reference's stress divided by mean
  weight becomes nonfinite and suppresses the aggregate material gradient.
  This port reproduces that suppression and sets status 16. Native parity on
  `zero_volume_weight` checks that behavior; it is not a mathematical
  finite-difference derivative of the zero-weight forward model.
- Dirichlet-position gradients are an added feature, checked by finite differences
  in the supported no-contact branch.

## Historical evidence

The retained [CUDA report](validation/parity-reference-cuda.json) and
`test-results.json`, `poke-recorded-acceptance-cuda.json` and retained media files are
**historical artifacts from earlier source versions**. They are not fresh
validation of this replacement. The CUDA report contained 12 cases and did not
include `mesh_boundary_poke`, despite the previous compatibility table citing it.
Earlier prose also claimed 48×/300× speedups while that JSON recorded different
values; no performance claim is carried forward here.

For `SolverDiffPD`, current unit tests compare its forward step against an
independent dense transcription. The separate 200-step duck C++ comparison in
[DiffPD method](method_diff_pd.md) is historical and has not been rerun in this
session. `SolverDiffPD` does not implement an adjoint.

## Parity fixture schema

The harness exchanges NPZ files with schema `newton.diff_phd_parity.v1`. Arrays
use SI units, canonical particle order and positively oriented tetrahedra.
Numerical fixture values are rounded to float32 before both runs, then stored
as float64, reflecting Newton's public state and material storage.

Inputs include `rest_vertices (N,3)`, `tets (E,4)`, `density`, per-element `mu`
and `lambda`, `material`, `dirichlet (N)`, initial `q`/`v`,
`f_ext (steps,N,3)`, `gravity`, `dt`, solver tolerances and iteration budgets.
Optional actuation uses `muscle_directions (E,3)`, `muscle_stiffness (E)` and
`activations (steps,E)` with contraction `u = 1 - a`. Contact fields contain the
boundary kind, parameters, registered vertices, friction and restitution;
penalty mesh contact supplies parameters and both face arrays. `objective_q`
and `objective_v` define a linear objective on the final state.

Outputs contain per-step `q_next`, `v_next`, iteration counts and convergence;
contact multipliers/active sets and planar gaps where available; and
`dL_dq`, `dL_dv`, `dL_df_ext`, material and optional activation derivatives.
The penalty mesh case also records force differences and pair counts. Elastic
energies, general contact-force vectors, independent force-balance residuals,
and nonplanar gap comparisons are **not currently covered by this harness**.

## Reproduce

With the locked environment available normally:

```bash
uv sync --locked --extra examples
uv run --locked ruff check .
uv run --locked ruff format --check .
uv run --locked python -m unittest discover -s tests -v
NEWTON_TEST_DEVICE=cuda:0 uv run --locked python -m unittest discover -s tests -v
```

Rerun the current native-reference fixtures with:

```bash
export UV_CACHE_DIR=/tmp/diffphd-audit-uv
export WARP_CACHE_PATH=/tmp/diffphd-audit-warp
export PYTHONPATH=src
uv run --no-sync python tools/parity/port_cases.py \
  --out /tmp/diffphd-reference-cpu-2026-11-02 --device cpu \
  --reference-root /home/beer/Dev/Robotics/DiffPhD-reference-build
uv run --no-sync python tools/parity/compare.py \
  --results /tmp/diffphd-reference-cpu-2026-11-02 \
  --reference-root /home/beer/Dev/Robotics/DiffPhD-reference-build \
  --report docs/validation/parity-reference-cpu.json
```

The earlier raw public fixtures, both outputs, per-case reference logs and combined run
log are retained locally under
`outputs/reference-audit-2026-11-02/native-parity/` (ignored by git).

## Historical poke training performance correction

The following measurements used the earlier NCP scene: all probe particles
prescribed, die material on both bodies, 1.5 mm contact radius and a 2,000-step
iteration cap. They are incompatible with current version-5 checkpoints and
the restored original-training scene. The state-only Woodbury omission remains
valid, but the current example's `mesh_contact` state force does not use that
NCP contact projection.

State-only differentiation skips the Woodbury contact projection, whose
result is used only for material and activation gradients. State and prescribed
position gradients retain their existing formula. Parameter fitting still runs
the complete projection, and contact snapshots remain available if gradients
are enabled after recording.

On the GB10, a matched diagnostic of the last **60 backward steps** took
102.39 s before and 0.67 s after the change. Both ran the complete 1,050-step
forward trajectory first and produced identical observation-position hashes.
The baseline CUDA/NVTX trace attributes 104.12 s of a separately profiled
105.05 s backward segment to contact response batch graphs. This is a partial
backward comparison, not a full-training speedup claim. Boundary gradients
differed by relative L2 3.29e-6 (maximum absolute error 1.19e-7); they are not
bitwise identical and are within the existing 1e-4 contact comparison bound.

The earlier example was also profiled with its then-default fine mesh (3,689
particles, 18,660 tetrahedra), 35 observations × 30 substeps, friction 0.5,
1.5 mm contact radius, relative tolerance 1e-4 and 2,000-iteration budget.

| Measured work | Forward | Backward | Total |
| --- | ---: | ---: | ---: |
| Initial evaluation, including first solver setup | 67.80 s | 16.69 s | 84.50 s |
| First accepted optimizer update, no backtracking | 5.04 s | 16.37 s | 21.41 s |

These are wall times under Nsight with the null viewer. Scene/robot setup took
another 34.30 s, including cold kernel compilation. An unprofiled complete
initial evaluation measured 66.00 s forward and 14.79 s backward. The trace
records CUDA graph replay for the inner solves; graph capture does not remove
their iterations. The first-step CPU sparse factor setup, remaining adjoint
solves, and any repeated line-search rollouts are still significant costs.
Viewer FPS includes offline training pauses and is not solver throughput.

To collect a new full initial evaluation plus one update of the **current**
configuration, with the prepared local recording present, run the command below.
It does not reproduce the historical timing table's physics:

```bash
./docker/run.sh --nsys softbody_diff_phd_ur5_poke \
  --no-video-args --frames 38 --warmup 0 --cuda-graph-trace graph \
  -- --train-iters 1 --save-alignment /tmp/poke-profile-alignment.json
```

Raw traces and diagnostics are retained under ignored `nsys/results/`. The
[training audit](validation/training-performance-2026-11-02.json) records their
paths, timing scope and validation results. Whole-graph tracing preserves
conditional graph execution but does not measure individual kernels inside
those graphs. No mesh, substeps, physical parameters or convergence budgets
were reduced for this optimization.

Reference build instructions and further case details are in
[tools/parity](../tools/parity/README.md). Recorded poke experiments require the
user's separately prepared local recording; no private data is redistributed.
