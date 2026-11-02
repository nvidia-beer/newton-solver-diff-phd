# Papers and reference code

The default forward solver follows the **executed code** of the public DiffPhD
reference. Backward deviations are explicitly listed in the compatibility
table; this is not a claim of full repository or paper equivalence. The authority is
[Akinesia112/DiffPhD at d319435](https://github.com/Akinesia112/DiffPhD/tree/d319435373a8a94aac2035bfe41f16f49bc1d2a1).
Where the reference README and its code disagree, the code was followed.
[Reference compatibility](reference_compatibility.md) has the line-level mapping.

| Paper | Relation to this implementation |
| --- | --- |
| Lai et al., [DiffPhD: A Unified Differentiable Solver for Projective Heterogeneous Materials in Elastodynamics with Contact-Rich GPU-Acceleration (2026)](https://arxiv.org/abs/2605.14526), [arXiv:2605.14526v1](https://arxiv.org/html/2605.14526v1) (14 May 2026) | The reference implements this paper's AlgPhd solver. Sections 4.1–4.9 were inspected during this audit. The mapping below identifies implemented steps and omissions; parity is judged against executable reference code (see [validation](validation.md#reference-parity)). |
| Du et al., [DiffPD (2021)](https://doi.org/10.1145/3490168) | The reference builds on the DiffPD code base: projective local/global steps, rotation/volume projections and the projective-dynamics backward fixed point. The Woodbury contact projection in the backward pass is the reference's version. |
| Zeng et al., [Fast But Accurate (2025)](https://doi.org/10.1145/3730834) | The reference applies $A^{-1}$ through exact sparse inverse factors (RealSim `SparseLDLT` with METIS). This package builds equivalent exact factors with its own nested-dissection ordering. |

Behaviour that a paper may describe but the reference does not execute is absent:

- Eigenvalue filtering. It appears in the reference README; no executed code
  filters eigenvalues.
- A trust-region filtered Hessian in the backward pass.
- Stiffness-scaled Rayleigh damping.
- Self-contact.

The reference's nonfinite scrubbing, gradient clipping and divergence/cascade
guards are reproduced in the backward pass. They are part of the executable
algorithm and can suppress gradients. Intentional additions and derivative
limitations are listed in the compatibility table.

## DiffPhD paper mapping

Source: [arXiv v1, Section 4](https://arxiv.org/html/2605.14526v1#S4).
The source-code compatibility table gives the numerical conventions where
paper and executable differ.

| Paper step | Implementation and scope |
| --- | --- |
| Eq. 11; Algorithm 1, lines 13–14 | `kernels.local_step`, `global_rhs`, `SolverDiffPhD._build`; reference mean-volume convention. |
| Eqs. 13–15 | `nh_prox_stretches`, `nh_prox_hessian`; executable Newton objective and derivatives govern parity. |
| Eqs. 20–22; Algorithm 1, lines 20–30 | `Anderson.mix`, `check_convergence`; reference default window 5. |
| Eqs. 32–33 | `SparseInverse.apply`; exact factors, different ordering. |
| Algorithm 1, lines 16–19 | `FrictionalContact.iterate`; one NCP Newton step. |
| Eqs. 23–29; Algorithm 2, lines 3–14 | Adaptive Hessian filtering is absent from the selected executable path and this port. |
| Eqs. 30–31 | Rayleigh damping is not implemented. |
| Algorithm 2, lines 16–26 | `Adjoint`; the backward projection Jacobian uses the reference mean weight per energy family, with its clipping and divergence guards. Dirichlet-target derivatives are an addition; contact and NH parameter derivative limitations remain. |

## Cantilever Hex8 example

[Section 5.2](https://arxiv.org/html/2605.14526v1#S5.SS2) describes a cantilever
with a stiff middle segment under twist release and axial loading.
`softbody_diff_phd_cantilever` now uses the original 756-node, 500-Hex8 grid,
including its original vertex/cell ordering, 150 stiff cells, root clamp and
2×2×2 Gauss integration. The previous 3,000-tetrahedron adaptation is removed.

The default reproduces the forward problem in the released
[`cantilever_3d.py`](https://github.com/Akinesia112/DiffPhD/blob/d319435373a8a94aac2035bfe41f16f49bc1d2a1/python/example/cantilever_3d.py):
rest initial state, zero gravity, original muscle/vertex-plane energies, 25
steps at 0.01 s and the per-tip-vertex axial ramp 80*t/25 N. Native comparisons
use `HexDeformable` and execute the source `CantileverEnv3d` setup directly.
Newton stores output states as float32 while the native solver uses float64;
trajectory comparisons report this numerical difference rather than claiming
bitwise identity.

`--motion twist` runs the same Hex environment from an explicitly chosen
30-degree twist, with no axial ramp, 100 steps at 0.001 s and relative
tolerance 1e-6. This resolves the twist more finely than the axial script. The source does not supply the paper's
initial twist state, so this remains a specified release scenario, not an
exact reproduction of an unavailable trajectory. The Hex8 branch is forward
only and rejects recorded Warp Tapes; it makes no new gradient claim.

## Bunny reference audit

The released bunny task from [Section 5.4.2](https://arxiv.org/html/2605.14526v1#S5.SS4.SSS2)
was evaluated independently using the original simplified OBJ, offline mesh
recipe, 100× material contrast and seed-211 launch. Six native matching-input
steps agree within 26.65 nm. The original 66-vertex floor contact subset leaves
other boundary vertices penetrating by about 5.8 mm in both solvers. Expanding
the subset to every surface vertex produces inverted elements in the fast
launch; a gentler drop failed the convergence gate. These are diagnostic
results, not a validated contact demonstration. No Bunny optimizer or runnable
viewer example is exposed.

The native scene registers four grouped energies. The port's forward weights
match those groups, while its backward per-family mean weights do not reproduce
the grouped local Jacobian. Native AlgPhD state adjoints also omit contact,
making launch gradients through impact approximate even after fixing that
grouping mismatch. See the [validation report](validation.md).

## `SolverDiffPD`

`SolverDiffPD` follows the executed code of
[mit-gfx/diff_pd_public at 1594aa0](https://github.com/mit-gfx/diff_pd_public/tree/1594aa07a6983ff9a9c107e9f0d83d07fa99c8d1),
the code release of Du et al., [DiffPD (2021)](https://doi.org/10.1145/3490168).
Only the forward step is ported; the paper's backward pass is not implemented yet.
Paths are relative to `cpp/core/src/` in the reference and to
`src/newton_solver_diff_phd/_src/solvers/diff_pd/` in this package.

| Reference function | Implementation |
| --- | --- |
| `TetDeformable::InitializeFiniteElementSamples` (`fem/tet_deformable.cpp`), `Mesh::ComputeElementVolume` and the mean volume in `Mesh::Initialize` (`mesh/mesh.cpp`) | `SolverDiffPD.__init__` (`solver.py`) |
| `Deformable::SetupProjectiveDynamicsSolver` (`fem/deformable_projective_dynamics_forward.cpp`): $A=\tfrac{m}{h^2}I+w\bar V\sum_e G_eG_e^T$ with Dirichlet rows | `SolverDiffPD._build` (`solver.py`), factor in `SparseInverse` (`sparse.py`) |
| `Deformable::ForwardProjectiveDynamics`: `rhs` and $v=(x-q)/h$ | `begin_step`, `write_state` (`kernels.py`) |
| `Deformable::ForwardStateForce`, `GravitationalStateForce`, `PlanarContactStateForce::ForwardForce`, `ArcContactStateForce::ForwardForce` (`state_force/`) | `_pack_state_force` (`solver.py`), `_state_force`, `_penalty`, `_friction` (`kernels.py`) |
| `Deformable::ComputePdEnergy`, `PdEnergyForce` (`fem/deformable_pd_energy.cpp`), `PdElementEnergy::EnergyDensity` (`pd_energy/pd_element_energy.cpp`) | `evaluate_elements`, `gather`, `objective_partials` (`kernels.py`) |
| `CorotatedPdElementEnergy::ProjectToManifold`, `VolumePdElementEnergy::ProjectToManifold`, Eigen `JacobiSVD` (`pd_energy/`) | `unsigned_svd`, `corotated_projection`, `volume_offset` (`projections.py`) |
| `Deformable::PdNonlinearSolve` with `use_bfgs = 1`: L-BFGS two-loop recursion with $H_0=A^{-1}$ and the gradient fallback | `SolverDiffPD._iterate` (`solver.py`); `history_advance`, `history_store`, `history_rho`, `slot_dot_partials`, `set_alpha`, `set_coefficient`, `slot_axpy`, `direction_partials`, `choose_direction`, `fallback_direction` (`kernels.py`) |
| `PdNonlinearSolve` Armijo line search | `SolverDiffPD._line_search` (`solver.py`); `trial_position`, `line_search_check`, `accept` (`kernels.py`) |
| `PdNonlinearSolve` convergence test | `threshold_partials`, `check_convergence` (`kernels.py`) |
