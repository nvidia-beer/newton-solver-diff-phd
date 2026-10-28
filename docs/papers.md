# Papers mapped to this implementation

The methods below are linked to executable code, with departures stated explicitly.

| Paper | Implemented portion and code |
| --- | --- |
| Lai et al., [DiffPhD (2026)](https://arxiv.org/abs/2605.14526), Algorithms 1–2, §§4.3–4.8 | Material-weighted sparse preconditioning, persistent inverse factors, bounded Anderson acceleration, contact response and implicit adjoints. See the mapping below. The complete published algorithm is **not** reproduced. |
| Zeng et al., [Fast But Accurate (2025)](https://doi.org/10.1145/3730834), Algorithm 4, §§4–6 | Sparse inverse-factor applications and reusable contact compliance in [SparseInverse](../src/newton_solver_diff_phd/_src/solvers/diff_phd/sparse.py) and [ContactResponse](../src/newton_solver_diff_phd/_src/solvers/diff_phd/response.py). This implementation uses natural-map friction residuals and safeguarded Newton solves, rather than the paper's exact NCP/CR pipeline. |
| Du et al., [DiffPD (2021)](https://doi.org/10.1145/3490168), Algorithms 1–2, §4 | Projective rotation/volume derivatives in [elasticity.py](../src/newton_solver_diff_phd/_src/solvers/diff_phd/elasticity.py); fixed-point dynamics and the converged-system adjoint in [solver.py](../src/newton_solver_diff_phd/_src/solvers/diff_phd/solver.py) and [adjoint.py](../src/newton_solver_diff_phd/_src/solvers/diff_phd/adjoint.py). Its static-contact Woodbury Algorithm 4 is not used. |

## DiffPhD pseudocode to code

| Paper step | Current implementation |
| --- | --- |
| Algorithm 1, 1–6: assemble and cache the material-dependent factor | [SolverDiffPhD._prepare](../src/newton_solver_diff_phd/_src/solvers/diff_phd/solver.py) builds the scalar nodal matrix with per-element weight $2\mu_e+\lambda_e$; [SparseInverse.__init__](../src/newton_solver_diff_phd/_src/solvers/diff_phd/sparse.py) creates exact sparse LDLT inverse factors; `apply`/`apply_batch` use them on the selected device. |
| Algorithm 1, 7–10: predictor and contact columns | [kernels.predictor](../src/newton_solver_diff_phd/_src/solvers/diff_phd/kernels.py), [ContactResponse.update](../src/newton_solver_diff_phd/_src/solvers/diff_phd/response.py), `make_columns` and `make_compliance`. The compliance matrix is rebuilt when frozen contact geometry changes. |
| Algorithm 1, 12–19: mechanics and constrained update | [SolverDiffPhD._solve](../src/newton_solver_diff_phd/_src/solvers/diff_phd/solver.py) applies the inverse to the physical force residual, then [ContactResponse.solve](../src/newton_solver_diff_phd/_src/solvers/diff_phd/response.py) proposes the frictional correction. `material="pd"` has local rotation/volume projections; `neo_hookean` uses the logarithmic energy gradient. The latter does **not** execute the paper's proximal local step. |
| Algorithm 1, 20–32: acceleration and acceptance | [_solve](../src/newton_solver_diff_phd/_src/solvers/diff_phd/solver.py) implements bounded regularized Anderson mixing for contact-free iterations, checks both force residual and update size, and falls back to [_newton_candidate](../src/newton_solver_diff_phd/_src/solvers/diff_phd/solver.py) with a physical KKT tangent. This differs from the paper's dual-gate stopping rule. |
| Algorithm 1, 33–34: state and backward cache | [kernels.write_state](../src/newton_solver_diff_phd/_src/solvers/diff_phd/kernels.py), then [record_adjoint](../src/newton_solver_diff_phd/_src/solvers/diff_phd/adjoint.py) stores independent primal/contact snapshots only when a Tape tracks the step. |
| Algorithm 2, 1, 16 onward: seed, transpose solve and parameter gradients | [record_adjoint.backward](../src/newton_solver_diff_phd/_src/solvers/diff_phd/adjoint.py), `_seed`, [KKTSystem.solve(transpose=True)](../src/newton_solver_diff_phd/_src/solvers/diff_phd/adjoint.py), `_position_vjp`, `_velocity_vjp`, `_force_vjp` and `elasticity.material_vjp`. The matrix-free solve checks its true residual and uses the existing inverse as a preconditioner. |
| Algorithm 2, 2–15: trust-region filtered proximal Hessian | **Not implemented in the stepping adjoint.** [KKTSystem](../src/newton_solver_diff_phd/_src/solvers/diff_phd/adjoint.py) uses the unfiltered physical tangent. Standalone [neohookean_prox*](../src/newton_solver_diff_phd/_src/solvers/diff_phd/elasticity.py) routines exist but are not called by `SolverDiffPhD.step`. |

The paper's stiffness-scaled Rayleigh term is also absent: `mass_damping` is
mass-proportional only. No-contact backward still solves the physical tangent;
it does not reduce to one inverse application. Contact detection and branch
selection are held fixed during differentiation. See [method](method.md) for
the exact equations and [validation](validation.md) for measured checks.
