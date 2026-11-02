# DiffPD projective dynamics (`SolverDiffPD`)

`SolverDiffPD` ports the forward step of the public DiffPD reference
([mit-gfx/diff_pd_public at 1594aa0](https://github.com/mit-gfx/diff_pd_public/tree/1594aa07a6983ff9a9c107e9f0d83d07fa99c8d1)):
`Deformable::ForwardProjectiveDynamics` and `PdNonlinearSolve` for the method
`pd_eigen` with `use_bfgs = 1`, the options of the duck example. It is a
separate solver family from `SolverDiffPhD` in
`src/newton_solver_diff_phd/_src/solvers/diff_pd/` and shares no code with it.

## Implemented

- Linear tetrahedra (`TetDeformable`, one sample per element), as the duck
  uses. Element gradients $G_e$ come from float64 rest positions
  (`rest_positions`, default `model.particle_q`).
- Uniform mass $m=\rho\bar V$ for every vertex and $\bar V$ as every element's
  volume, where $\bar V$ is the mean rest volume (the reference's own TODO says
  this mass is not lumped).
- PD energies with global scalar weights, in reference order:
  corotated $\tfrac{w_c}{2}\lVert F-UV^T\rVert^2$ and volume
  $\tfrac{w_v}{2}\lVert F-U\,\mathrm{diag}(\sigma+D)V^T\rVert^2$, each times
  $\bar V$, with Eigen's unsigned Jacobi SVD and the reference's 50-step volume
  projection. `material_type='none'`: no elastic material term.
- State forces evaluated once at the step start $(q, v)$: gravity $mg$, then
  `planar_contact` and `arc_contact` in the given order with the reference's
  parameter lists (penalty $k_n\max(0,-d)^{p-1}$, friction
  $-\min(k_f,\mu f_n/(\lVert u_s\rVert+\epsilon))u_s$, the arc's strict angular window).
- $\mathrm{rhs}=q+hv+\tfrac{h^2}{m}f_{ext}+\tfrac{h^2}{m}f_{state}(q,v)$ and
  $A=\tfrac{m}{h^2}I+(w_c+w_v)\bar V\sum_e G_eG_e^T$ per coordinate. Dirichlet
  (zero-inverse-mass) particles have unit rows, skipped couplings, a masked
  gradient and stay at their start positions.
- From $x=q$, L-BFGS on
  $\Phi(x)=\tfrac12\tfrac{m}{h^2}\lVert x-\mathrm{rhs}\rVert^2+E_{pd}(x)$ with
  $H_0=A^{-1}$, `bfgs_history_size` pairs, no scaling or curvature check, the
  gradient fallback when $d\cdot g\le0$, and Armijo backtracking
  ($\gamma=10^{-4}$, halving, at most `max_ls_iter` halvings, absolute
  `DBL_EPSILON` slack, the last trial accepted untested). It stops when
  $\lVert g\rVert\le$ `rel_tol` $\lVert\tfrac{m}{h^2}\mathrm{rhs}\rVert+$ `abs_tol`
  and sets $v=(x-q)/h$. Defaults are the duck's options (4000, 10, $10^{-9}$,
  $10^{-4}$, 10).

All solver arithmetic is float64 in Warp kernels. The iteration, the L-BFGS
ring buffer and the line search run on the device as nested
`wp.capture_while` loops: one cached CUDA graph per step, or the caller's graph
when it captures `step`; CPU devices run the same kernels from host loops.
Reductions sum fixed chunks in order. `A^{-1}` is applied by an exact sparse
inverse factor of the same matrix (the reference uses `SimplicialLDLT`).
Element forces are gathered per vertex in element order, as the reference's
serial scatter.

## Differences from the reference

- Newton states are float32: every step starts from float32-rounded $q, v$.
  The float64 result of the last step is available as `positions` and `velocities`.
- The reference's exceptions ("PD method fails to converge.", "Singular F.")
  become status bits that `check_status` raises; a singular projection stops
  the step, and the output state then holds the singular trial point.
  Bit 4 reports a line search that used all halvings (silent in the reference).
- A changed `dt` refactors $A$; the reference freezes the matrix at the first step's `dt`.
- `model.gravity` is used unless `gravity` gives the float64 vector.
- `model.tet_materials` is ignored; the energy weights are constructor arguments.
- Rounding only: the energy and objective sum fixed chunks of 16 elements or
  vertices in parallel, where the reference sums element energies sequentially
  and uses Eigen's vectorized dot; $G_e$ uses NumPy's LU inverse instead of
  Eigen's cofactor inverse. Element volumes and their mean use the reference's
  operation order.

## Not implemented yet

- The backward pass and gradients (a recorded `wp.Tape` raises `NotImplementedError`).
- The duck's optimization (`duck_3d.py` fits `kn`, `kf`, `mu` by L-BFGS-B;
  its gradient bugs B3 and B4 apply there).
- `use_bfgs = 0`, frictional boundaries and the contact active set, actuation,
  muscles, vertex energies, elastic materials, hexahedral elements and the
  other state forces (`hydrodynamics`, `billiard_ball`).

## Duck scene

`examples/example_softbody_diff_pd_duck.py` is the forward rollout of
diff_pd_public `python/example/duck_3d.py` with `python/py_diff_pd/env/duck_env_3d.py`
at the initial contact parameters `state_force_parameters = [0, 0, -9.81, 1e5, 0.025, 1e4]`:

- Mesh `examples/assets/duck.usda`: `asset/mesh/bob.obj` (Keenan Crane's Bob,
  CC0 1.0) tetrahedralized by the reference recipe `tetrahedralize(...,
  normalize_input=False, options={"minratio": 1.1})`: trimesh watertight
  assert, STL round trip into pyvista, `make_manifold`, tetgen, the first four
  element columns, the reference's element sign fix and
  `filter_unused_vertices`. Baked once by `tools/bake_diff_pd_duck.py` with
  tetgen 0.8.4, pyvista 0.49.2, trimesh 5.2.0 and pymeshfix 0.18.1, passing
  tetgen 0.5.1's differing keyword defaults explicitly; the reference pins
  tetgen 0.5.1 and pyvista 0.26.1, whose TetGen library differs, so the mesh
  (5,130 vertices, 21,438 tetrahedra, 6,174 surface triangles, 0.6857 m^3) is
  not bit-identical to the reference's. Float64 vertices are stored in
  `newton:referencePoints`.
- Material: density 1000 kg/m^3, $E=4\cdot10^7$ Pa, $\nu=0.45$, volume weight
  $w_v=\lambda=E\nu/((1+\nu)(1-2\nu))=1.241\cdot10^8$ Pa. Uniform particle mass
  $\rho\bar V=0.0320$ kg (164 kg in total for 0.686 m^3, as the reference).
- State forces in reference order: gravity $(0,0,-9.81)$; `arc_contact` with
  center $(-0.6, 0, 7.7)$ m, axis $(0,1,0)$, radius 7.5 m, angle
  $100^\circ-60^\circ$, $p=3$, $k_n=10^5$, $k_f=0.025$, $\mu=10^4$; `planar_contact`
  $z=0$ with $p=3$, $k_n=10^3$, $k_f=0.1$, $\mu=10^4$.
- Initial state: `DuckEnv3d.default_init_position` (rest mesh rotated to
  $70^\circ$ and placed at $\sqrt{R^2-x_{off}^2}$ from the center), every vertex
  at $10\,(-\sin 70^\circ, 0, -\cos 70^\circ)$ m/s; $h=5$ ms, 200 steps; `pd_opt`
  (4000, 10, $10^{-9}$, $10^{-4}$, L-BFGS with 10 pairs). The red point is the
  optimizer's target $(-7.85, 0, 0.315)$ m for the mean vertex position. The
  visual ramp is the reference's `obs.obj` (50 segments, $60^\circ$ to $100^\circ$).

Reference bugs reproduced on purpose (named in the example's comments):

- **B1**: `kn, kf, mu = state_force_parameters[3:]` rebinds the Lame $\mu$, so
  `AddPdEnergy("corotated", [2 * mu])` uses $2\cdot10^4$ Pa, twice the arc
  friction coefficient, instead of $2E/(2(1+\nu))=2.76\cdot10^7$ Pa.
- **B2**: the arc start vector is $(\cos 60^\circ, 0, -\sin 100^\circ)$; after
  normalization the contact window spans $63.08^\circ$ to $103.08^\circ$,
  while the visual ramp spans $60^\circ$ to $100^\circ$.

B3 and B4 are in the optimizer's gradient assembly (for example
`variable_to_env_gradient` scales `grad_mu` by `10 ** x[1]` instead of
`10 ** x[2]`) and do not affect the forward rollout.

The duck slides down the ramp, leaves its raised end and lands on the ground at
the end of the rollout; its final center is $(-7.739, 0, 0.570)$ m, 0.278 m from
the target (loss $0.0386$ m^2). The first step builds the inverse factor of
$A$ on the host (`SparseInverse`, 3.6 million nonzeros), which takes about 56 s
for this mesh; each step then takes about 0.4 s on a GB10 (227 L-BFGS
iterations per step on average).

## Validation

`tests/reference_oracle_diff_pd.py` is a dense NumPy transcription of the C++
step. `tests/test_diff_pd_solver.py` compares both on free fall of a deformed
block, a pinned cantilever, a block on `planar_contact` and on `arc_contact`
(window edge inside the block), external forces, model gravity, and a large
step that wraps the L-BFGS history and halves the line search: iteration
counts are equal and positions agree within $10^{-12}$ m on CPU and CUDA.
`tools/parity/diff_pd_duck.py` runs the duck through the executed C++
reference `pd_eigen` (the DiffPhD build, whose `pd_eigen`, state force and
homogeneous corotated/volume code equals diff_pd_public 1594aa0 here) on the
same mesh and options. With every reference step started from the port's
float32 state, all 200 steps take the same iteration counts (45,461 in total)
and positions agree within $6.0\cdot10^{-13}$ m ($2.4\cdot10^{-13}$ of the
2.55 m mesh diagonal; declared threshold $10^{-10}$). The reference's own
float64 rollout differs from the port's float32-state rollout by at most
0.34 mm over 200 steps; its final center is 0.2776 m from the target
(port 0.2778 m). `tests/test_diff_pd_duck.py` checks the mesh (positive volumes,
closed manifold surface equal to the baked one) and that the first three steps,
before any contact, follow the implicit Euler free flight of the center.
