# Reference DiffPhD projective dynamics

`SolverDiffPhD` runs the executed algorithm of the public DiffPhD reference
([Akinesia112/DiffPhD at d319435](https://github.com/Akinesia112/DiffPhD/tree/d319435373a8a94aac2035bfe41f16f49bc1d2a1),
`ForwardProjectiveDynamicsAlgPhd` and `BackwardProjectiveDynamicsAlgPhd`,
with the backward deviations listed below);
it has no legacy or opt-in mode. `SolverDiffPD`, a separate solver family in the
same package, ports the DiffPD forward step; see [DiffPD method](method_diff_pd.md).
[Reference compatibility](reference_compatibility.md) maps every step to its
source line, implementation and test, including each deviation.

## Conventions copied from the reference

- Tet4 elements, or explicit regular Hex8 cells as described below, in SI units.
  Newton float32 states; all solver arithmetic is in float64.
- Every dynamic particle has the same mass $m=\rho\bar V$, where $\bar V$ is the
  mean rest tetrahedron volume. Every element uses $\bar V$ as its volume.
  `density` is a required argument. The model's particle masses only mark
  zero-inverse-mass particles, which are Dirichlet values.
- `material="pd"`: corotated energy with weight $w=2\mu_e\bar V$, plus
  volume preservation with weight $\lambda_e\bar V$.
- `material="neo_hookean"`: the reference proximal Neo-Hookean projection with
  weight $2\mu_e\bar V$. The projection uses the mean $\bar\mu$, $\bar\lambda$
  and $k=2\bar\mu$ of all elements, so the fixed point is the reference's
  Moreau-envelope energy, not the hyperelastic residual.
- Optional fixed fibres: weight $k_m\bar V$, target length $a=1-u$ for the
  contraction `tet_activations` $u$.
- No damping. Projections use Eigen's unsigned SVD convention ($\sigma\ge0$,
  sorted; $UV^T$ is a reflection when $\det F<0$).

## Forward step

With $\tilde q=q+hv+\tfrac{h^2}{m}(f_{ext}+mg)$, the per-coordinate matrix
$A=\tfrac{m}{h^2}I+\sum_e w_eG_e^TG_e$ (Dirichlet rows eliminated) and element
projections $P_e$, each iteration computes

$$
x\leftarrow A^{-1}\Bigl(\tfrac{m}{h^2}\tilde q+\sum_e w_eG_e^T\bigl(P_e(G_ex)-G_ex_D\bigr)\Bigr),
$$

starting from $x=q$, followed by type-II Anderson mixing. Anderson uses window 5,
regularization $\max(10^{-6}\lVert dG\rVert^2/m,10^{-12})$, and resets when
$\lVert\gamma\rVert\ge10$. Iteration stops after the first iteration when both
$\lVert\Delta x\rVert\le r\lVert x_{prev}\rVert+a$ and
$\lVert\Delta p\rVert\le r\lVert p_{prev}\rVert+a$ hold, where $p$ is the
projection right-hand side. Defaults are $a=10^{-9}$, $r=10^{-6}$ and 10000
iterations. The output is $q^+=x$, $v^+=(x-q)/h$.

`step(..., dirichlet_positions=targets)` optionally supplies a full-particle
array of prescribed endpoint positions; only zero-inverse-mass particles use
it. Incoming positions and velocities remain the physical start state. This
keeps a moving root's contact velocity independent of its next prescribed pose,
as in the reference's separate Dirichlet boundary updates.

$A^{-1}=S^TS$ is applied through exact sparse inverse factors. They are built
on the host (nested-dissection LDLᵀ in [sparse.py](../src/newton_solver_diff_phd/_src/solvers/diff_phd/sparse.py))
whenever dt changes or after `notify_model_changed`. Changing
`model.tet_materials` therefore needs `notify_model_changed(MODEL_PROPERTIES)`.
Host setup uses symbolic sparse fill and vectorized numeric updates. Inverse
rows use one linear-size work array; no dense matrix inverse is constructed.
The update order and exact-zero omission are preserved. The original-scene
comparison checks every entry of both CSR factors against the previous setup.

## Hex8 forward branch

`SolverDiffPhD(model, hex_mesh=mesh, hex_materials=materials, density=...)`
uses Newton particle states with the explicit eight-node cells in `mesh`.
It creates no volume tetrahedra. The implemented reference Hex geometry is a
grid of equally sized, axis-aligned rest cubes, in binary x/y/z corner order.
Other rest shapes are rejected. Materials are float64 `[mu, lambda]` per cell.

`hex.py` uses trilinear shape gradients at eight tensor-product Gauss points.
Each sample contributes weights `2*mu*mean_volume/8` and
`lambda*mean_volume/8`; each particle still has native mass
`density*mean_volume`. Matrix assembly and local projections sum all eight
samples. The existing sparse inverse and Anderson iteration run on these
contributions, including conditional CUDA graphs.

Hex fibres use explicit native target lengths in `solver.muscle_targets`.
Target zero contracts the fibre; it does not disable the energy. Their sample
weight is `stiffness*mean_volume/8`. Optional `planar_energies` reproduce the
reference vertex PD energy: stiffness adds an isotropic diagonal, while the
right-hand side uses the projection onto the permitted half-space.
These are distinct from the frictional boundary solver.

`deformation_determinants(state)` reports det(F) at every cell/sample. The
cantilever audits these values at every step. Hex8 currently rejects Warp Tape
recording and frictional/mesh contact: its supported scope is forward
elastic, muscle and vertex-plane energies. Existing Tet4 derivatives retain
their documented scope below. No Hex8 adjoint is claimed.

## Contact

One planar ($(n\cdot x+d)/\lVert n\rVert$) or spherical ($\lVert x-c\rVert-r$)
frictional boundary acts on registered vertices, by default the boundary-face
vertices. Every registered vertex is a candidate on every step. Frames and the
Delassus matrix $W_{ij}=(n_i\cdot n_j)A^{-1}_{ij}$ are frozen at the step
start. Each iteration takes one semismooth Fischer–Burmeister Newton step with
$r_c=\max(h^2W_{cc},10^{-10})$:

$$
(\Omega W\Omega+E)\,d\lambda=-\phi+\Omega n\cdot x-\Omega n\cdot A^{-1}g,\qquad
\lambda\in[0,10^8].
$$

Coulomb friction is an isotropic stick/slip disk with $W_{tt}=A^{-1}_{cc}$. The
applied force $\omega\lambda n$ plus the friction impulse enters the next solve.
Multipliers warm-start the next step, and restitution is applied after the loop.
`boundary_parameters` is a device array: moving obstacles update it between
steps, also between replays of a captured graph.

The `"mesh"` boundary is the reference `MeshFrictionalBoundary` between two
bodies of one model, `("mesh", (contact_radius, split_index, faces_left,
faces_right))`. A registered vertex below `split_index` is tested against
`faces_right`, every other one against `faces_left`; the force acts on the
registered vertex only, so the opposite body is an obstacle. At the step start
only registered vertices whose distance to the opposite surface is at most
$6r$ ($r$ = `contact_radius`) are candidates. Their normal is fixed to
$(p-c)/\lVert p-c\rVert$ for the closest point $c$, with Gram-Schmidt tangents.
Every iteration measures the gap $(p-c)\cdot n-r$ against the opposite surface
at the current iterate, so a prescribed or deforming surface is tracked within
the step. Restitution acts within $0.1r$. Filtered vertices stay in the
fixed-size arrays with $\omega=\lambda=0$, which is exactly the reference's
shorter candidate list, and their warm start restarts at zero as in the
reference.

Two bodies of one model can also interact through the reference
`mesh_contact` state force (`mesh_contact=(contact_radius, kn_left, kn_right,
kf, mu, faces_a, faces_b)`). Every vertex of one body's surface is tested
against every face of the other. Within `contact_radius` of the closest point,
the vertex gets the penalty force $k_n(r-d)\,n$ plus the friction
$-\min(k_f,\mu f_n/(\lVert v_t\rVert+\epsilon))\,v_t$ of its velocity relative
to the face, and the face's vertices get the opposite force by barycentric
weight. As in the reference, the force is evaluated once per step on the
start-of-step state, enters $\tilde q$ like an external force, and has no
gradient.

Self-contact, damping and Newton's native `Contacts` are unsupported.

## Graph capture and status

The whole iteration, including Anderson, the convergence reductions and the
contact solve, runs on the device as one `wp.capture_while` loop. Inside a
caller's `wp.ScopedCapture` it becomes part of that graph. Otherwise it replays
a cached graph (CUDA) or a host loop (CPU). Factor rebuilds cannot be captured.

Failures set device status bits instead of raising: 1 = `max_iterations`
reached (the last iterate is kept, as in the reference), 2 = singular volume
projection, 4 = invalid activation, 8 = adjoint fixed point not converged,
16 = a reference backward safeguard modified or suppressed a contribution.
`read_status()` returns and clears them; `check_status()` raises
`SolverDiffPhD.ConvergenceError` for bits 1, 8 and 16 and `ValueError` for bits 2 and 4.
The poke example warns on 1, 8 and 16 and retains the reference outputs. Element contributions are gathered per vertex
in a fixed order, so repeated runs are bitwise identical.

## Backward step

Steps run under a `wp.Tape` record the reference backward pass. With
$g=\partial L/\partial q^++\partial L/\partial v^+/h$ it solves the backbone
$(A-K)\mu=g$ by the fixed point $\mu\leftarrow A^{-1}(g+K\mu)$ with Anderson
mixing, where $K=\sum_eG_e^T\,\bar w\,\partial P_e/\partial F\,G_e$ at $x$.
The weight is the global mean within each energy family, following the
reference's per-element material encoding; the forward matrix still uses
per-element weights. Then:

- State gradients come from $\mu$ and **ignore contact**, as in the reference:
  $\partial L/\partial q\mathrel{+}=\tfrac{m}{h^2}\mu-\partial L/\partial v^+/h$,
  $\partial L/\partial v\mathrel{+}=h\tfrac{m}{h^2}\mu$,
  $\partial L/\partial f_{ext}\mathrel{+}=\mu$.
  For homogeneous PD materials at an unclipped, converged fixed point, this
  equals the derivative of a rollout whose contact forces are frozen.
- Active contacts ($\lambda>0$, $\omega\ge0.5$, plus tangents when friction is
  on) enter material and activation gradients through the Woodbury projection
  $z=\mu-W_TG^{-1}J\mu$.
- Material gradients are $\partial L/\partial w_e=-z^T\bar VG_e^T(F-P(F))$ for
  the PD energies. For Neo-Hookean they use the hyperelastic stress at $F$
  divided by $2\bar\mu$, which is the reference formula but not the derivative
  of the proximal forward step.
- Activation gradients are with respect to $u$.
- Dirichlet values receive $\partial L/\partial x_D=g_D-(A-K)_{DF}\mu_F$. The
  reference treats them as constants; this chain is added so that clamped
  particles moved by a parameter, such as the die placement, are differentiable.
  With explicit endpoint targets this contribution goes to the target array;
  copying clamped die positions into that array preserves their placement chain.

The Woodbury projection is evaluated only when material or activation gradients
are requested. State-only differentiation, including poke position fitting,
uses $\mu$ directly and skips the contact response column solves. This changes
no derivative formula. Contact snapshots remain available when gradients are
enabled after the forward recording; material and activation fitting retain
the full projection.

The reference's nonfinite-value scrubbing, norm clipping at $10^6$, divergence
suppression and inertia-gradient cascade guard are reproduced. They can change
or suppress a derivative and are reported through solver status. On
heterogeneous meshes, the mean-weight $K$ differs from the derivative of the
per-element forward map and can make the adjoint iteration indefinite. Exact
finite-difference claims are therefore restricted to homogeneous, unclipped,
converged PD branches away from degenerate singular values. The Neo-Hookean
material formula and contact derivatives retain the reference limitations.
Contact geometry, candidate sets, density and dt are constants.
At zero mean PD volume weight, the reference's stress/weight expression is
nonfinite and suppresses the aggregate material gradient. This port reproduces
that behavior and reports status 16. It is not the finite-difference material
derivative of the zero-weight forward model.

## Example scenes

The UR5 example fits die placement or one material parameter to authorized
local PokeFlex scans with a nearest-vertex Chamfer loss. Its physical settings
follow `dice_material_optimize.py`: the bidirectional `mesh_contact` state force
with radius 0.55 mm, normal stiffnesses 5e3/5e4 N/m, tangential damping
50 N s/m and friction 0.5. The die defaults to E=500 kPa and Poisson ratio 0.40;
the probe uses E=1e13 Pa and Poisson ratio 0.45. Both use density 800 kg/m³ and
zero gravity. Only the probe root cap is prescribed; its other particles are
free. Independent endpoint targets leave incoming physical velocities intact
for contact. The recorded trajectory uses 30 substeps per 30 Hz observation
and retains 0.30 of every particle's velocity per substep, including prescribed
roots. This example-level damping is outside the solver. The iteration budget
is 5,000, relative tolerance 1e-4 and Anderson window 5.

The prepared meshes and scanned-surface embedding differ from the original
script's generated geometry. Material fitting updates only die tetrahedra;
the stiff probe remains included in the reference projection's mean parameters.
Matching these equations does not establish full recorded-scene parity.

The unverified signed-SVD and block-pivot contact experiments have been removed.
The default solver and examples use the reference projection and contact step.
Neither the reference algorithm nor passing small parity fixtures guarantees
that a deep poke avoids penetration, inversion or iteration caps. A fresh
recorded rollout is required to establish those properties; see
[validation](validation.md). The default path enables no injectivity guard,
alternate contact QP, material model or clamp remeshing. Poke alignment versions 1–4 are incompatible
and must be refitted (current format: version 5); checkpoints record both a
physics signature and the loaded scene fingerprint.
