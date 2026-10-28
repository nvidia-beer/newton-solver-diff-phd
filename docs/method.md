# Differentiable heterogeneous tetrahedral mechanics

`SolverDiffPhD` uses Tet4 rest gradients, lumped mass and SI units. State arrays
are Newton float32 particles; constitutive evaluation, sparse solves and
adjoints use float64. Zero inverse mass pins a node, optionally with prescribed
velocity. Native rigid dynamics, springs and triangle elasticity are excluded.

The [paper-to-code mapping](papers.md) identifies the implemented portions of
DiffPhD, Fast But Accurate and DiffPD, including their algorithm steps and
explicit differences from the published methods.

## Energy and discrete residual

For $F_e=D_sD_m^{-1}$ and $J_e=\det F_e>0$, Neo-Hookean mode uses

$$
E(q)=\sum_e V_e\psi_e(F_e),\qquad
\psi_e(F)=\frac{\mu_e}{2}\bigl(\mathrm{tr}(F^TF)-3\bigr)
-\mu_e\log J+\frac{\lambda_e}{2}(\log J)^2.
$$

Projective mode instead uses weighted distance-to-rotation and volume projection
energies. Optional fixed fibre directions add activation-controlled energies.
The material mode changes the physical energy, not only the preconditioner.
For timestep $h$, mass damping $c$, forces $f$, and contact Jacobian $G$, the
backward-Euler mechanics residual is

$$
r_q=\frac{M}{h^2}(q^+-q-hv)+\frac{cM}{h}(q^+-q)
+\nabla E(q^+)-f-Mg-G^Tr=0,\qquad v^+=(q^+-q)/h.
$$

Position-level contact enforces compliant Signorini complementarity
$0\le r_n\perp(g_n+\varepsilon r_n)\ge0$ and a Coulomb friction cone
$\|r_t\|\le\mu_f r_n$, with sticking or sliding selected by the contact solve.
Geometry comes from a configured plane or swept vertex–face/edge–edge detection.
Nonadjacent features can represent self/contact between soft bodies. Newton's
native `Contacts` are not consumed; rendered plane shapes do not configure the
physical plane automatically.

A constant projective stiffness based on $2\mu_e+\lambda_e$, inertia and damping
preconditions the iterations. Sparse inverse factors are prepared with NumPy
and applied in Warp; material/mass/timestep changes rebuild them. This matrix
is not substituted for the physical residual Jacobian in the adjoint. Default
scaled mechanics/contact tolerances are $10^{-7}$/$10^{-9}$ m; examples may
choose explicit tolerances. Nonconvergence raises `ConvergenceError`.

## Implicit derivatives

For converged coupled state $z=(q^+,r)$ and input $\theta$, write
$R(z,\theta)=0$. Reverse differentiation solves

$$
R_z^Ta=\nabla_z L,\qquad
\nabla_\theta L=\partial_\theta L-R_\theta^Ta.
$$

Position, velocity, external forces, Lamé coefficients and fixed-fibre activation
have VJPs. Contact feature IDs, barycentric coordinates, normals and stick/slip
branches are frozen. Rest geometry, mass, timestep, damping, friction and fibre
geometry are fixed. There is no derivative through contact creation or topology.
Owned primal/material/contact snapshots prevent later steps overwriting taped
inputs. True linear residuals are checked. Taped output states must be distinct.

The bear has 276 nodes/768 tetrahedra, eight fibre groups and sixteen sine/cosine
weights: $u_g(t)=0.3\tanh(a_g\sin(4\pi t)+b_g\cos(4\pi t))$.
It optimizes forward mass-centre velocity with lateral/vertical and controller
penalties over 100 steps of 0.008 s. The bundled controller runs fresh inference.
The UR5 example fits placement or one material parameter against authorized
local PokeFlex scans using a nearest-vertex Chamfer loss. Nearest associations
are frozen per backward pass. The robot is prescribed kinematics; the probe
provides no two-way rigid coupling.

Whole-step CUDA graph capture is unsupported because outer convergence and
geometry refresh use host decisions; inner conditional graphs are used. Warp
Tape/runtime hooks require the pinned Warp version. See [validation](validation.md)
for numerical checks and the recorded-poke gradient limitation, and
[usage](usage.md) for the recording prerequisite and example commands.
