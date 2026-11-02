# Example controls and local data

The standard launcher discovers both DiffPhD and DiffPD examples. After `./docker/build.sh`, run
`./docker/run.sh` to select one, or pass its name directly.

Run the original hexahedral cantilever forward benchmark:

```bash
./docker/run.sh softbody_diff_phd_cantilever
./docker/run.sh --headless softbody_diff_phd_cantilever --test
```

The default follows the released `cantilever_3d.py`: 756 vertices and 500 intact
Hex8 cells, a fixed left end, zero gravity, and 25 steps of 0.01 seconds.
The middle 150 cells have Young's modulus 1e7 Pa; the other 350 use 1e6 Pa.
Poisson ratio is 0.45 and density is 1,000 kg/m³. Eight Gauss samples integrate
each cell. The native mean-cell-volume mass convention, 20 bottom-edge muscle
energies with target length zero, and two tip vertex-plane energies are retained.
At step t=0..24, each of the 36 tip vertices receives +X force 80*t/25 N.
Surface triangles are only for display; no tetrahedral subdivision is used.

For the paper's twist-and-release scenario on the same Hex8 environment:

```bash
./docker/run.sh softbody_diff_phd_cantilever --motion twist
```

This option starts with a twist increasing to 30 degrees at the tip and runs
100 steps at 0.001 seconds, without the axial ramp. The smaller twist step
resolves the oscillation that the source axial timestep strongly damps. The paper describes
twist release but its exact initial state is absent from the released Python
script; 30 degrees is an explicit example setting, not a recovered paper value.
Both modes compute fresh forward physics and need no checkpoint or training.
The viewer's **Motion** selector switches between **Axial stretch** and
**Twist release**, restarting the chosen mode. **Replay** restarts it again.
The twist mode uses relative tolerance 1e-6; axial retains the source's 1e-4.
Interactive twist playback holds the initial pose for 1.5 seconds. Mesh edges
and a live tip-rotation readout make the release visible.
The default video shows the 25 axial steps at 12 fps; playback duration is not
a solver-throughput measurement.

The other retained examples are the DiffPD duck and experimental recorded UR5 scene:

```bash
./docker/run.sh softbody_diff_pd_duck
./docker/run.sh softbody_diff_phd_ur5_poke --forward-only
./docker/run.sh softbody_diff_phd_ur5_poke --reference --forward-only
./docker/run.sh softbody_diff_phd_ur5_poke --forward-only --swept-contact --show-target
```

The bear gait, basic block drop and synthetic swept-poke demos have been removed.
The generic swept-contact extension remains available to the UR5 diagnostic.
Its deep recorded poke still fails; it is not a validated complete demonstration.

The UR5 example requires your authorized, prepared PokeFlex recording. Its
optional local default is `examples/assets/pokeflex_local/recording_fine.json`.
The JSON manifest contains relative `geometry`/`recording` paths and SHA-256
checksums; keep the referenced geometry, recording NPZ and the manifest together. A
prepared self-contained NPZ is also accepted. The loader requires rest vertices,
tetrahedra, boundary triangles, pinned nodes, display vertices/triangles/UVs,
fixed tetrahedral embedding, observation frame IDs, target samples and indices,
and the initial XY placement. Missing data produces an explicit error.

The scene follows the original `dice_material_optimize.py` training settings:
two-way `mesh_contact` state forces, a 0.55 mm contact radius, normal stiffnesses
5e3/5e4 N/m, tangential damping 50 N s/m and friction 0.5. The probe has Young's
modulus 1e13 Pa, Poisson ratio 0.45 and density 800 kg/m³. Only its root cap is
prescribed; the other probe particles are elastic. These are penalty contact
equations and do not guarantee zero penetration. The default prepared die and
probe meshes differ from the original script's generated slab and cylinder.

The blue overlay contains measured cube surfaces; the white surface follows
the simulated tetrahedra through an embedded scan surface. That scanned surface
and the tetrahedral collision boundary are slightly different; the original
asset has up to 0.523 mm scan protrusion at rest. Its preparation now uses a
complete containing-cell fallback instead of assuming the nearest 128 cell
centres include the containing tetrahedron. The robot follows the reference script's hand-authored
trajectory. The local recording supplies no measured probe poses or timestamps,
so the 30 Hz timing is assumed. The default initial placement offsets the white
cube by +30 mm in X and +26 mm in Y from the recording, as an initial guess for
position fitting; `--forward-only` does not fit that offset.

`--forward-only --swept-contact` selects the optional swept collision extension.
It checks vertex/triangle sweeps in both directions and edge/edge sweeps over
each physics step, then applies contact response through the implicit mechanical
matrix. This mode prescribes every probe vertex, defaults to zero friction and
does not use the original thin-shell penalty force. The material equations are
unchanged. Both meshes must start separated. The sweeps interpolate mesh vertices
linearly between physics endpoints; they do not test self-collision within the
die or an exact continuous rigid rotation between endpoints.

The swept mode rejects unresolved collision searches, uncertified contact
response, nonconverged mechanical steps and inverted/degenerate final elements.
It restores the rejected step's input state and raises an error before advancing
the poke clock. A successful collision response therefore must move the free
die while preserving prescribed probe positions; silently stopping the probe is
not used as a substitute. This mode currently supports forward simulation only:
training, saved alignment files, nonzero friction and recorded Warp Tapes are
rejected because the extension has no contact adjoint yet.

An earlier implementation of the prepared deep poke completed 477 of 1,050
substeps, then rejected an inverted element near the fixed clamp. Independent
checks found errors in contact geometry and convergence, which have been
corrected. The earlier fine-mesh rerun accepts 472 substeps and rejects step 473
for an inverted element near the clamp. The latest original-asset run
(`--reference`) accepts 479 substeps and rejects step 480 for an inverted
element at the loaded surface; its contact solve passes. Separate clamp-aligned
and locally refined remeshing experiments also fail, at steps 482 and 424;
neither replaces the default. The failed
command restores its input state and raises an error;
it is not silently accepted. See the [measured validation](validation.md#optional-swept-contact-validation).

Private scans and derived geometry are excluded from release packages and
Docker images; local checkout mounts can access your copy. Use only media
permitted by your dataset access terms. Asset attribution/access notices are in
`src/newton_solver_diff_phd/examples/assets/NOTICE.md`.

Use `--fit-mode position`, `stiffness`, or `compressibility` to fit one parameter
group at a time. `--show-target` overlays the recorded surface. Training saves
`--save-alignment PATH`; `--load-alignment PATH` runs fresh untaped inference.
For live simulation without fitting or a saved checkpoint, use `--forward-only`
with the desired `--initial-xy`, `--young` and `--poisson`. With neither forward
option, the example trains. Forward-only and saved-alignment modes are mutually
exclusive and do not accept training output or gradient-check options.
The prepared recording determines mesh resolution. The default is 30 substeps
per 30 Hz observation with 0.30 retained velocity on every particle after each
substep, including prescribed roots. Independent endpoint positions constrain
the root without replacing the physical velocity used by contact. The forward
iteration budget is 5,000, relative tolerance 1e-4 and Anderson window 5. Swept
contact uses the stricter relative tolerance 1e-7. The optional
`--reference` flag selects a separately prepared mesh and full-scan objective;
it loads `pokeflex_local/recording_reference.json`. The reference preparation
uses the original frame-18 convex hull, 85% decimation, TetGen volume and
24-section cylindrical probe. Both volumes are stored in `die_reference.usdc`;
the companion NPZ contains observations and clamp indices. The offline
`tools/prepare_poke_reference.py` reproduces this preparation from local scans;
it requires PyVista and TetGen in its preparation environment. No runtime
remeshing is performed. Selecting these assets does not establish a successful
full poke or select a different solver. The robot is prescribed visualization
and only the physical probe's root has fixed motion; there is no two-way robot
reaction solve. Alignment files are format version 5, with a physics signature
and a fingerprint of loaded geometry, clamp, motion and observations. Versions
1–4 were fitted with other physics and are rejected; refit them before inference.
The default save path ends in `_v5.json`.
`--contact-tolerance` was removed.

The first live frame builds the sparse solver factors. Preparation has been
accelerated while preserving the factors bit-for-bit, but the larger original
asset still measured about 45 seconds for factor setup alone. The viewer waits during setup; the
terminal now reports its start and completion, then motion progress. The
recording contains 35 frame intervals, or 1.167 seconds of simulated motion.
Afterward the tool holds its final pose while deformable physics continues;
the terminal and GUI explicitly identify that state. Use **Restart poke** in
the example controls to restore the initial state and replay the motion while
reusing the solver factors. Windowed playback remains open after the motion
ends. The viewer's **Pause** control (Space) stops stepping; **Pause Rendering**
freezes only the displayed image.

Training is offline: the viewer pauses for the initial loss/gradient evaluation
and each optimizer update, then replays the stored trajectory. Its FPS includes
those pauses. The terminal and GUI report completed forward/backward times and
rollout counts separately. The default 35 observation intervals at 30 substeps
require 1,050 solver steps per rollout; a line search may repeat the forward
rollout before accepting one update. The first rollout also includes matrix
factorization and graph setup. Inner forward and adjoint iterations use CUDA
conditional graphs when supported; the complete training loop is not captured.
The optional NCP contact correction is skipped for state-only gradients. The
current poke example uses the penalty `mesh_contact` state force instead, whose
reference backward pass holds the force fixed in every fit mode.

`--check-gradients` compares the gradient with central differences that
re-solve contact and recompute nearest associations. The reference gradient
holds each step's contact force fixed. Its heterogeneous mean-weight operator,
Neo-Hookean material formula and original clipping guards are also approximate
derivatives of the executed forward map, so differences need interpretation;
see [validation](validation.md).

## Recording the examples

`python tools/record_examples.py --all` uses `docs/videos.json`. Poke capture uses `--forward-only` and the prepared
local recording; it does not require an old fitted checkpoint. To record a new
fit, supply its compatible version-5 alignment explicitly. Existing alignment
files need refitting. Historical clips use earlier physics and do not validate
the restored solver or certify the material parameters.

Poke previews include the private scanned surface. The [dataset access
terms](https://pokeflex.ait.ethz.ch/) restrict use to noncommercial academic
purposes and prohibit redistribution of the dataset and converted models.
These files are local research previews; the terms do not explicitly grant
redistribution rights for rendered movies. No media upload is performed by the
recording tool.
