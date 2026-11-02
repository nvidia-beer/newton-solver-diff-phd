# Parity against the executed C++ DiffPhD reference

Development harness that runs the same scenes through `SolverDiffPhD` and the
built reference (`pd_eigen_alg_phd`) in separate environments and compares the
results. The current CPU report is `docs/validation/parity-reference-cpu.json`.
`docs/validation/parity-reference-cuda.json` is retained historical evidence for
an earlier source version, not a current GPU validation.

| File | Runs in | Purpose |
| --- | --- | --- |
| `port_cases.py` | port (`uv run --locked`) | Writes one fixture NPZ per case ("Parity fixture schema" in `docs/validation.md`), runs the port, then launches `reference_runner.py` as a subprocess |
| `reference_runner.py` | reference venv | Builds a `TetDeformable` from the fixture, calls `PyForward`/`PyBackward`, writes results |
| `compare.py` | port | Rejects incompatible fixtures, applies the declared thresholds, writes the JSON report |
| `aarch64-build.patch` | reference clone | Every source change needed to build the reference on this host |
| `diff_pd_duck.py` | port, then reference venv | `SolverDiffPD` duck example against the reference `pd_eigen` (see "DiffPD duck" below) |

## 1. Build the reference (Linux aarch64, GB10 sm_121, CUDA 13, gcc 13)

`install.sh` assumes x86_64 and a conda environment. These steps replace it
with the system toolchain (`/usr/bin/gcc`, `/usr/bin/cmake`, `/usr/bin/swig`,
`/usr/local/cuda` nvcc 13.0) and a uv-managed Python 3.12. pbrt is skipped.

```bash
cd ~/Dev/Robotics
git clone DiffPhD DiffPhD-reference-build          # local clone of the user's checkout
cd DiffPhD-reference-build
git checkout d319435373a8a94aac2035bfe41f16f49bc1d2a1
git submodule update --init --depth 1 external/eigen   # baf9a985ece8c02d45ae151d64bdf25d3e2faeaa
git apply ../NewtonSolvers/newton-solver-diff-phd/tools/parity/aarch64-build.patch
mkdir -p external/sse2neon
curl -sL https://raw.githubusercontent.com/DLTcollab/sse2neon/v1.8.0/sse2neon.h -o external/sse2neon/sse2neon.h
# sha256 07723c9f9457dd4316f1fde3dd4eb6f31dd67d9955f6c21f4e609ac1698be48a

# Python with headers for the SWIG module (no system python3-dev on this host).
uv venv --python 3.12 --managed-python .venv
VIRTUAL_ENV=.venv uv pip install numpy scipy
export PATH=$PWD/.venv/bin:/usr/local/cuda/bin:$PATH

# oneTBB, GKlib (build dir is Linux-aarch64), METIS: same steps as install.sh, system gcc.
(cd RealSim/deps/oneTBB && mkdir -p build && cd build \
  && cmake -DCMAKE_INSTALL_PREFIX=../../onetbb_install -DTBB_TEST=OFF -DCMAKE_BUILD_TYPE=Release \
       -DCMAKE_C_COMPILER=/usr/bin/gcc -DCMAKE_CXX_COMPILER=/usr/bin/g++ .. \
  && cmake --build . -j"$(nproc)" && cmake --install .)
(cd RealSim/deps/METIS/GKlib && make config prefix="$(pwd)/../GKlib_build" openmp=set cc=/usr/bin/gcc \
  && cd build/Linux-aarch64 && make -j"$(nproc)" install)
(cd RealSim/deps/METIS && make config prefix="$(dirname "$(pwd)")/METIS_install" \
       gklib_path="$(pwd)/GKlib_build" cc=/usr/bin/gcc && make -j"$(nproc)" || true)
mkdir -p RealSim/deps/METIS_install/lib RealSim/deps/METIS_install/include
cp RealSim/deps/METIS/build/libmetis/libmetis.a RealSim/deps/METIS_install/lib/
cp RealSim/deps/METIS/include/metis.h RealSim/deps/METIS_install/include/

# SWIG bindings before cmake (sources are globbed at configure time).
(cd cpp/core/src && swig -c++ -python py_diff_pd_core.i)
mv -f cpp/core/src/py_diff_pd_core.py python/py_diff_pd/core/
printf "root_path = '%s'\n" "$PWD" > python/py_diff_pd/common/project_path.py
mkdir -p cpp/build && cd cpp/build
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=/usr/bin/gcc -DCMAKE_CXX_COMPILER=/usr/bin/g++ \
  -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++ -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=121           # must print "cuSPARSE found: CUDAToolkit 13.0.88"
make py_diff_pd_core -j"$(nproc)"
cd ../..
LD_LIBRARY_PATH=$PWD/RealSim/deps/onetbb_install/lib:/usr/local/cuda/lib64 PYTHONPATH=$PWD/python \
  .venv/bin/python -c "import py_diff_pd.core.py_diff_pd_core"
```

What the patch changes (nothing in the solver sources):

- `cpp/CMakeLists.txt`: `CMAKE_CUDA_ARCHITECTURES 121`; drop the x86-only
  `-mpreferred-stack-boundary=4`; link flags `-L/lib/aarch64-linux-gnu`
  instead of `-L/lib/x86_64-linux-gnu`; cuSPARSE/cuBLAS/cudart from the
  system CUDA toolkit (`CUDA::` targets) instead of hard-coded conda CUDA 12
  `.so` paths, so `CUDA_AVAILABLE` (the GPU path) stays enabled; Python
  include dir from `sysconfig` instead of `python3-config`; `-mavx2 -mfma` for
  the MAS sources only on x86_64 (those sources already have a non-AVX path).
- MAS preconditioner (`SeVectorSimd.h`, `SeSchwarzPreconditioner.cpp`):
  include `sse2neon.h` instead of `<xmmintrin.h>`/`<immintrin.h>` on aarch64.

## 2. Run the cases

```bash
cd ~/Dev/Robotics/NewtonSolvers/newton-solver-diff-phd
OUT=/tmp/diff_phd_parity
uv run --locked python tools/parity/port_cases.py --out $OUT --device cuda:0 \
  --reference-root ../../DiffPhD-reference-build
uv run --locked python tools/parity/compare.py --results $OUT \
  --reference-root ../../DiffPhD-reference-build --report docs/validation/parity-reference-cuda.json
```

`--cases NAME ...` selects cases; unknown names are rejected.
Each reference run writes `$OUT/<case>.log` (its stdout and stderr), which records
the `[AlgPhd-GPU-VERIFY]` lines when that path is available. `compare.py` rejects
a different reference revision and returns a failing exit status for failed
checks, rejected fixtures or a run with no compared cases. It records which
heterogeneous gradient comparisons were reported without being judged.

For CPU validation use `--device cpu` and write the report
to `docs/validation/parity-reference-cpu.json`. The existing C++ build also runs
without CUDA by falling back to CPU after cuSPARSE initialization fails. This
does not validate its GPU path. The 2026-11-02 rerun used `uv run --no-sync`
with the already installed pinned Newton/Warp versions because dependency
fetching was blocked by DNS; see [validation](../../docs/validation.md#reproduce).

## DiffPD duck

```bash
uv run --locked python tools/parity/diff_pd_duck.py --out /tmp/diff_pd_duck \
  --reference-root ../../DiffPhD-reference-build --device cuda:0
```

Runs the 200-step `softbody_diff_pd_duck` rollout, then the same mesh and
parameters through the reference `pd_eigen` (`use_bfgs = 1`), stepwise from the
port's states and as the reference's own float64 rollout. The build's
`pd_eigen` differs from diff_pd_public 1594aa0 only in heterogeneous-material
plumbing, verbosity levels (`PD iteration` lines need `verbose >= 2`), the
order of the right-hand-side scatter (rounding) and `max_contact_iter`, which
applies only to frictional boundaries; its state forces and PD energies are
identical. Writes `report.json` to `--out` (about 9 min on a GB10).

## Cases

| Case | Scene |
| --- | --- |
| `homogeneous_free` | Free 4x2x2 grid, stretched initial state, one step |
| `pinned_contrast_{10,100}x_{per_element,grouped}` | Left-pinned grid with a stiff half, both reference material encodings |
| `neo_hookean_large_deformation` | Free grid, Neo-Hookean, large stretch |
| `plane_impact`, `sticking_friction`, `sliding_friction` | Planar frictional boundary, three steps |
| `multistep_pinned` | Left-pinned grid, four steps |
| `muscle_activation` | Left-pinned grid with per-tet fibres and activations, two steps |
| `mesh_contact_poke` | Bottom-pinned 4x2x2 grid (A) and a free one-cell grid (B) lowered into A's top, coupled by the reference `mesh_contact` state force (`AddStateForce("mesh_contact", ...)` vs `SolverDiffPhD(mesh_contact=...)`), three steps. Vertex-face pairs exist in both directions every step, and Coulomb friction acts with the `kf` cap active on some pairs. Both sides give the force zero backward, so gradients are judged in the exact regime. The report also lists per-step pair counts (port) and the forces' difference (reference `PyForwardStateForce` minus gravity) |
| `mesh_contact_prescribed` | Three Neo-Hookean steps with prescribed root endpoints independent of incoming physical velocities, two-way mesh contact force, and 0.30 damping of all output velocities. Exercises the original training demo's boundary and velocity semantics. |
| `mesh_boundary_poke` | One step of a free block above a pinned grid, using `MeshFrictionalBoundary` on both sides. Each solver evaluates its own mesh geometry; states, active sets, multipliers and backward outputs are compared. |



- Fixtures are rounded to float32 (Newton stores states, rest positions and
  materials in float32), so both sides start from identical values. Rest grids
  use 0.125 m cells at dyadic coordinates: the float32 rest-pose inverses of
  Newton are then exact. Non-dyadic rest geometry gives the port a
  ~1e-8 relative perturbation of `A` and `G` (float32 `tet_poses`).
- Port positions are the float64 iterate `solver._x`; port velocities are
  `(x - q) / dt`. Multi-step port rollouts store states in float32 between
  steps; the reference keeps float64 (its forward tape requires exact
  sequential states).
- Reference multipliers come from its `ALG_PHD_DEBUG_CONTACT_VERBOSE` trace
  (6 significant digits); it exposes them nowhere else.
- Heterogeneous PD materials are encoded on the reference side either as
  per-element stiffness arrays (`per_element`, where the reference backward
  uses the mean weight of each energy) or as one energy pair per material
  group (`grouped`, scalar weight per group).
- `dL/du = -dL/da` for the reference target fibre length `a = 1 - u`.
