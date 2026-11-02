# Third-party notices

Adapted solver and example code and derived probe geometry include work
copyright 2026 The Newton Developers, licensed under Apache License 2.0.
Original copyright/SPDX headers are retained; see LICENSE.

The DiffPhD solver in `src/newton_solver_diff_phd/_src/solvers/diff_phd/`
(`solver.py`, `kernels.py`, `projections.py`, `anderson.py`, `contact.py`,
`mesh_contact.py`, `adjoint.py`) contains portions translated from DiffPhD
<https://github.com/Akinesia112/DiffPhD>, commit
d319435373a8a94aac2035bfe41f16f49bc1d2a1, distributed under this license:

The Bunny diagnostic in `examples/_diff_phd_bunny.py`, its offline preparation tool
`tools/prepare_bunny_reference.py` and native validation runner
`tools/parity/bunny_reference_runner.py` also adapt that reference code. The
Stanford-derived Bunny geometry has separate terms recorded in
`examples/assets/NOTICE.md` and is prepared locally rather than packaged.

The cantilever scene in `examples/_diff_phd_cantilever.py`, its grid recipe in
`tools/bake_cantilever.py`, and native comparison in
`tools/parity/cantilever_reference_runner.py` adapt the same reference's
`CantileverEnv3d` dimensions, stiffness regions and solver setup. The bundled
`cantilever.usda` reproduces its original Hex8 grid. Geometry and setup retain
the source's MIT attribution; the USD representation and optional explicitly
specified twist-release mode retain this package's Apache-2.0 attribution.
Its provenance is in `examples/assets/NOTICE.md`.

```text
MIT License

Copyright (c) 2021 MIT Graphics Group

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

The DiffPD solver in `src/newton_solver_diff_phd/_src/solvers/diff_pd/`
(`solver.py`, `kernels.py`, `projections.py`), the test oracle
`tests/reference_oracle_diff_pd.py`, the duck example scene
`examples/example_softbody_diff_pd_duck.py` and its mesh `examples/assets/duck.usda`
(tetrahedralized from the reference's `asset/mesh/bob.obj`) contain portions translated from DiffPD
<https://github.com/mit-gfx/diff_pd_public>, commit
1594aa07a6983ff9a9c107e9f0d83d07fa99c8d1, distributed under this license:

```text
MIT License

Copyright (c) 2021 MIT Graphics Group

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

The reference's `bob.obj` is a MeshLab-processed copy of the "Bob" model
from Keenan Crane's 3D model repository
<https://www.cs.cmu.edu/~kmcrane/Projects/ModelRepository/>, which dedicates
the model to the public domain under CC0 1.0 Universal
<https://creativecommons.org/publicdomain/zero/1.0/>. The reference repository
does not state the origin; the name and the duck shape identify it.

`unsigned_svd` in both `projections.py` files and the dense LDLT solves in `dense.py` and
`anderson.py` are original Warp implementations. They reproduce the numerical
conventions of the reference's linear algebra (Eigen's two-sided Jacobi SVD and
pivoted LDLT, used only as an algorithmic reference) so results match the
reference; no Eigen source is included.

The bundled Universal Robots UR5e model and meshes are copyright 2018 ROS
Industrial Consortium, licensed under BSD 3-Clause. The complete notice and
license are in `src/newton_solver_diff_phd/examples/assets/universal_robots_ur5e/LICENSE`
and wheel license metadata. The model is distributed by MuJoCo Menagerie:
<https://github.com/google-deepmind/mujoco_menagerie/tree/main/universal_robots_ur5e>.

Private PokeFlex recordings and scan-derived geometry are excluded from source
archives, wheels, and Docker build contexts. Their access terms prohibit
redistribution, including converted data. These files are not covered by the
package's Apache license. Dataset attribution and local usage requirements are
in `src/newton_solver_diff_phd/examples/assets/NOTICE.md`.
