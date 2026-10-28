# Solver development

This is an independent differentiable Newton solver package. Keep the implementation in this package; use Newton's public API and preserve the installed Newton package.

Keep the implementation in `src/newton_solver_diff_phd/_src/solvers/diff_phd/`. Export its public API through the family `__init__.py`, `_src/solvers/__init__.py`, and the package root. Use relative imports within the solver family.

Keep each solver self-contained. Put runtime geometry loading under `_src/geometry/`
and any required HHT implementation under `_src/solvers/implicit/`; copy only code
that this solver uses. Store geometry-only example meshes and their reference
embeddings as USD assets under `examples/assets/`, not runtime mesh generators.
Keep materials, quadrature, interpolation and differentiation in solver code.

DiffPhD owns its implicit solve and adjoint; it does not use the generalized HHT implementation.

Examples consume only the solver API exported from the package `__init__.py`; do not import solver `_src` modules or access private solver members. Keep shared rendering helpers inside `examples`.

Document the implemented equations and gradient assumptions in `docs/method.md`. Map paper equations and pseudocode to actual functions in `docs/papers.md`. Keep `docs/usage.md` for example-specific prerequisites. Regenerate the root README with `python3 tools/build_readme.py`; do not edit generated text. Keep one actual demonstration video per discovered example, recorded with `tools/record_examples.py`. Preserve required copyright and license notices.

Use `uv`, Warp kernels, and `unittest`. Keep ordinary modules and direct calls; add abstractions only when an actual solver needs them. Preserve physical accuracy during optimization, and measure live simulation separately from rendering or replay.

Run `uv run --locked ruff check .`, `uv run --locked ruff format --check .`, and `uv run --locked python -m unittest discover -s tests -v`. GPU checks use `NEWTON_TEST_DEVICE=cuda:0`. Inspect the supported derivative branches in tests before adding gradient claims. Keep dependencies and lockfile synchronized.

Example launchers discover runnable `examples/**/example_*.py` files. Give each a
module docstring and a `__main__` guard; use `HEADLESS = True` for terminal-only
examples. Do not add hard-coded example lists to Docker or Python launchers.

Use `nsys/profile_examples.py` for opt-in profiles of README video workloads.
Keep their resolution, numerical budgets and scripted interactions intact;
separate initialization/training, physics and rendering costs. Inspect raw CUDA
and NVTX evidence before making performance claims. Profiling output belongs in
ignored `nsys/results/`, never in the solver runtime or generated example assets.
