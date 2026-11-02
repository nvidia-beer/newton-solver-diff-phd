# Codex task: replace the current solver with a faithful DiffPhD implementation

## Explicit user direction

**Replace the existing solver implementation directly. Do not preserve the old algorithm as the default, add an opt-in reference mode, retain a legacy solver, or build a dual implementation.**

The corrected DiffPhD implementation must be the normal implementation used by `SolverDiffPhD` and the examples. Backward compatibility with the current numerical behavior or public API is not a requirement. Change interfaces when necessary for correctness, update all affected callers, and document those changes. Avoid unrelated API churn.

This instruction supersedes the earlier brief's requirement to preserve the existing default and API and add an explicit opt-in algorithm mode. Do not implement that earlier architecture.

## Objective

Work in [newton-solver-diff-phd](https://github.com/nvidia-beer/newton-solver-diff-phd). Inspect [the public DiffPhD reference](https://github.com/Akinesia112/DiffPhD), then replace the current approximations and substituted algorithms with a faithful implementation of the selected reference behavior.

Deliver working solver changes, source-to-code traceability, reference-comparison tooling, tests, and accurate documentation. A documentation-only change or unused helper routine is not completion. Do not declare fidelity without executable evidence.

Read the current `AGENTS.md` and more-specific repository instructions before editing. Keep the implementation self-contained and use Newton's public APIs; do not modify the installed Newton package.

## Scope and source authority

The immediate target remains the pinned public reference implementation. Distinguish this from full fidelity to the paper; do not silently combine the two if their equations or executable behavior differ.

Previously inspected snapshots:

```text
Newton:    4fda7c6c89bb621cc764ce117230ac86ab6c12f3
Reference: d319435373a8a94aac2035bfe41f16f49bc1d2a1
```

Record the actual working revision and all local changes. Do not reset or discard user work. Pin the reference revision used for comparisons. If adopting another revision, explain why and regenerate comparison results.

Read the [DiffPhD paper](https://arxiv.org/abs/2605.14526) and record its exact version. For reference-code parity, inspect executable behavior instead of trusting README claims. For paper-fidelity claims, use the actual paper equations and pseudocode.

Inspect licenses and attribution requirements before copying or translating reference code. This task does not authorize relicensing, importing restricted assets, or redistributing private recordings. A development-only reference checkout is acceptable; it must not become a production dependency.

Focus on tetrahedral mechanics and the contact and derivative features mapped to both implementations. Record unsupported features explicitly. A validated subset is a milestone, not proof of full reference-repository equivalence.

## Architecture: one corrected solver

- Replace the existing solver internals; do not wrap them in a compatibility-mode selector.
- Do not introduce `algorithm="reference"`, `legacy=True`, a second public solver class, or equivalent old/new execution paths.
- Remove superseded routines once the replacement is working and their remaining uses have been migrated.
- Ordinary material models or physically meaningful options may remain where justified. They must not disguise selection between legacy and corrected algorithms.
- Update package exports, examples, CLI arguments, checkpoints, tests, and documentation as necessary for the replacement.
- Document changed semantics and invalidated checkpoint assumptions. Do not silently load incompatible old checkpoints.
- Git history and baseline logs are sufficient historical records; do not ship the previous implementation merely to retain its behavior.
- Implement in coherent increments, with the replaced path becoming the normal path as each increment is validated. Temporary local development comparisons do not justify shipping two solver implementations.

## Phase 1: audit before editing

Read `AGENTS.md`, dependency pins, the solver call graph, tests, and the method/paper/validation documentation. Inspect reference forward and backward dispatch, not just helpers with promising names.

Run the available baseline checks and retain versions, commands, devices, logs, failures, and skips. Historical test counts or timings are not fresh evidence.

Create `docs/reference_compatibility.md` with these columns: source permalink, equation/algorithm step, reference function, current function, discrepancy, replacement, validating test, and status.

Start with the discrepancies documented in the [existing mapping](https://github.com/nvidia-beer/newton-solver-diff-phd/blob/4fda7c6c89bb621cc764ce117230ac86ab6c12f3/docs/papers.md), but reverify them against current code:

| Area | Required investigation |
|---|---|
| Neo-Hookean forward update | Actual singular-value proximal objective, gradient, Hessian, projection, penalty, and convergence |
| Material assembly | Reference weights versus the current preconditioner weights; rest-volume, mass, and unit conventions |
| Heterogeneity | Local versus global material parameters, including any parameter averaging in the reference |
| Global solve | Boundary elimination, factorization, inverse application, caching, and rebuild triggers |
| Acceleration | Anderson variant, bounds, regularization, restarts, and contact handling |
| Stopping | Position/projected-RHS convergence gates, norms, units, and scales |
| Contact | NCP variables, normal/tangential law, frames, compliance, contact-space solver, and geometry refresh |
| Backward pass | Differentiated equation, actual filtering, transpose operator, parameter chains, and snapshot ownership |
| Damping | What the public reference executes versus the paper's stiffness-scaled Rayleigh contribution |

Do not assume all paper components exist in the public reference. In particular, establish whether adaptive proximal-Hessian filtering is actually invoked in its stepping adjoint. Conversely, unused Newton proximal/filtering helpers do not constitute implementation.

Write a concise execution plan, then implement it. Do not stop after the audit unless a genuine blocker prevents further work.

## Phase 2: replace mechanics and forward solve

Derive and implement the selected reference local/global formulation. Match physical assumptions, material weights, constraints, forces, integration, and damping where present in the target.

Verify local proximal stationarity and derivatives with tiny independently assembled cases. Confirm positive-determinant handling, singular-value conventions, and large-deformation behavior. Do not silently substitute a physical-gradient preconditioned solve for a reference proximal update.

Match the reference acceleration and stopping rules. Additional safeguards must be explicitly identified and assessed for changes to the algorithm. Never use an undocumented fallback to the removed solver to make a case pass.

Use sparse production mechanical operators and factor reuse. Small dense test oracles are allowed; dense full mechanical inverses in production are not. An equivalent sparse ordering/factorization is acceptable if it solves the same operator within a justified error budget.

## Phase 3: replace contact and adjoints

Map signs, units, unknowns, frames, gaps, friction, and regularization before implementing contact. Match the selected reference contact formulation; do not silently replace it with a penalty method or smoothed approximation.

Validate normal contact, sticking, and sliding independently. Separate parity using externally fixed contact geometry from parity using independent collision detection and geometry refresh. Passing fixed-contact comparisons does not establish end-to-end detection parity.

Derive the adjoint of the replaced solver's actual converged system. Include transpose/dot-product tests, supported parameter chains, contact linearization, and owned taped snapshots.

Distinguish exact physical derivatives, filtered/reference derivatives, and frozen-branch derivatives. If filtering deliberately changes the derivative, assess agreement with the reference operator separately from finite differences of the physical forward solve. Do not call a biased or filtered result an exact gradient.

Do not require the backward solve to reduce to one inverse application merely because factors are reused. Establish and solve the actual adjoint equation.

Contact creation and branch switching are nonsmooth. Test fixed-branch derivatives separately and report branch-changing finite differences honestly. Do not hide errors by pretending topology derivatives are supported.

If the reference and paper disagree materially, record the discrepancy and follow the reference for the immediate replacement target. Do not automatically add a second algorithm mode to satisfy both.

## Phase 4: build a cross-repository parity harness

Add development-only exporter/comparison tooling using separate environments or subprocesses for the reference's native dependencies and this package.

Use a documented NPZ/JSON fixture schema containing SI units, canonical indexing, rest geometry, masses/density, materials, boundary conditions, initial state, forces, timestep, contacts, options, and objective definition. Every result must include revisions, environment, and device.

Compare positions, velocities, physically comparable energies, force balance, gaps, contact forces, convergence diagnostics, objective values, and supported directional derivatives. Reject incompatible units, topology, or physical definitions before numerical comparison.

Include:

- Homogeneous tetrahedral deformation without contact.
- A pinned heterogeneous mesh with 10-fold and 100-fold stiffness contrasts.
- Large-deformation Neo-Hookean proximal behavior with positive determinant.
- Matched plane support or impact.
- Sticking and sliding friction.
- Short multi-step state, force, and material-gradient comparisons.
- Actuation only where the two sides expose compatible definitions.

Generated mathematical fixtures are permitted as labeled software tests; they are not experimental data. Never generate reference outputs with the replacement solver and label them reference results.

Declare error thresholds before judging final results. Justify absolute and scale-aware relative tolerances using precision, conditioning, solver convergence, and public float32 state rounding. Include tighter-solve/perturbation studies where necessary. Do not loosen thresholds retrospectively to manufacture a pass.

If the reference cannot be run, deliver the adapter and reproducible setup plus independent local checks. Mark cross-repository parity **NOT VERIFIED**, with the precise blocker.

## Phase 5: investigate the recorded-poke limitation

The pinned [validation document](https://github.com/nvidia-beer/newton-solver-diff-phd/blob/4fda7c6c89bb621cc764ce117230ac86ab6c12f3/docs/validation.md#L38-L53) reports X/Y gradient discrepancies of 0.784%/2.017% against a strict 1% threshold. Reverify current status; do not claim those are new measurements.

With authorized prepared data, vary finite-difference perturbation sizes and solve tolerances, record contact/nearest-association changes, separate frozen-association from recomputed evaluations, and isolate the cause. Fix demonstrated defects and add a public minimal regression where possible.

Without authorized data, retain the unresolved limitation and mark that experiment blocked. Do not download private assets, hard-code fitted parameters, weaken the gate, or claim synthetic tests resolve the recorded-data discrepancy.

## Tests and migration

Preserve valid physical invariants and independent numerical oracles, not numerical outputs that merely encode the superseded algorithm. If an old test's assumptions no longer apply, explain why, replace it with an independently justified test, and retain before/after evidence. Do not indiscriminately rewrite expected values from the replacement solver's own outputs.

Update all affected examples to use the corrected solver normally. No opt-in argument should be required to obtain the faithful implementation. Confirm public imports, launchers, optimization paths, and checkpoint validation remain coherent after necessary API changes.

Use the current repository's supported commands. The pinned development instructions specify:

```bash
uv sync --locked --extra examples
uv run --locked ruff check .
uv run --locked ruff format --check .
uv run --locked python -m unittest discover -s tests -v
python3 tools/build_readme.py
python3 tools/build_readme.py --check
uv build
```

If compatible CUDA is available:

```bash
NEWTON_TEST_DEVICE=cuda:0 uv run --locked python -m unittest discover -s tests -v
```

Run the documented parity commands as well. Disclose unavailable prerequisites and environment substitutions. Do not silently change dependency pins to bypass failures.

Measure synchronized live physics separately from compilation, rendering, replay, and movie playback. Do not transfer published speedups to this implementation without matched measurements.

## Blocker policy

Proceed with available CPU and public fixtures. Missing GPU or private data blocks those checks, not all development.

Do not ask whether to preserve the old implementation or add an opt-in mode: the user has explicitly rejected that direction. Necessary API changes and direct algorithm replacement are authorized by this brief.

Stop only the affected workstream for unclear licensing, required private access/secrets, an unavoidable contradiction not resolved by the selected source authority, or repository instructions that cannot be reconciled with this task. Record alternatives and a recommended resolution; continue independent work. Never destroy unrelated user changes.

## Completion criteria

- [ ] One corrected solver is the normal execution path; no legacy/opt-in algorithm architecture ships.
- [ ] Required API and caller migrations are complete and documented.
- [ ] Source-to-function-to-test mappings cover every fidelity claim.
- [ ] Supported forward, contact, and backward comparisons satisfy predetermined tolerances.
- [ ] Physical invariants and independent derivative checks remain valid.
- [ ] Paper discrepancies and unsupported features remain explicit.
- [ ] CPU, GPU, reference, and private-data results are labeled passed, failed, blocked, or not run.
- [ ] No restricted data or unapproved license changes enter the release.
- [ ] Method, paper mapping, validation, usage, and generated README reflect the replaced implementation.
- [ ] No old accuracy/performance result is presented as validation of changed code.

## Final handoff

Provide changed files, a migration note, exact commands, source revisions, numerical comparison tables, saved logs, and remaining blockers. State fidelity only for the scope actually demonstrated. A partial implementation remains partial even if it is now the default.
