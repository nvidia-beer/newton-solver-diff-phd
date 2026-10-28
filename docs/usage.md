# Example controls and local data

The standard launcher discovers two examples. After `./docker/build.sh`, run
`./docker/run.sh` to select one, or pass its name directly.

```bash
./docker/run.sh diffsim_diff_phd_bear
./docker/run.sh diffsim_diff_phd_bear --trained
./docker/run.sh softbody_diff_phd_ur5_poke --recording /workspace/recording_debug.json
```

Bear training optimizes sixteen periodic muscle weights and displays the current
trajectory. `--trained` advances fresh physics with the bundled controller.
Use `--save-controller PATH` during training and `--load-controller PATH` for
later inference. Displayed training trajectory replay is not a speed benchmark.

The UR5 example requires your authorized, prepared PokeFlex recording. Its
optional local default is `examples/assets/pokeflex_local/recording_debug.json`.
The JSON manifest contains relative `geometry`/`recording` paths and SHA-256
checksums; keep `die.usdc`, `recording_debug.npz` and the manifest together. A
prepared self-contained NPZ is also accepted. The loader requires rest vertices,
tetrahedra, boundary triangles, pinned nodes, display vertices/triangles/UVs,
fixed tetrahedral embedding, observation frame IDs, target samples and indices,
and the initial XY placement. Missing data produces an explicit error.

Private scans and derived geometry are excluded from release packages and
Docker images; local checkout mounts can access your copy. Use only media
permitted by your dataset access terms. Asset attribution/access notices are in
`src/newton_solver_diff_phd/examples/assets/NOTICE.md`.

Use `--fit-mode position`, `stiffness`, or `compressibility` to fit one parameter
group at a time. `--show-target` overlays the recorded surface. Training saves
`--save-alignment PATH`; `--load-alignment PATH` runs fresh untaped inference.
The default prepared scene uses 109 die nodes/362 tets and one step per 30 Hz
observation. The optional `--reference` mode expects a separately prepared,
higher-resolution recording; it is not included, and uses 30 substeps with
0.30 retained velocity by default. The robot is prescribed visualization and
the physical probe has fixed motion; there is no two-way robot reaction solve.

The retained full-recording gradient check misses its strict 1% gate in one
coordinate. See [validation](validation.md) before using those fitted values
as an accuracy result.

## Recording the examples

`python tools/record_examples.py --all` uses `docs/videos.json`. Bear capture
uses the bundled controller. Poke capture additionally requires
`outputs/diff_phd_poke/alignment_fast.json`, created by running its training mode
against the same prepared recording. The supplied local alignment fits XY
placement; it does not certify the Young modulus or Poisson ratio.

Poke previews include the private scanned surface. The [dataset access
terms](https://pokeflex.ait.ethz.ch/) restrict use to noncommercial academic
purposes and prohibit redistribution of the dataset and converted models.
These files are local research previews; the terms do not explicitly grant
redistribution rights for rendered movies. No media upload is performed by the
recording tool.
