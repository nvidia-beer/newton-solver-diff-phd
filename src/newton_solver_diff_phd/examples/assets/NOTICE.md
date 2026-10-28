# Example assets and data

`bear.usda` contains 276 nodes, 768 tetrahedra, material regions and eight muscle
groups. Its detailed source volume is retained for total-mass normalization.
`bear_controller.json` stores trained weights and losses, not saved motion.
`poke_probe.usda` is a twelve-sector, 10 mm radius, 100 mm long probe with 26
vertices and 36 tetrahedra. These assets retain the Newton Developers'
Apache-2.0 attribution.

The UR5e model/meshes are copyright 2018 ROS Industrial Consortium, BSD-3-Clause;
see `universal_robots_ur5e/LICENSE` and the package third-party notices.

## Local PokeFlex recordings

`pokeflex_local/` contains authorized local recordings when installed by the
user. The prepared scene uses frames 18–53, a 13,781-vertex display surface,
affine embedding, and a 109-node/362-tetrahedron simulation mesh. Geometry is
in `die.usdc`, measured samples in `recording_debug.npz`, and their checksums in
`recording_debug.json`. Dataset preparation and access notices remain in the
private `provenance.json`.

PokeFlex: Jan Obrist, Miguel Zamora, Hehui Zheng, Ronan Hinchet, Firat Ozdemir,
Juan Zarate, Robert K. Katzschmann, and Stelian Coros.
[Dataset](https://pokeflex-dataset.github.io/) ·
[Access terms](https://pokeflex.ait.ethz.ch/).

**Recorded data and converted geometry must not be redistributed.** The private
subtree is ignored by Git and excluded from wheels, source archives and Docker
build contexts. Supply your authorized prepared recording with `--recording PATH`.
See [usage](../../../../docs/usage.md) for commands and required fields.
