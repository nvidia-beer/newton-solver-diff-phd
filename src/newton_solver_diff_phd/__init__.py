"""Differentiable tetrahedral mechanics for stock Newton."""

from ._src.coupling import UR5Kinematics
from ._src.geometry import Mesh, load_mesh, load_poke_recording
from ._src.solvers import (
    BearProblem,
    PokeAlignmentProblem,
    SolverDiffPhD,
    command_poke_velocities,
    damp_poke_velocities,
)

__all__ = [
    "BearProblem",
    "Mesh",
    "SolverDiffPhD",
    "load_mesh",
    "load_poke_recording",
    "PokeAlignmentProblem",
    "UR5Kinematics",
    "command_poke_velocities",
    "damp_poke_velocities",
]
