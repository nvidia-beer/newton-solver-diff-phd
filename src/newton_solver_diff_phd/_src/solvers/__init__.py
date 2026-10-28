"""Differentiable solvers for Newton."""

from .diff_phd import BearProblem, PokeAlignmentProblem, SolverDiffPhD, command_poke_velocities, damp_poke_velocities

__all__ = ["BearProblem", "SolverDiffPhD", "PokeAlignmentProblem", "command_poke_velocities", "damp_poke_velocities"]
