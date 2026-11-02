"""Differentiable solvers for Newton."""

from .diff_pd import SolverDiffPD
from .diff_phd import SolverDiffPhD

__all__ = ["SolverDiffPD", "SolverDiffPhD"]
