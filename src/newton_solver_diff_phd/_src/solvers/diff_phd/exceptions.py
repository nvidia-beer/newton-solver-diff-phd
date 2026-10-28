# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Failures for which a smaller simulation interval may restore convergence."""


class ConvergenceError(RuntimeError):
    """A numerical solve or collision search did not reach its required accuracy."""
