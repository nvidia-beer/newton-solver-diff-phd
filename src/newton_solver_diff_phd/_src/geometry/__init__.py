# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Static tetrahedral geometry used by the examples."""

from .recording import load_poke_recording
from .usd import Mesh, load_mesh

__all__ = ["Mesh", "load_mesh", "load_poke_recording"]
