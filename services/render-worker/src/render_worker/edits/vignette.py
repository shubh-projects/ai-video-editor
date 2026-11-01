# Copyright (c) 2025 Devayan Dewri. All rights reserved.
# Licensed under the Elastic License 2.0 - see LICENSE in the repo root.
"""Vignette filter primitive for darkening the frame edges."""

from __future__ import annotations

import math


def vignette_filter(
    rel_start: float,
    rel_end: float,
    intensity: float = 0.4,
) -> str:
    """Return an FFmpeg vignette filter enabled over the effect window.

    FFmpeg's ``angle`` is a lens angle in radians clipped to [0, PI/2], and a
    LARGER angle means a STRONGER vignette (PI/2 -> pure black edges). The old
    ``PI/(1 - intensity)`` was always >= PI/2, so every intensity rendered
    pure-black edges. Here ``intensity`` in [0, 1] maps linearly to
    [0, PI/2]; the default 0.4 equals FFmpeg's own default look (~PI/5).
    """
    intensity = max(0.0, min(1.0, float(intensity)))
    angle = intensity * math.pi / 2
    return (
        f"vignette={angle:.4f}:eval=init:"
        f"enable='between(t\\,{rel_start:.3f}\\,{rel_end:.3f})'"
    )
