"""Step 6b – local contrast inside hot objects (the knee zone)."""

from __future__ import annotations

import numpy as np

from ..frame import Domain, FrameData
from ..stage import Stage
from .base_detail import guided_filter_self
from .multiscale import limit_add


class HotSpotContrast(Stage):
    """Show the inner structure of hot objects that the soft knee compresses.

    The soft knee of ``ToneMap`` puts everything hotter than the scene into a
    narrow display band (e.g. 0.75–1). Hot objects are bright and never
    clipped, but their inner temperature differences become small – a hot
    strip looks like an even white area.

    This module raises local contrast only there:

    1. weight ``w``: 0 below the knee, rising to 1 over ``width`` display
       units above it (smooth, so there is no visible border);
    2. inner structure = ``y − guided_filter(y)`` (self-guided filter,
       radius ``sigma``, edge ``edge``). The guided filter keeps the strong
       border between the hot object and the scene in its smooth part, so
       the structure layer is ≈ 0 at the border and contains only the
       texture inside the object;
    3. ``y_out = y + w · gain · structure``, added with a smooth limit so no
       pixel reaches 0 or 1.

    A first version used the local mean of the hot pixels instead of the
    guided filter. It drew a dark outline around hot objects (pixels in the
    fade-in zone are darker than the hot core and were pushed down). The
    guided filter does not do this (WORKLOG entry 20).

    The order of temperatures *inside* a hot object can change locally (as
    with any local contrast method); the object as a whole stays the
    brightest part of the image. The rest of the scene is not changed.

    Reads the knee position from ``meta["tonemap.knee"]`` (written by
    ``ToneMap``); ``knee`` is used if it is missing.

    Parameters
    ----------
    gain : extra local contrast inside hot objects (0 = off)
    sigma : radius (px) of the guided filter = size of the raised structure
    edge : contrast (display units) that counts as an object border
    width : display range above the knee over which the effect fades in
    knee : fallback knee position
    enabled : switched on / off
    """

    name = "HotSpotContrast"

    def __init__(self, gain: float = 2.0, sigma: float = 3.0,
                 edge: float = 0.03, width: float = 0.05, knee: float = 0.75,
                 enabled: bool = True) -> None:
        super().__init__()
        self.gain = gain
        self.sigma = sigma
        self.edge = edge
        self.width = width
        self.knee = knee
        self.enabled = enabled

    def warmup(self, frame: FrameData) -> FrameData:
        """No state between frames."""
        return frame

    def process(self, frame: FrameData) -> FrameData:
        frame.require(Domain.DISPLAY)
        y = frame.image.astype(np.float32, copy=False)
        if self.gain <= 0:
            return frame
        k = float(frame.meta.get("tonemap.knee", self.knee))
        t = np.clip((y - k) / max(self.width, 1e-3), 0.0, 1.0)
        w = (t * t * (3.0 - 2.0 * t)).astype(np.float32)  # smoothstep
        if not w.any():
            frame.meta["hotspot.share"] = 0.0
            return frame
        r = max(1, int(round(self.sigma)))
        base = guided_filter_self(y, r, self.edge ** 2)
        add = w * self.gain * (y - base)
        frame.image = limit_add(y, add).astype(np.float32)
        frame.meta["hotspot.share"] = float(np.mean(w > 0.5))
        return frame
