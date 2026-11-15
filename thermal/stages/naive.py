"""Baseline: the naive processing of the original ``script.py``.

Used only for comparison with the real pipeline. It shows the typical
problems: hard clipping (hot objects become flat white blobs), flicker
(percentiles change every frame), amplified noise and stripes.
"""

from __future__ import annotations

import numpy as np

from ..frame import Domain, FrameData
from ..stage import Stage


class NaiveAGC(Stage):
    """Per-frame percentile stretch with hard clipping, as in ``script.py``.

    Actions (identical to ``script.py.get_frame``):
    1. ``vmin``, ``vmax`` = percentiles ``low`` and ``high`` of this frame only.
    2. Clip the frame to ``[vmin, vmax]``.
    3. Scale linearly to 0..255 and convert to uint8 by truncation.

    The overlay text with the frame number from ``script.py`` is not
    reproduced, because it changes pixel values.

    Output: ``image`` is uint8, ``domain = OUTPUT``. Diagnostics in ``meta``:
    ``naive.lo``, ``naive.hi`` (DN), ``naive.black`` and ``naive.white``
    (share of pixels at 0 and 255).

    Parameters
    ----------
    low, high : percentiles used for the stretch (script.py: 2 and 98)
    """

    name = "NaiveAGC"

    def __init__(self, low: float = 2.0, high: float = 98.0) -> None:
        super().__init__()
        if not 0 <= low < high <= 100:
            raise ValueError("need 0 <= low < high <= 100")
        self.low = low
        self.high = high

    def process(self, frame: FrameData) -> FrameData:
        frame.require(Domain.LINEAR)
        img = frame.image
        vmin, vmax = np.percentile(img, [self.low, self.high])
        span = max(float(vmax - vmin), 1e-6)
        clipped = np.clip(img, vmin, vmax)
        out = ((clipped - vmin) / span * 255).astype(np.uint8)  # truncation

        frame.image = out
        frame.domain = Domain.OUTPUT
        frame.meta["naive.lo"] = float(vmin)
        frame.meta["naive.hi"] = float(vmax)
        frame.meta["naive.black"] = float(np.mean(out == 0))
        frame.meta["naive.white"] = float(np.mean(out == 255))
        return frame
