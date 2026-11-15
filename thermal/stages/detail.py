"""Step 7 – detail enhancement (sharpness) in the display domain."""

from __future__ import annotations

import numpy as np

from ..frame import Domain, FrameData
from ..stage import Stage
from ..utils import gaussian
from .multiscale import limit_add


class DetailEnhance(Stage):
    """Noise-gated unsharp masking of the fine detail band.

    Measurement (WORKLOG entry 12): scene detail exists up to ≈ 0.22 cycles
    per pixel; above that there is only noise. The band boosted here is
    ``band = y - gaussian(y, sigma)``; with ``sigma`` ≈ 1.2 px it covers the
    upper part of the useful range (≈ 0.1–0.25 cycles/px).

    1. **Noise gate (coring)**: the expected noise in the band, in display
       units, is ``noise_sigma_map * tonemap.slope * band_factor``. A band
       value much larger than that is kept fully, a value of the order of
       the noise is suppressed:
       ``w = band^2 / (band^2 + (coring_k * noise)^2)``.
       So sharpening does not bring back the noise removed in step 3.
    2. **Halo limit**: the added amount is clipped to ``±max_amp`` (display
       units), so strong edges (hot objects against cold background) do not
       get bright/dark rims.
    3. ``add = gain * w * clip(band)`` is added with a smooth limit that
       depends on the distance to 0 / 1:
       ``y + add * (1 - y) / (1 - y + add)`` for add > 0 (mirror for add < 0).
       Detail can never push a pixel to 0 or 1, and a pixel that is already
       0 or 1 (hard clipping in ``ToneMap`` for comparison) stays there.

    Mid-size detail (texture of tree crowns, field structure) is controlled
    by ``ToneMap(detail_gain=...)``; this module handles the fine scale.

    Output: sharpened ``image`` (DISPLAY). ``meta["detail.added_std"]``.

    Parameters
    ----------
    gain : extra amount of the fine band (0 = off, 1 = double)
    sigma : Gaussian std (px) that defines the band
    coring_k : noise gate threshold, x expected band noise
    max_amp : maximum added value (display units, 0.06 ≈ 15 grey levels)
    """

    name = "DetailEnhance"

    def __init__(self, gain: float = 1.0, sigma: float = 1.2,
                 coring_k: float = 2.0, max_amp: float = 0.06) -> None:
        super().__init__()
        self.gain = gain
        self.sigma = sigma
        self.coring_k = coring_k
        self.max_amp = max_amp

    def warmup(self, frame: FrameData) -> FrameData:
        """No state between frames: nothing to prepare."""
        return frame

    def process(self, frame: FrameData) -> FrameData:
        frame.require(Domain.DISPLAY)
        y = frame.image.astype(np.float32, copy=False)
        if self.gain <= 0:
            return frame
        band = y - gaussian(y, self.sigma)

        sig = frame.layers.get("noise_sigma_map")
        slope = frame.layers.get("tonemap.slope")
        if sig is not None and slope is not None:
            # White noise through (1 - Gaussian): variance x (1 - 3 / (4 pi s^2)).
            factor = np.sqrt(max(1.0 - 3.0 / (4.0 * np.pi * self.sigma ** 2), 0.05))
            noise = sig * slope * factor
            t2 = (self.coring_k * noise) ** 2
            w = band * band / (band * band + t2 + 1e-12)
        else:
            w = 1.0

        add = self.gain * w * np.clip(band, -self.max_amp, self.max_amp)
        out = limit_add(y, add)
        frame.image = out.astype(np.float32)
        frame.meta["detail.added_std"] = float(np.std(add))
        return frame
