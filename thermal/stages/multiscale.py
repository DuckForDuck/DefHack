"""Step 7 (alternative) – multi-scale detail enhancement."""

from __future__ import annotations

import numpy as np

from ..frame import Domain, FrameData
from ..stage import Stage
from ..utils import gaussian


def limit_add(y: np.ndarray, add: np.ndarray) -> np.ndarray:
    """Add ``add`` to ``y`` (0..1) without pushing any pixel to 0 or 1.

    ``y + add * (1 - y) / (1 - y + add)`` for add > 0, mirrored for add < 0.
    Pixels that are already 0 or 1 stay there.
    """
    eps = 1e-6
    return np.where(add >= 0,
                    y + add * (1.0 - y) / (1.0 - y + add + eps),
                    y + add * y / (y - add + eps))


class MultiScaleDetail(Stage):
    """Boost detail on three scales; small noise and strong edges are spared.

    The display image is split into band-pass layers with Gaussians
    (difference of Gaussians, like a Laplacian pyramid without
    down-sampling)::

        fine   = y          - G(sigma_1)        ≈ 1-2 px detail
        mid    = G(sigma_1) - G(sigma_2)        ≈ 2-5 px (leaves, small objects)
        coarse = G(sigma_2) - G(sigma_3)        ≈ 5-12 px (crowns, field structure)

    Each band value ``b`` is boosted with a curve that depends on its size:

    - noise gate: ``w = b² / (b² + (coring_k · noise_band)²)`` – values of the
      size of the remaining noise are not boosted (noise from the noise map
      of step 3 × tone-map slope × band factor);
    - edge protection: the extra gain falls off for ``|b|`` larger than
      ``edge`` – strong edges (hot object against cold background) are not
      boosted, so no halos;
    - ``b_out = b + gain · b · w · edge² / (edge² + b²)``.

    So mainly texture of moderate contrast is raised – the part that looks
    "soft" – and the result does not look "painted" the way a strong single
    unsharp mask does. Compared with ``DetailEnhance`` (one fine band).

    Off by default; switch on in the tuner and switch ``DetailEnhance`` off
    to compare.

    Parameters
    ----------
    fine, mid, coarse : extra gain per band (0 = band unchanged)
    sigma_1, sigma_2, sigma_3 : Gaussian std (px) that define the bands
    edge : band value (display units) above which the boost falls off
    coring_k : noise gate threshold, x expected band noise
    enabled : switched on / off
    """

    name = "MultiScaleDetail"

    def __init__(self, fine: float = 0.6, mid: float = 1.0, coarse: float = 0.6,
                 sigma_1: float = 1.0, sigma_2: float = 2.5, sigma_3: float = 6.0,
                 edge: float = 0.06, coring_k: float = 3.0,
                 enabled: bool = False) -> None:
        super().__init__()
        self.fine, self.mid, self.coarse = fine, mid, coarse
        self.sigma_1, self.sigma_2, self.sigma_3 = sigma_1, sigma_2, sigma_3
        self.edge = edge
        self.coring_k = coring_k
        self.enabled = enabled

    def warmup(self, frame: FrameData) -> FrameData:
        """No state between frames."""
        return frame

    @staticmethod
    def _band_noise_factor(s_lo: float, s_hi: float) -> float:
        """Std of white noise (std 1) after G(s_lo) - G(s_hi); s_lo = 0 = identity."""
        def energy(s):  # sum of squared Gaussian weights
            return 1.0 if s <= 0 else 1.0 / (4.0 * np.pi * s * s)
        def cross(a, b):  # sum of products of two Gaussians
            if a <= 0:  # identity x Gaussian = centre weight of the Gaussian
                return 1.0 / (2.0 * np.pi * b * b)
            return 1.0 / (2.0 * np.pi * (a * a + b * b))
        v = energy(s_lo) + energy(s_hi) - 2.0 * cross(s_lo, s_hi)
        return float(np.sqrt(max(v, 1e-6)))

    def process(self, frame: FrameData) -> FrameData:
        frame.require(Domain.DISPLAY)
        y = frame.image.astype(np.float32, copy=False)
        g1 = gaussian(y, self.sigma_1)
        g2 = gaussian(y, self.sigma_2)
        g3 = gaussian(y, self.sigma_3)
        bands = [(y - g1, self.fine, 0.0, self.sigma_1),
                 (g1 - g2, self.mid, self.sigma_1, self.sigma_2),
                 (g2 - g3, self.coarse, self.sigma_2, self.sigma_3)]

        sig = frame.layers.get("noise_sigma_map")
        slope = frame.layers.get("tonemap.slope")
        noise = sig * slope if (sig is not None and slope is not None) else None
        e2 = self.edge * self.edge

        add = np.zeros_like(y)
        for b, gain, s_lo, s_hi in bands:
            if gain <= 0:
                continue
            if noise is not None:
                n = noise * self._band_noise_factor(s_lo, s_hi)
                w = b * b / (b * b + (self.coring_k * n) ** 2 + 1e-12)
            else:
                w = 1.0
            add += gain * b * w * e2 / (e2 + b * b)

        frame.image = limit_add(y, add).astype(np.float32)
        frame.meta["multiscale.added_std"] = float(add.std())
        return frame
