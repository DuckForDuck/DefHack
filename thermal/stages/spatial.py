"""Step 3b – light, noise-adaptive spatial denoising."""

from __future__ import annotations

import cv2
import numpy as np

from ..frame import Domain, FrameData
from ..stage import Stage
from ..utils import estimate_noise_sigma


def _box(a: np.ndarray, r: int) -> np.ndarray:
    return cv2.blur(a, (2 * r + 1, 2 * r + 1), borderType=cv2.BORDER_REFLECT)


class SpatialDenoise(Stage):
    """Remove the noise that is left after temporal denoising.

    The strength follows the remaining noise per pixel
    (``layers["noise_sigma_map"]`` from ``TemporalDenoise``; if missing,
    ``meta["noise_sigma"]`` is used for the whole frame). So areas without
    temporal history (new image content at the frame border, moving objects,
    occlusions) get more spatial filtering, well-averaged areas almost none.

    Methods:

    - ``"guided"`` (default): self-guided filter (He et al.). Per pixel
      ``a = var / (var + eps)``, ``out = a * x + (1 - a) * mean`` with
      ``eps = (strength * sigma_map)^2``. Flat areas (var ≈ noise) are
      smoothed, edges and texture (var >> noise) are kept. Fast, no halos.
    - ``"bilateral"``: OpenCV bilateral filter; one strength per frame
      (median of the noise map).

    Non-local means was tried, but OpenCV 5 Python accepts only 8-bit input
    for it, which would lose the precision of the linear data.

    Measured on a synthetic sequence (real frame + known motion + noise
    sigma 6, see WORKLOG): after ``TemporalDenoise`` the error drops from
    2.19 to 2.08 DN (strength 1.0) and 2.00 DN (strength 1.5).

    Output: denoised ``image``; ``layers["noise_sigma_map"]`` and
    ``meta["noise_sigma"]`` are scaled by an estimate of the noise reduction.

    Parameters
    ----------
    method : "guided" or "bilateral"
    strength : multiplier of the noise sigma used by the filter
    radius : window radius (guided: box radius; bilateral: d = 2r+1)
    """

    name = "SpatialDenoise"

    def __init__(self, method: str = "guided", strength: float = 1.5,
                 radius: int = 2) -> None:
        super().__init__()
        if method not in ("guided", "bilateral"):
            raise ValueError("method must be 'guided' or 'bilateral'")
        self.method = method
        self.strength = strength
        self.radius = radius

    def _guided(self, x: np.ndarray, sig: np.ndarray) -> np.ndarray:
        r = self.radius
        m = _box(x, r)
        var = np.maximum(_box(x * x, r) - m * m, 0.0)
        eps = (self.strength * sig) ** 2
        a = var / (var + eps + 1e-6)
        b = m - a * m
        return _box(a, r) * x + _box(b, r)

    def process(self, frame: FrameData) -> FrameData:
        frame.require(Domain.LINEAR)
        x = frame.image.astype(np.float32, copy=False)
        sig = frame.layers.get("noise_sigma_map")
        if sig is None:
            s = frame.meta.get("noise_sigma") or estimate_noise_sigma(x)
            sig = np.full_like(x, s)
        sigma = float(np.median(sig))

        if self.method == "guided":
            out = self._guided(x, sig)
        else:
            d = 2 * self.radius + 1
            out = cv2.bilateralFilter(x, d, 2.0 * self.strength * sigma, d / 2.0)

        # Rough estimate of the remaining noise: the removed part is mostly
        # noise, so var_out ≈ max(var_in - var_removed, small floor).
        removed = float(np.var(out - x))
        factor = np.sqrt(max(1.0 - removed / max(sigma * sigma, 1e-6), 0.1))
        frame.image = out.astype(np.float32, copy=False)
        frame.layers["noise_sigma_map"] = (sig * factor).astype(np.float32)
        frame.meta["noise_sigma"] = sigma * factor
        frame.meta["spatial.removed_std"] = float(np.sqrt(removed))
        return frame
