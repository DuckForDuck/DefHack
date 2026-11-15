"""Step 3a – motion-compensated temporal denoising (recursive filter)."""

from __future__ import annotations

import cv2
import numpy as np

from ..frame import Domain, FrameData
from ..stage import Stage
from ..utils import estimate_noise_sigma, gaussian

_PRESETS = {
    "ultrafast": cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST,
    "fast": cv2.DISOPTICAL_FLOW_PRESET_FAST,
    "medium": cv2.DISOPTICAL_FLOW_PRESET_MEDIUM,
}
_INTERP = {"linear": cv2.INTER_LINEAR, "cubic": cv2.INTER_CUBIC}


def _ramp(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """1 below lo, 0 above hi, linear in between."""
    return np.clip((hi - x) / (hi - lo), 0.0, 1.0)


class TemporalDenoise(Stage):
    """Average each pixel over past frames along the motion of the scene.

    For every new frame ``x``:

    1. **Motion**: dense optical flow (OpenCV DIS) from ``x`` to the previous
       *output*. Both are converted to 8 bit with the same range for the
       flow only; ``x`` is lightly smoothed first so noise does not disturb it.
    2. **Warp** the previous output ``p`` and its per-pixel frame count ``n``
       onto the current frame. Pixels that come from outside the previous
       frame get ``n = 0`` (no history).
    3. **Match test** – where the warped history does not fit the current
       frame (occlusion, parallax, flow error, an object that changed), the
       history is not used. With ``d = x - p_warped`` and the expected noise
       of ``d``, ``s_d = sigma * sqrt(1 + v)``:

       - local test: mean of ``d^2`` in a ``window`` x ``window`` box,
         divided by ``s_d^2`` (≈ 1 for pure noise); full trust below
         ``local_lo``, none above ``local_hi``;
       - pixel test: ``|d| / s_d``; full trust below ``pixel_lo``, none above
         ``pixel_hi`` (catches small objects that the box test dilutes).

       ``alpha`` = product of both trusts (0 … 1).
    4. **Recursive average** (running mean with memory ``max_frames``)::

           n_new = min(alpha * n_warped + 1, max_frames)
           out   = p_warped + (x - p_warped) / n_new

       ``alpha = 1`` → normal running mean; ``alpha = 0`` → ``out = x``.

    The module also tracks the remaining noise variance per pixel and writes
    it as ``layers["noise_sigma_map"]`` (DN). ``SpatialDenoise`` uses it to
    clean more where the temporal average could not be used.

    Physics note: works on LINEAR data, where the noise is additive with a
    constant sigma – this is what the match test assumes.

    Output: denoised ``image``; ``layers["noise_sigma_map"]``,
    ``layers["denoise.frames"]`` (frames averaged per pixel);
    ``meta["noise_sigma"]`` (median remaining noise),
    ``meta["temporal.rejected"]`` (share of pixels with alpha < 0.5),
    ``meta["temporal.flow_median"]`` (px).

    Parameters
    ----------
    max_frames : memory of the running mean; noise std drops up to about
        sqrt(2 * max_frames - 1) times (8 → ≈ 3.9x)
    window : size of the local match test (odd)
    local_lo, local_hi : thresholds of the local test (mean d^2 / s_d^2)
    pixel_lo, pixel_hi : thresholds of the pixel test (|d| / s_d)
    flow_preset : "ultrafast", "fast" or "medium" (DIS presets)
    flow_smooth : Gaussian std (px) applied before the flow
    interp : "cubic" (sharper) or "linear" for warping the history
    """

    name = "TemporalDenoise"

    def __init__(self, max_frames: int = 8, window: int = 5,
                 local_lo: float = 1.6, local_hi: float = 3.0,
                 pixel_lo: float = 3.0, pixel_hi: float = 5.0,
                 flow_preset: str = "medium", flow_smooth: float = 1.0,
                 interp: str = "cubic") -> None:
        super().__init__()
        if window < 3 or window % 2 == 0:
            raise ValueError("window must be odd and >= 3")
        self.max_frames = max_frames
        self.window = window
        self.local_lo, self.local_hi = local_lo, local_hi
        self.pixel_lo, self.pixel_hi = pixel_lo, pixel_hi
        self.flow_preset = flow_preset
        self.flow_smooth = flow_smooth
        self.interp = interp
        self._dis = cv2.DISOpticalFlow_create(_PRESETS[flow_preset])
        self.reset()

    def reset(self) -> None:
        self._prev: np.ndarray | None = None   # previous output
        self._n: np.ndarray | None = None      # frames averaged per pixel
        self._v: np.ndarray | None = None      # remaining noise variance / sigma^2
        self._grid: tuple[np.ndarray, np.ndarray] | None = None

    # ----- motion ----------------------------------------------------------
    def _flow(self, cur: np.ndarray, prev: np.ndarray) -> np.ndarray:
        cur_s = gaussian(cur, self.flow_smooth) if self.flow_smooth > 0 else cur
        lo, hi = np.percentile(cur_s[::4, ::4], [0.5, 99.5])
        scale = 255.0 / max(float(hi - lo), 1e-3)
        to8 = lambda a: np.clip((a - lo) * scale, 0, 255).astype(np.uint8)
        return self._dis.calc(to8(cur_s), to8(prev), None)

    def _warp(self, img: np.ndarray, mx: np.ndarray, my: np.ndarray,
              interp: int) -> np.ndarray:
        return cv2.remap(img, mx, my, interp, borderMode=cv2.BORDER_REPLICATE)

    # ----- main ------------------------------------------------------------
    def process(self, frame: FrameData) -> FrameData:
        frame.require(Domain.LINEAR)
        x = frame.image.astype(np.float32, copy=False)
        sigma = frame.meta.get("noise_sigma") or estimate_noise_sigma(x)
        h, w = x.shape

        if self._prev is None or self._prev.shape != x.shape:
            out = x.copy()
            n = np.ones_like(x)
            v = np.ones_like(x)
            rejected, flow_med = 1.0, 0.0
        else:
            if self._grid is None:
                gy, gx = np.mgrid[0:h, 0:w].astype(np.float32)
                self._grid = (gx, gy)
            gx, gy = self._grid
            flow = self._flow(x, self._prev)
            mx, my = gx + flow[..., 0], gy + flow[..., 1]
            inside = (mx >= 0) & (mx <= w - 1) & (my >= 0) & (my <= h - 1)

            p = self._warp(self._prev, mx, my, _INTERP[self.interp])
            n_w = self._warp(self._n, mx, my, cv2.INTER_LINEAR)
            v_w = self._warp(self._v, mx, my, cv2.INTER_LINEAR)
            n_w[~inside] = 0.0
            v_w[~inside] = 1.0

            d = x - p
            s2 = sigma * sigma * (1.0 + v_w)
            k = self.window
            local = cv2.blur(d * d, (k, k), borderType=cv2.BORDER_REFLECT) / s2
            pixel = np.abs(d) / np.sqrt(s2)
            alpha = (_ramp(local, self.local_lo, self.local_hi)
                     * _ramp(pixel, self.pixel_lo, self.pixel_hi))
            alpha[~inside] = 0.0

            n = np.minimum(alpha * n_w + 1.0, float(self.max_frames))
            gain = 1.0 / n
            out = p + gain * d
            # Remaining noise variance (history and current noise independent).
            v = (1.0 - gain) ** 2 * v_w + gain ** 2
            rejected = float(np.mean(alpha < 0.5))
            flow_med = float(np.median(np.hypot(flow[..., 0], flow[..., 1])))

        self._prev, self._n, self._v = out, n, v
        sigma_map = (sigma * np.sqrt(v)).astype(np.float32)

        frame.image = out.astype(np.float32, copy=False)
        frame.layers["noise_sigma_map"] = sigma_map
        frame.layers["denoise.frames"] = n
        frame.meta["noise_sigma"] = float(np.median(sigma_map))
        frame.meta["temporal.sigma_in"] = float(sigma)
        frame.meta["temporal.rejected"] = rejected
        frame.meta["temporal.flow_median"] = flow_med
        return frame
