"""Step 2a – residual non-uniformity correction (scene-based, per pixel)."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from ..frame import Domain, FrameData
from ..stage import Stage
from ..utils import estimate_noise_sigma, gaussian


class ResidualNUC(Stage):
    """Remove the fixed per-pixel offset pattern that the factory NUC left.

    Idea (scene-based NUC, LMS type – Scribner et al.): the fixed pattern
    stays at the same sensor pixel, while the scene moves with the drone.
    The high-pass residual ``e = y - blur(y)`` of a corrected frame ``y``
    therefore contains scene detail, which averages out over time, and the
    fixed pattern, which does not. The per-pixel offset ``o`` follows the
    running average of ``e``::

        y = x - o
        o <- o + alpha * w * clip(e, +-clip_k * sigma)

    Protection against learning the scene ("burn-in", ghosts):

    - **motion gate**: no update if the frame did not change more than
      ``motion_k`` x the change expected from noise alone (hovering drone);
    - **edge weight** ``w``: smaller update near strong scene edges, where
      ``e`` is large because of the scene, not the sensor;
    - **clipping** of ``e``: hot objects cannot push the offset far;
    - bad pixels (``masks["bad_pixels"]``) are not updated.

    ``fit`` computes a start offset from many frames before the run (all
    frames are corrected well from the first one). ``reset`` returns to the
    fitted offset, so ``Pipeline.run`` keeps it.

    Only offsets are corrected. A residual gain error would show up as a
    pattern that grows with scene contrast; with ~500 DN of scene range it is
    small here and can be added later if needed.

    Output: corrected ``image``. ``meta``: ``nuc.updated`` (bool),
    ``nuc.motion`` (frame change / noise-only change), ``nuc.offset_std``.

    Parameters
    ----------
    alpha : update rate per frame (0.02 = time constant of ~50 frames)
    blur_sigma : std (px) of the Gaussian that separates scene from pattern
    clip_k : clip of the residual, in units of the noise std
    edge_k : edge scale for the weight, in units of the noise std
    motion_k : minimum frame change (relative to noise only) for an update
    adapt : keep adapting during the run (False = use the fitted offset only)
    """

    name = "ResidualNUC"

    def __init__(self, alpha: float = 0.02, blur_sigma: float = 2.0,
                 clip_k: float = 5.0, edge_k: float = 3.0,
                 motion_k: float = 3.0, adapt: bool = True) -> None:
        super().__init__()
        self.alpha = alpha
        self.blur_sigma = blur_sigma
        self.clip_k = clip_k
        self.edge_k = edge_k
        self.motion_k = motion_k
        self.adapt = adapt
        self.offset: np.ndarray | None = None
        self._fitted: np.ndarray | None = None
        self._prev: np.ndarray | None = None

    # ----- helpers ---------------------------------------------------------
    def _residual_and_weight(self, y: np.ndarray, sigma: float):
        smooth = gaussian(y, self.blur_sigma)
        e = np.clip(y - smooth, -self.clip_k * sigma, self.clip_k * sigma)
        gx = cv2.Sobel(smooth, cv2.CV_32F, 1, 0, ksize=3) / 8.0
        gy = cv2.Sobel(smooth, cv2.CV_32F, 0, 1, ksize=3) / 8.0
        grad = np.sqrt(gx * gx + gy * gy)
        scale = self.edge_k * sigma / self.blur_sigma
        w = 1.0 / (1.0 + (grad / max(scale, 1e-3)) ** 2)
        return e, w.astype(np.float32)

    # ----- calibration -----------------------------------------------------
    def fit(self, images: Iterable[np.ndarray]) -> "ResidualNUC":
        """Estimate the offset from many (moving) frames before the run.

        ``images``: iterable of decoded linear float32 frames, e.g. every
        3rd frame of the recording. The weighted mean of the residuals over
        all frames is the start offset (constant-statistics idea).
        """
        num = den = None
        n = 0
        for x in images:
            x = x.astype(np.float32, copy=False)
            sigma = estimate_noise_sigma(x)
            e, w = self._residual_and_weight(x, sigma)
            num = e * w if num is None else num + e * w
            den = w.copy() if den is None else den + w
            n += 1
        if n == 0:
            raise ValueError("fit() got no frames")
        off = num / np.maximum(den, 1e-6)
        off -= off.mean()
        self._fitted = off.astype(np.float32)
        self.offset = self._fitted.copy()
        return self

    def save(self, path: str | Path) -> None:
        if self.offset is None:
            raise ValueError("no offset yet")
        np.save(path, self.offset)

    def load(self, path: str | Path) -> "ResidualNUC":
        self._fitted = np.load(path).astype(np.float32)
        self.offset = self._fitted.copy()
        return self

    def reset(self) -> None:
        self.offset = None if self._fitted is None else self._fitted.copy()
        self._prev = None

    # ----- main ------------------------------------------------------------
    def process(self, frame: FrameData) -> FrameData:
        frame.require(Domain.LINEAR)
        x = frame.image
        if self.offset is None:
            self.offset = np.zeros_like(x)
        y = x - self.offset

        sigma = frame.meta.get("noise_sigma") or estimate_noise_sigma(y)
        updated = False
        motion = float("nan")
        if self.adapt:
            if self._prev is not None:
                # mean |x1 - x2| of two noise-only frames = 2/sqrt(pi) * sigma
                motion = float(np.mean(np.abs(x - self._prev))) / (1.128 * sigma)
                if motion > self.motion_k:
                    e, w = self._residual_and_weight(y, sigma)
                    bad = frame.masks.get("bad_pixels")
                    if bad is not None:
                        w = np.where(bad, 0.0, w)
                    self.offset += self.alpha * w * e
                    self.offset -= self.offset.mean()
                    y = x - self.offset
                    updated = True
            self._prev = x.copy()

        frame.image = y
        frame.meta["nuc.updated"] = updated
        frame.meta["nuc.motion"] = motion
        frame.meta["nuc.offset_std"] = float(self.offset.std())
        return frame
