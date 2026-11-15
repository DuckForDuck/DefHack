"""Step 2b – per-frame removal of column and row stripes."""

from __future__ import annotations

import cv2
import numpy as np

from ..frame import Domain, FrameData
from ..stage import Stage
from ..utils import estimate_noise_sigma, gaussian


class Destripe(Stage):
    """Remove offsets that are constant along a whole column or row.

    Column read-out amplifiers add one offset to all pixels of a column;
    this offset can change from frame to frame, so ``ResidualNUC`` (which
    learns slowly) does not remove it fully.

    For columns::

        hp      = x - horizontal_box_blur(x, width)   # keeps narrow vertical structure
        offset  = mean over rows of clip(hp, +-clip_k * sigma)
        x      -= offset

    Rows: the same with a vertical blur.

    Protection of the scene:

    - clipping limits the influence of any single pixel to ``clip_k * sigma``;
    - **edge exclusion**: pixels where the smoothed image has a strong
      gradient across the stripe direction (|d/dx| for columns, |d/dy| for
      rows) above ``edge_k * sigma`` per pixel are not used. Scene edges are
      blurred by the optics over several pixels and have such a gradient; a
      1-px stripe does not. Without this, a long horizontal scene edge (the
      tree line in frames 100-125) leaks into the row offsets. With
      edge_k = 0.5 the correlation of removed row offsets with scene edges
      drops from 0.14 to 0.04, while the removed noise stays the same.
    - a column/row with less than ``min_valid`` share of usable pixels gets
      no correction.

    Measured on this recording (after ``ResidualNUC``): in differences of
    consecutive frames, row offsets estimated from the left and the right
    half agree (corr. ~0.43) – real per-frame row noise; columns agree less
    (~0.17).

    Output: corrected ``image``. ``meta``: ``destripe.col_std``,
    ``destripe.row_std`` (std of the removed offsets, DN); ``noise_sigma``
    is estimated again on the corrected frame (≈ 4.1 DN here instead of
    ≈ 6 DN before step 2, because the fixed pattern was counted as noise).

    Parameters
    ----------
    columns, rows : which direction to correct
    width : length of the box blur across the stripes (odd, px)
    clip_k : clip of the high-pass, in units of the noise std
    edge_k : gradient threshold for edge exclusion, in units of the noise std
    edge_sigma : std (px) of the smoothing before the gradient
    min_valid : minimum share of usable pixels in a column/row
    temporal_alpha : 0 = offsets from the current frame only (default).
        > 0 = the offsets are smoothed over time (EMA with this weight per
        frame, idea from Michael's ``thermal_isp``: col_alpha 0.1). Stripes
        are fixed to the sensor while the scene moves, so averaging over
        frames removes scene edges that leak into the estimate, but a
        stripe that changes from frame to frame is then corrected only
        partly. Uses only past frames (streaming).
    """

    name = "Destripe"

    def __init__(self, columns: bool = True, rows: bool = True,
                 width: int = 9, clip_k: float = 2.0, edge_k: float = 0.5,
                 edge_sigma: float = 1.5, min_valid: float = 0.2,
                 temporal_alpha: float = 0.0) -> None:
        super().__init__()
        if width < 3 or width % 2 == 0:
            raise ValueError("width must be odd and >= 3")
        self.columns = columns
        self.rows = rows
        self.width = width
        self.clip_k = clip_k
        self.edge_k = edge_k
        self.edge_sigma = edge_sigma
        self.min_valid = min_valid
        self.temporal_alpha = temporal_alpha
        self.reset()

    def reset(self) -> None:
        self._col: np.ndarray | None = None
        self._row: np.ndarray | None = None

    def _smooth(self, old, new):
        a = float(self.temporal_alpha)
        if a <= 0 or old is None or old.shape != new.shape:
            return new
        return (old + a * (new - old)).astype(np.float32)

    def _offsets(self, x: np.ndarray, axis: int, sigma: float) -> np.ndarray:
        ksize = (self.width, 1) if axis == 0 else (1, self.width)
        hp = x - cv2.blur(x, ksize, borderType=cv2.BORDER_REFLECT)
        c = self.clip_k * sigma
        hp = np.clip(hp, -c, c)
        smooth = gaussian(x, self.edge_sigma)
        dx, dy = (1, 0) if axis == 0 else (0, 1)  # gradient across the stripes
        grad = np.abs(cv2.Sobel(smooth, cv2.CV_32F, dx, dy, ksize=3)) / 8.0
        valid = grad < self.edge_k * sigma
        n = valid.sum(axis=axis)
        off = np.where(valid, hp, 0.0).sum(axis=axis) / np.maximum(n, 1)
        off = np.where(n >= self.min_valid * x.shape[axis], off, 0.0)
        return (off - off.mean()).astype(np.float32)

    def process(self, frame: FrameData) -> FrameData:
        frame.require(Domain.LINEAR)
        x = frame.image
        sigma = frame.meta.get("noise_sigma") or estimate_noise_sigma(x)
        col_std = row_std = 0.0
        if self.columns:
            off = self._col = self._smooth(self._col, self._offsets(x, 0, sigma))
            x = x - off[None, :]
            col_std = float(off.std())
        if self.rows:
            off = self._row = self._smooth(self._row, self._offsets(x, 1, sigma))
            x = x - off[:, None]
            row_std = float(off.std())
        frame.image = x.astype(np.float32, copy=False)
        # The fixed pattern is gone now, so the noise estimate made before
        # step 2 (which counted the pattern as noise) is too high; update it.
        bad = frame.masks.get("bad_pixels")
        frame.meta["noise_sigma"] = estimate_noise_sigma(
            frame.image, bad if bad is not None and bad.any() else None)
        frame.meta["destripe.col_std"] = col_std
        frame.meta["destripe.row_std"] = row_std
        return frame
