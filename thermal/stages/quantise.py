"""Step 9 – quantisation of the display image to 8 bit, with dithering."""

from __future__ import annotations

import cv2
import numpy as np

from ..frame import Domain, FrameData
from ..stage import Stage


def blue_noise_tile(size: int = 64, seed: int = 7) -> np.ndarray:
    """Tileable threshold map in [0, 1) with mostly high-frequency content.

    White noise is high-pass filtered (wrap-around, so the tile repeats
    without seams) and then rank-transformed to a uniform distribution.
    The result has little low-frequency energy, so the dither is seen as a
    fine, even grain instead of blotches.
    """
    rng = np.random.default_rng(seed)
    n = rng.standard_normal((size, size)).astype(np.float32)
    big = np.tile(n, (3, 3))
    low = cv2.GaussianBlur(big, (0, 0), 1.5)[size:2 * size, size:2 * size]
    hp = n - low
    ranks = np.argsort(np.argsort(hp.ravel())).reshape(size, size)
    return ((ranks + 0.5) / ranks.size).astype(np.float32)


def bayer_tile(n: int = 8) -> np.ndarray:
    """Ordered-dither (Bayer) threshold map n x n in [0, 1)."""
    m = np.array([[0]], dtype=np.int64)
    while m.shape[0] < n:
        m = np.block([[4 * m, 4 * m + 2], [4 * m + 3, 4 * m + 1]])
    return ((m + 0.5) / m.size).astype(np.float32)


class Quantise(Stage):
    """Display float 0..1 → uint8 0..255, with dithering against banding.

    Without dither, smooth gradients (sky, road) show visible steps between
    neighbouring grey levels (banding), especially after strong contrast
    changes. A threshold map ``t`` in [0, 1) is used instead of plain
    rounding: ``out = floor(255 * y + 0.5 + amplitude * (t - 0.5))``.
    With ``amplitude = 1`` the average over an area equals the exact value
    (no bias), and the grain has an amplitude of one grey level.

    The pattern is fixed in place (not changing between frames), so it does
    not look like moving noise in the video. Values 0 and 1 stay exactly 0
    and 255 (no dither added at the ends).

    Output: ``image`` uint8, ``domain = OUTPUT``. ``meta``:
    ``quant.white`` / ``quant.black`` (share of 255 / 0).

    Parameters
    ----------
    dither : "blue" (default), "bayer" or "none"
    amplitude : dither amplitude in grey levels (0..1)
    """

    name = "Quantise"
    run_when_frozen = True   # also shows / writes held (shutter) frames

    def __init__(self, dither: str = "blue", amplitude: float = 1.0) -> None:
        super().__init__()
        if dither not in ("blue", "bayer", "none"):
            raise ValueError("dither must be 'blue', 'bayer' or 'none'")
        self.dither = dither
        self.amplitude = amplitude
        self._tile = None
        self._map: np.ndarray | None = None

    def _threshold(self, shape) -> np.ndarray:
        if self._map is not None and self._map.shape == shape:
            return self._map
        if self._tile is None or getattr(self, "_tile_kind", None) != self.dither:
            self._tile = blue_noise_tile() if self.dither == "blue" else bayer_tile()
            self._tile_kind = self.dither
        th, tw = self._tile.shape
        reps = (-(-shape[0] // th), -(-shape[1] // tw))
        self._map = np.tile(self._tile, reps)[:shape[0], :shape[1]]
        return self._map

    def process(self, frame: FrameData) -> FrameData:
        frame.require(Domain.DISPLAY)
        y = np.clip(frame.image.astype(np.float32, copy=False), 0.0, 1.0) * 255.0
        if self.dither == "none" or self.amplitude <= 0:
            q = np.floor(y + 0.5)
        else:
            t = self._threshold(y.shape)
            q = np.floor(y + 0.5 + self.amplitude * (t - 0.5))
            q = np.where(y <= 0.0, 0.0, np.where(y >= 255.0, 255.0, q))
        out = np.clip(q, 0, 255).astype(np.uint8)
        frame.image = out
        frame.domain = Domain.OUTPUT
        frame.meta["quant.white"] = float(np.mean(out == 255))
        frame.meta["quant.black"] = float(np.mean(out == 0))
        return frame
