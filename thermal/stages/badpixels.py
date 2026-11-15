"""Step 1 – detection and replacement of bad pixels."""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np

from ..frame import Domain, FrameData
from ..stage import Stage
from ..utils import estimate_noise_sigma, neighbour_max_min


class BadPixels(Stage):
    """Find dead, stuck, hot/cold and blinking pixels and replace them.

    Detection, in each frame:

    - **Impulse test** (hot / cold / blinking): the excess of a pixel over
      the *maximum* of its 8 neighbours (or below their *minimum*), in units
      of the noise std. Scene details are rarely single pixels, because the
      optics and the bolometer blur them; the hot spot of ~10x6 px in
      frame 100 has no excess at all.
    - A pixel with excess > ``k_replace`` is replaced in this frame
      (blinking pixels, strong single-frame defects).
    - **Persistent**: a pixel with excess > ``k`` in more than
      ``persist_ratio`` of recent frames is added to the permanent mask.
      Why two thresholds: in this recording, thin scene details (sky seen
      through 1-px gaps in leaves, frames 102-123) reach 8-12 sigma in one
      frame. They move with the scene, so they never become persistent, and
      they stay below ``k_replace``. Real bad pixels reach tens or hundreds
      of sigma.
    - **Stuck / dead**: a pixel whose value almost does not change while the
      scene moves (temporal change < ``stuck_ratio`` x the median change of
      all pixels) is added to the permanent mask.

    Replacement: median of the valid neighbours in a 3x3 window; if none is
    valid, in a 5x5 window; else the frame median.

    Output: corrected ``image``; ``masks["bad_pixels"]`` (all pixels replaced
    in this frame); ``meta["noise_sigma"]`` (estimated noise, DN);
    ``meta["badpix.static"]``, ``meta["badpix.dynamic"]`` (counts).

    Known limit: clusters of 2+ adjacent bad pixels that are not in the
    permanent mask pass the impulse test. Give them in ``static_mask``
    (e.g. from a factory map or from ``save_mask`` of an earlier run).

    Parameters
    ----------
    k : threshold (noise std units) for learning persistent bad pixels
    k_replace : threshold for replacing a single-frame impulse at once
    static_mask : bool array (True = bad) or path to a .npy file; kept on reset
    learn : learn persistent and stuck pixels during the run
    persist_ratio : share of frames in which a pixel must be an impulse
    ema : rate of the running averages used by ``learn``
        (0.05 = memory of ~20 frames)
    stuck_ratio : threshold for stuck pixels (see above)
    warmup : frames before learned masks are used
    """

    name = "BadPixels"

    def __init__(self, k: float = 8.0, k_replace: float = 15.0, static_mask: np.ndarray | str | None = None,
                 learn: bool = True, persist_ratio: float = 0.5,
                 ema: float = 0.05, stuck_ratio: float = 0.05,
                 warmup: int = 30) -> None:
        super().__init__()
        self.k = k
        self.k_replace = k_replace
        self.learn = learn
        self.persist_ratio = persist_ratio
        self.ema = ema
        self.stuck_ratio = stuck_ratio
        self.warmup = warmup
        if isinstance(static_mask, (str, Path)):
            static_mask = np.load(static_mask)
        self.static_mask = None if static_mask is None else static_mask.astype(bool)
        self.reset()

    def reset(self) -> None:
        self._n = 0
        self._prev: np.ndarray | None = None
        self._impulse_rate: np.ndarray | None = None
        self._change: np.ndarray | None = None
        self.learned_mask: np.ndarray | None = None

    # ----- masks -----------------------------------------------------------
    def permanent_mask(self, shape: tuple[int, int]) -> np.ndarray:
        m = np.zeros(shape, bool)
        if self.static_mask is not None:
            m |= self.static_mask
        if self.learned_mask is not None:
            m |= self.learned_mask
        return m

    def save_mask(self, path: str | Path) -> None:
        """Save static + learned mask, to reuse as ``static_mask`` later."""
        if self.learned_mask is None and self.static_mask is None:
            raise ValueError("no mask yet; run some frames first")
        shape = (self.learned_mask if self.learned_mask is not None
                 else self.static_mask).shape
        np.save(path, self.permanent_mask(shape))

    def _update_learned(self, img: np.ndarray, impulses: np.ndarray) -> None:
        a = self.ema
        if self._impulse_rate is None:
            self._impulse_rate = impulses.astype(np.float32)
        else:
            self._impulse_rate += a * (impulses - self._impulse_rate)
        if self._prev is not None:
            change = np.abs(img - self._prev)
            if self._change is None:
                self._change = change
            else:
                self._change += a * (change - self._change)
        self._prev = img.copy()
        if self._n < self.warmup or self._change is None:
            return
        scene_change = float(np.median(self._change))
        persistent = self._impulse_rate > self.persist_ratio
        if scene_change > 0:
            stuck = self._change < self.stuck_ratio * scene_change
        else:  # camera and scene static: cannot judge stuck pixels
            stuck = np.zeros_like(persistent)
        self.learned_mask = persistent | stuck

    # ----- replacement -----------------------------------------------------
    @staticmethod
    def _replace(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
        ys, xs = np.nonzero(mask)
        if ys.size == 0:
            return img
        out = img.copy()
        todo = np.ones(ys.size, bool)
        h, w = img.shape
        for r in (1, 2):
            pad_img = np.pad(img, r, mode="reflect")
            pad_bad = np.pad(mask, r, mode="reflect")
            offs = [(dy, dx) for dy in range(-r, r + 1) for dx in range(-r, r + 1)
                    if (dy, dx) != (0, 0)]
            yy, xx = ys[todo] + r, xs[todo] + r
            vals = np.stack([pad_img[yy + dy, xx + dx] for dy, dx in offs], 1)
            bad = np.stack([pad_bad[yy + dy, xx + dx] for dy, dx in offs], 1)
            vals = np.where(bad, np.nan, vals)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN rows
                med = np.nanmedian(vals, axis=1)
            ok = ~np.isnan(med)
            idx = np.flatnonzero(todo)
            out[ys[idx[ok]], xs[idx[ok]]] = med[ok]
            todo[idx[ok]] = False
            if not todo.any():
                return out
        out[ys[todo], xs[todo]] = np.median(img[~mask]) if (~mask).any() else 0
        return out

    # ----- main ------------------------------------------------------------
    def process(self, frame: FrameData) -> FrameData:
        frame.require(Domain.LINEAR)
        img = frame.image
        permanent = self.permanent_mask(img.shape)

        sigma = estimate_noise_sigma(img, permanent if permanent.any() else None)
        mx, mn = neighbour_max_min(img)
        excess = np.maximum(img - mx, mn - img) / max(sigma, 1e-3)
        candidates = (excess > self.k) & ~permanent
        impulses = (excess > self.k_replace) & ~permanent

        if self.learn:
            self._update_learned(img, candidates)
        self._n += 1

        mask = permanent | impulses
        frame.image = self._replace(img, mask)
        frame.masks["bad_pixels"] = mask
        frame.meta["noise_sigma"] = sigma
        frame.meta["badpix.static"] = int(permanent.sum())
        frame.meta["badpix.dynamic"] = int(impulses.sum())
        return frame
