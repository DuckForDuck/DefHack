"""Compact hot-object boost (white top-hat) – idea from Michael's ``thermal_isp``."""

from __future__ import annotations

import cv2
import numpy as np

from ..frame import Domain, FrameData
from ..stage import Stage
from .multiscale import limit_add


class CompactHotBoost(Stage):
    """Make small hot objects brighter than large warm surfaces.

    White top-hat of the linear frame: ``th = x - opening(x, size)``. The
    morphological opening removes every bright structure smaller than the
    window, so ``th`` is large only for *compact* objects that are hotter
    than their surroundings (an animal, a person, an engine, a lone hot
    spot). Large warm areas (road, roofs, tree crowns) give ``th ≈ 0``.

    The part of ``th`` above ``threshold_dn`` (or ``sigma_k`` x noise, if
    larger) is added to the display image:
    ``add = gain · lin_slope · max(th - thr, 0)`` where ``lin_slope`` is
    display units per DN of the tone map (``meta["tonemap.lin_slope"]``).

    ``soft = True`` (default): the addition is limited smoothly by the
    distance to white (``limit_add``), so a hot object comes close to white
    but keeps its inner structure. ``soft = False``: plain addition and
    clipping, as in Michael's code (the object becomes a flat white blob).

    Computed at half resolution (like Michael: opening 7x7 at half res ≈
    14 px at full res), then upsampled. Needs ``layers["linear"]`` (set by
    ``ToneMap``). Place it right after ``ToneMap``.

    Parameters
    ----------
    gain : boost, x linear tone-map slope (Michael: 1.1)
    size : opening window at half resolution (px, odd); objects smaller
        than ≈ 2 x size px count as compact
    threshold_dn : contrast above the surroundings before the boost starts
    sigma_k : threshold floor in units of the noise std
    soft : smooth approach to white instead of clipping
    """

    name = "CompactHotBoost"

    def __init__(self, gain: float = 1.1, size: int = 7,
                 threshold_dn: float = 150.0, sigma_k: float = 25.0,
                 soft: bool = True, enabled: bool = False) -> None:
        super().__init__()
        self.gain = gain
        self.size = size
        self.threshold_dn = threshold_dn
        self.sigma_k = sigma_k
        self.soft = soft
        self.enabled = enabled

    def warmup(self, frame: FrameData) -> FrameData:
        return frame

    def process(self, frame: FrameData) -> FrameData:
        frame.require(Domain.DISPLAY)
        x = frame.layers.get("linear")
        slope = frame.meta.get("tonemap.lin_slope")
        if x is None or slope is None or self.gain <= 0:
            return frame
        h, w = x.shape
        k = int(self.size) | 1
        xs = cv2.resize(x, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
        th = xs - cv2.morphologyEx(xs, cv2.MORPH_OPEN, np.ones((k, k), np.uint8))
        th = cv2.resize(th, (w, h), interpolation=cv2.INTER_LINEAR)
        thr = max(self.threshold_dn, self.sigma_k * float(frame.meta.get("noise_sigma", 0.0)))
        add = (self.gain * slope) * np.maximum(th - thr, 0.0)
        y = frame.image
        if self.soft:
            out = limit_add(y, add)
        else:
            out = np.clip(y + add, 0.0, 1.0)
        frame.image = out.astype(np.float32)
        frame.meta["tophat.share"] = float(np.mean(add > 1.0 / 255))
        return frame
