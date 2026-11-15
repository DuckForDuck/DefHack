"""Step 5 – split the linear frame into a base layer and a detail layer."""

from __future__ import annotations

import cv2
import numpy as np

from ..frame import Domain, FrameData
from ..stage import Stage


def guided_filter_self(x: np.ndarray, radius: int, eps: float) -> np.ndarray:
    """Self-guided filter (He et al.): edge-preserving smoothing.

    Where the local variance is much larger than ``eps`` (strong edges,
    hot objects) the image is kept; where it is smaller (texture, noise)
    the local mean is used.
    """
    k = (2 * radius + 1, 2 * radius + 1)
    box = lambda a: cv2.blur(a, k, borderType=cv2.BORDER_REFLECT)
    m = box(x)
    var = np.maximum(box(x * x) - m * m, 0.0)
    a = var / (var + eps)
    b = m - a * m
    return box(a) * x + box(b)


class BaseDetailSplit(Stage):
    """Edge-preserving split: ``image = base + detail`` (Digital Detail Enhancement).

    - ``base``: large temperature differences (sky / ground / trees, hot
      objects). Tone mapping (step 6) compresses it.
    - ``detail``: texture and fine edges of a few to some tens of DN. It is
      added back after tone mapping (steps 6–7), so it is not lost in the
      compression.

    A guided filter is used, not a Gaussian: the result does not depend on
    object *size* but on local *contrast*. A small hot object with a
    contrast far above ``edge_dn`` (the hot spot in frame 100: +390 DN)
    stays in ``base`` and therefore goes through the soft knee of step 6.

    ``image`` is not changed. Output: ``layers["base"]``, ``layers["detail"]``;
    ``meta["split.detail_std"]``.

    Parameters
    ----------
    radius : window radius of the guided filter (px)
    edge_dn : contrast (DN) that counts as an edge: eps = edge_dn^2.
        Smaller → more goes to base; larger → more goes to detail.
    """

    name = "BaseDetailSplit"

    def __init__(self, radius: int = 8, edge_dn: float = 25.0) -> None:
        super().__init__()
        self.radius = radius
        self.edge_dn = edge_dn

    def process(self, frame: FrameData) -> FrameData:
        frame.require(Domain.LINEAR)
        x = frame.image.astype(np.float32, copy=False)
        # Work relative to the mean for better float32 precision of x*x.
        mean = float(x.mean())
        base = guided_filter_self(x - mean, self.radius,
                                  self.edge_dn ** 2) + mean
        detail = x - base
        frame.layers["base"] = base.astype(np.float32)
        frame.layers["detail"] = detail.astype(np.float32)
        frame.meta["split.detail_std"] = float(detail.std())
        return frame
