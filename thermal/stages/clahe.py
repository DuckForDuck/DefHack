"""Optional local contrast (CLAHE) in the display domain."""

from __future__ import annotations

import cv2
import numpy as np

from ..frame import Domain, FrameData
from ..stage import Stage


class LocalContrast(Stage):
    """CLAHE – contrast-limited adaptive histogram equalisation.

    Equalises the histogram in tiles of the image (``tiles`` x ``tiles``),
    with a limited slope (``clip_limit``), and blends the result with the
    input (``amount``). It raises local contrast in large uniform areas.

    Off by default (``enabled = False``). Reasons (docs, step 6.2 point 5):
    strong CLAHE can make a cooler area brighter than a warmer one (the
    temperature order is lost between tiles), produces halos around hot
    objects, and amplifies noise in flat areas. ``ToneMap`` already does
    global plateau equalisation; CLAHE is here to compare and for mild use.

    Runs on a 16-bit copy of the display image (precision of float data is
    kept better than with 8 bit).

    Parameters
    ----------
    clip_limit : contrast limit of CLAHE (1 = none, 2–3 = typical)
    tiles : number of tiles per side
    amount : blend of the CLAHE result (0 = off, 1 = full CLAHE)
    enabled : switched on / off
    """

    name = "LocalContrast"

    def __init__(self, clip_limit: float = 2.0, tiles: int = 8,
                 amount: float = 0.5, enabled: bool = False) -> None:
        super().__init__()
        self.clip_limit = clip_limit
        self.tiles = tiles
        self.amount = amount
        self.enabled = enabled

    def warmup(self, frame: FrameData) -> FrameData:
        """No state between frames."""
        return frame

    def process(self, frame: FrameData) -> FrameData:
        frame.require(Domain.DISPLAY)
        y = frame.image.astype(np.float32, copy=False)
        if self.amount <= 0:
            return frame
        u16 = np.clip(np.round(y * 65535.0), 0, 65535).astype(np.uint16)
        t = max(1, int(self.tiles))
        clahe = cv2.createCLAHE(clipLimit=float(self.clip_limit), tileGridSize=(t, t))
        eq = clahe.apply(u16).astype(np.float32) / 65535.0
        out = (1.0 - self.amount) * y + self.amount * eq
        frame.image = np.clip(out, 0.0, 1.0).astype(np.float32)
        return frame
