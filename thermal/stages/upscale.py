"""Step 8 – upscaling of the display image."""

from __future__ import annotations

import cv2
import numpy as np

from ..frame import Domain, FrameData
from ..stage import Stage

_METHODS = {
    "lanczos": cv2.INTER_LANCZOS4,
    "cubic": cv2.INTER_CUBIC,
    "linear": cv2.INTER_LINEAR,
    "nearest": cv2.INTER_NEAREST,
}


class Upscale(Stage):
    """Enlarge the frame (default 2x: 640 x 512 → 1280 x 1024).

    Runs on the float display image, **before** quantisation to 8 bit, so
    no 8-bit steps are enlarged (they would become wider, more visible bands).

    It adds no new detail: the measurement for step 4 (WORKLOG entry 12)
    shows that the scene has detail only up to ≈ 0.23 cycles/px, so a good
    interpolation loses nothing and adds no pixel blocks. Lanczos keeps
    edges sharper than bicubic; its small overshoot at edges is clipped to
    0..1.

    Output: enlarged ``image``. Other layers keep their original size.

    Parameters
    ----------
    scale : enlargement factor
    method : "lanczos" (default), "cubic", "linear" or "nearest"
    """

    name = "Upscale"
    run_when_frozen = True   # also shows / writes held (shutter) frames

    def __init__(self, scale: float = 2.0, method: str = "lanczos") -> None:
        super().__init__()
        if method not in _METHODS:
            raise ValueError(f"method must be one of {list(_METHODS)}")
        self.scale = scale
        self.method = method

    def process(self, frame: FrameData) -> FrameData:
        if self.scale == 1.0:
            return frame
        img = frame.image
        h, w = img.shape[:2]
        size = (int(round(w * self.scale)), int(round(h * self.scale)))
        out = cv2.resize(img.astype(np.float32, copy=False), size,
                         interpolation=_METHODS[self.method])
        if frame.domain is Domain.DISPLAY:
            out = np.clip(out, 0.0, 1.0)
        frame.image = out.astype(np.float32, copy=False)
        frame.meta["upscale.size"] = f"{size[0]}x{size[1]}"
        return frame
