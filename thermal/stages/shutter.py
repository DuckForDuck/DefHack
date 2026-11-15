"""Shutter / FFC frames: detect them and show the last good image instead.

Idea taken from Michael's ``thermal_isp`` (stage 0, "shutter guard").
"""

from __future__ import annotations

from collections import deque

import numpy as np

from ..frame import Domain, FrameData
from ..stage import Stage


class ShutterGuard(Stage):
    """Mark shutter / FFC (flat-field correction) frames as *frozen*.

    During an FFC the camera looks at its closed shutter: the frame is almost
    flat (spatial std far below normal). Such a frame must not be shown and
    must not update the state of later modules (NUC offset, temporal
    denoising history, AGC range), otherwise the image flashes grey and the
    denoiser smears the flat frame into the next ones.

    Test: spatial std of the frame (every 4th pixel) <
    ``ratio`` x median std of the last ``history`` normal frames
    (needs at least 4 of them). At most ``max_freeze`` frames in a row are
    frozen; after that the change is accepted as a real scene change
    (e.g. a uniform sky) and the history restarts.

    A frozen frame gets ``meta["frozen"] = True``. The pipeline then skips
    every module that does not declare ``run_when_frozen = True`` – only
    ``ShutterHold`` (which shows the last good output) and the output
    modules (Upscale, Quantise, VideoWriter, Preview, Compare) run.

    Uses only the current frame and statistics of past frames (streaming).
    Put it right after ``Decode``.

    Parameters
    ----------
    ratio : std ratio below which a frame counts as flat (Michael: 0.35)
    max_freeze : max. frames in a row that are frozen
    history : number of past normal frames for the reference std
    """

    name = "ShutterGuard"

    def __init__(self, ratio: float = 0.35, max_freeze: int = 30,
                 history: int = 16, enabled: bool = True) -> None:
        super().__init__()
        self.ratio = ratio
        self.max_freeze = max_freeze
        self.history = history
        self.enabled = enabled
        self.reset()

    def reset(self) -> None:
        self._stds: deque = deque(maxlen=int(self.history))
        self._freeze = 0

    def process(self, frame: FrameData) -> FrameData:
        frame.require(Domain.LINEAR)
        s = float(frame.image[::4, ::4].std())
        frozen = False
        if len(self._stds) >= 4 and s < self.ratio * float(np.median(self._stds)):
            self._freeze += 1
            if self._freeze <= self.max_freeze:
                frozen = True
            else:
                self._stds.clear()  # the scene really changed: accept it
        if not frozen:
            self._freeze = 0
            self._stds.append(s)
        frame.meta["frozen"] = frozen
        frame.meta["shutter.std"] = s
        return frame


class ShutterHold(Stage):
    """Show the last good display image for frozen (shutter) frames.

    Normal frame: keeps a copy of the display image (``Domain.DISPLAY``).
    Frozen frame (``meta["frozen"]``): replaces the image with that copy.
    If there is no copy yet (first frames, or the interactive tuner after a
    jump), a mid-grey frame is shown.

    Put it at the end of the display steps (after detail enhancement,
    before Upscale / Quantise).
    """

    name = "ShutterHold"
    run_when_frozen = True

    def __init__(self, enabled: bool = True) -> None:
        super().__init__()
        self.enabled = enabled
        self.reset()

    def reset(self) -> None:
        self._last: np.ndarray | None = None

    def warmup(self, frame: FrameData) -> FrameData:
        return frame

    def process(self, frame: FrameData) -> FrameData:
        if frame.meta.get("frozen"):
            if self._last is not None:
                frame.image = self._last.copy()
            else:
                frame.image = np.full(frame.shape, 0.5, np.float32)
            frame.domain = Domain.DISPLAY
            frame.meta["shutter.held"] = True
            return frame
        frame.require(Domain.DISPLAY)
        self._last = frame.image.astype(np.float32, copy=True)
        frame.meta["shutter.held"] = False
        return frame
