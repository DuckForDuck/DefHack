"""Debug module: save or show frames after any step, without changing them."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np

from ..frame import Domain, FrameData
from ..stage import Stage


def to_preview_uint8(img: np.ndarray, domain: Domain,
                     percentiles: tuple[float, float] = (0.5, 99.5)) -> np.ndarray:
    """Make a viewable 8-bit copy of an image in any domain.

    - OUTPUT (uint8): returned as is.
    - DISPLAY (float, nominal 0..1): scaled by 255.
    - RAW / LINEAR or anything else: stretched between the given
      percentiles of this image. This stretch is only for looking at the
      data; it is not part of the processing.
    """
    if img.dtype == np.uint8:
        return img
    if domain is Domain.DISPLAY:
        return np.clip(np.round(img * 255.0), 0, 255).astype(np.uint8)
    x = img.astype(np.float32)
    lo, hi = np.percentile(x, percentiles)
    span = max(float(hi - lo), 1e-6)
    return np.clip(np.round((x - lo) / span * 255.0), 0, 255).astype(np.uint8)


class Preview(Stage):
    """Save and/or show selected frames. The frame passes through unchanged.

    Place it after any step to see what that step did, for example::

        pipe.insert_after("Decode", Preview(tag="00_decode", frames=[0, 100]))

    Parameters
    ----------
    tag : name of the sub-folder and of the window, e.g. "02_destripe"
    frames : frame indices to use; ``None`` = all frames
    source : what to look at: "image" (default), or the name of an entry in
        ``frame.layers`` or ``frame.masks`` (e.g. "detail", "bad_pixels")
    save : write PNG files to ``out_dir/tag/frame_00100.png``
    save_npy : also write the unscaled array as .npy (for analysis)
    show : show the frames in an OpenCV window
    delay_ms : wait time per shown frame; 0 = wait for a key.
        Keys in the window: space – pause/continue, q – stop showing.
    percentiles : stretch used to view LINEAR data (see ``to_preview_uint8``)
    overlay : draw the frame index on the shown image (never on saved files)
    """

    name = "Preview"
    run_when_frozen = True   # also shows / writes held (shutter) frames

    def __init__(self, tag: str = "preview", frames: Iterable[int] | None = None,
                 source: str = "image", save: bool = True, save_npy: bool = False,
                 show: bool = False, delay_ms: int = 40,
                 percentiles: tuple[float, float] = (0.5, 99.5),
                 out_dir: str | Path = "debug", overlay: bool = True) -> None:
        super().__init__(name=f"Preview[{tag}]")
        self.tag = tag
        self.frames = None if frames is None else set(frames)
        self.source = source
        self.save = save
        self.save_npy = save_npy
        self.show = show
        self.delay_ms = delay_ms
        self.percentiles = percentiles
        self.out_dir = Path(out_dir)
        self.overlay = overlay
        self._paused = False
        self._stopped = False
        self._window_open = False
        self._cv2 = None
        if show:
            import cv2  # imported only when needed
            self._cv2 = cv2

    def _select(self, frame: FrameData) -> tuple[np.ndarray, Domain]:
        if self.source == "image":
            return frame.image, frame.domain
        if self.source in frame.layers:
            return frame.layers[self.source], frame.domain
        if self.source in frame.masks:
            m = frame.masks[self.source]
            return (m > 0).astype(np.uint8) * 255, Domain.OUTPUT
        raise KeyError(f"frame {frame.index}: no image, layer or mask "
                       f"named '{self.source}'")

    def process(self, frame: FrameData) -> FrameData:
        if self.frames is not None and frame.index not in self.frames:
            return frame
        data, domain = self._select(frame)
        view = to_preview_uint8(data, domain, self.percentiles)

        if self.save or self.save_npy:
            folder = self.out_dir / self.tag
            folder.mkdir(parents=True, exist_ok=True)
            stem = folder / f"frame_{frame.index:05d}"
            if self.save:
                from PIL import Image
                Image.fromarray(view).save(stem.with_suffix(".png"))
            if self.save_npy:
                np.save(stem.with_suffix(".npy"), data)

        if self.show and not self._stopped:
            self._show(view, frame.index)
        return frame

    def _show(self, view: np.ndarray, index: int) -> None:
        cv2 = self._cv2
        if self.overlay:
            view = view.copy()
            cv2.putText(view, f"{self.tag}  #{index}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, 255, 2)
        cv2.imshow(self.tag, view)
        self._window_open = True
        while True:
            key = cv2.waitKey(0 if self._paused else self.delay_ms) & 0xFF
            if key == ord("q"):
                self._stopped = True
                self._close()
                return
            if key == ord(" "):
                self._paused = not self._paused
                if self._paused:
                    continue
            return

    def _close(self) -> None:
        if self._window_open:
            self._cv2.destroyWindow(self.tag)
            self._cv2.waitKey(1)
            self._window_open = False

    def reset(self) -> None:
        self._paused = False
        self._stopped = False

    def flush(self) -> list[FrameData]:
        self._close()
        return []
