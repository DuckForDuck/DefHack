"""Side-by-side comparison of the pipeline with a reference, frame by frame."""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Iterable

import numpy as np

from ..frame import Domain, FrameData
from ..pipeline import Pipeline, StopPipeline
from ..stage import Stage, StageLike
from .decode import Decode
from .naive import NaiveAGC
from .preview import to_preview_uint8

HELP = "space play/pause | d next | a back | s save | q quit"


class Compare(Stage):
    """Show the current frame next to a reference result in one window.

    Left: the frame as it is at this point of the pipeline.
    Right: the same raw frame processed by ``reference`` (default: the naive
    processing of ``script.py``). The frame itself passes through unchanged.

    The reference gets a fresh copy of the *raw* frame, so ``Decode`` must
    keep it (``keep_raw=True``, the default).

    Keys in the window:
        space – play / pause        d – next frame       a – previous frame
        s – save the current view   q – stop the whole run

    Going back (a) uses the last ``history`` already shown views, so it does
    not run the pipeline again and works with stateful (temporal) modules.

    Parameters
    ----------
    reference : module or Pipeline that turns a RAW ``FrameData`` into an
        image. Default ``Pipeline([Decode(14), NaiveAGC(2, 98)])``.
    label, reference_label : captions above the two images
    tag : window name and sub-folder for saved views
    frames : frame indices to show/save; ``None`` = all frames
    show : open the interactive window
    save : save every view to ``out_dir/tag/frame_NNNNN.png``
        (key s saves the current view also when ``save`` is False)
    start_paused : wait for a key on the first frame
    delay_ms : time per frame while playing
    history : number of views kept for going back
    scale : display size factor of the window (saved files are 1:1)
    max_width : the window is made smaller if it would be wider (px)
    """

    name = "Compare"
    run_when_frozen = True   # also shows / writes held (shutter) frames

    def __init__(self, reference: StageLike | None = None,
                 label: str = "pipeline", reference_label: str = "naive (script.py)",
                 tag: str = "compare", frames: Iterable[int] | None = None,
                 show: bool = True, save: bool = False, start_paused: bool = True,
                 delay_ms: int = 40, history: int = 300, scale: float = 1.0,
                 max_width: int = 1600,
                 out_dir: str | Path = "debug",
                 percentiles: tuple[float, float] = (0.5, 99.5)) -> None:
        super().__init__(name=f"Compare[{tag}]")
        self.reference = reference if reference is not None else Pipeline(
            [Decode(bit_depth=14, stats=False), NaiveAGC(low=2, high=98)],
            profile=False)
        self.label = label
        self.reference_label = reference_label
        self.tag = tag
        self.frames = None if frames is None else set(frames)
        self.show = show
        self.save = save
        self.start_paused = start_paused
        self.delay_ms = delay_ms
        self.history = history
        self.scale = scale
        self.max_width = max_width
        self.out_dir = Path(out_dir)
        self.percentiles = percentiles

        self._cache: deque[tuple[int, np.ndarray]] = deque(maxlen=history)
        self._paused = start_paused
        self._window_open = False
        self._cv2 = None
        import cv2  # needed for text drawing in both show and save modes
        self._cv2 = cv2

    # ----- building the view ---------------------------------------------
    def _reference_image(self, frame: FrameData) -> tuple[np.ndarray, Domain]:
        if frame.raw is None:
            raise ValueError("Compare needs frame.raw; use Decode(keep_raw=True)")
        ref = FrameData(index=frame.index, raw=frame.raw, domain=Domain.RAW,
                        timestamp=frame.timestamp)
        out = self.reference.process(ref)
        if out is None:
            raise ValueError("reference returned no frame (buffering is not "
                             "supported in Compare)")
        return out.image, out.domain

    @staticmethod
    def _stats(img8: np.ndarray) -> str:
        return (f"black {100 * np.mean(img8 == 0):.2f}%  "
                f"white {100 * np.mean(img8 == 255):.2f}%")

    def _panel(self, img8: np.ndarray, title: str) -> np.ndarray:
        cv2 = self._cv2
        h, w = img8.shape
        head = np.full((44, w), 32, np.uint8)
        cv2.putText(head, title, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255, 1,
                    cv2.LINE_AA)
        cv2.putText(head, self._stats(img8), (8, 37), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, 200, 1, cv2.LINE_AA)
        return np.vstack([head, img8])

    def _compose(self, frame: FrameData) -> np.ndarray:
        cv2 = self._cv2
        left = to_preview_uint8(frame.image, frame.domain, self.percentiles)
        ref_img, ref_dom = self._reference_image(frame)
        right = to_preview_uint8(ref_img, ref_dom, self.percentiles)
        if right.shape != left.shape:  # e.g. the pipeline already upscaled
            right = cv2.resize(right, left.shape[::-1],
                               interpolation=cv2.INTER_NEAREST)

        left_title = self.label
        if frame.domain in (Domain.RAW, Domain.LINEAR):
            left_title += f"  [{frame.domain.value}, view stretch p{self.percentiles[0]}-p{self.percentiles[1]}]"
        sep = np.full((left.shape[0] + 44, 6), 255, np.uint8)
        body = np.hstack([self._panel(left, left_title), sep,
                          self._panel(right, self.reference_label)])
        foot = np.full((26, body.shape[1]), 32, np.uint8)
        cv2.putText(foot, f"frame #{frame.index}    {HELP}", (8, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255, 1, cv2.LINE_AA)
        return np.vstack([body, foot])

    # ----- output ----------------------------------------------------------
    def _save(self, index: int, view: np.ndarray) -> Path:
        from PIL import Image
        folder = self.out_dir / self.tag
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"frame_{index:05d}.png"
        Image.fromarray(view).save(path)
        return path

    def _imshow(self, view: np.ndarray) -> None:
        cv2 = self._cv2
        scale = min(self.scale, self.max_width / view.shape[1])
        if scale != 1.0:
            view = cv2.resize(view, None, fx=scale, fy=scale,
                              interpolation=cv2.INTER_AREA if scale < 1
                              else cv2.INTER_NEAREST)
        cv2.imshow(self.tag, view)
        self._window_open = True

    def _wait_key(self, ms: int) -> int:
        return self._cv2.waitKey(ms) & 0xFF

    def _interact(self) -> None:
        """Show views; return when the pipeline should process the next frame."""
        pos = len(self._cache) - 1
        while True:
            index, view = self._cache[pos]
            self._imshow(view)
            latest = pos == len(self._cache) - 1
            key = self._wait_key(0 if self._paused else self.delay_ms)
            if key == ord("q"):
                self._close()
                raise StopPipeline("stopped by user in Compare window")
            if key == ord(" "):
                self._paused = not self._paused
                continue
            if key == ord("s"):
                print(f"[{self.name}] saved {self._save(index, view)}")
                continue
            if key == ord("a"):
                self._paused = True
                pos = max(0, pos - 1)
                continue
            if key == ord("d"):
                if latest:
                    return
                pos += 1
                continue
            if not self._paused:  # playing: timeout or other key
                if latest:
                    return
                pos += 1

    def process(self, frame: FrameData) -> FrameData:
        if self.frames is not None and frame.index not in self.frames:
            return frame
        view = self._compose(frame)
        if self.save:
            self._save(frame.index, view)
        if self.show:
            self._cache.append((frame.index, view))
            self._interact()
        return frame

    def _close(self) -> None:
        if self._window_open:
            self._cv2.destroyWindow(self.tag)
            self._cv2.waitKey(1)
            self._window_open = False

    def reset(self) -> None:
        self._cache.clear()
        self._paused = self.start_paused
        if callable(getattr(self.reference, "reset", None)):
            self.reference.reset()

    def flush(self) -> list[FrameData]:
        # End of stream: let the user still browse the last views.
        if self.show and self._cache and self._window_open:
            self._paused = True
            try:
                while True:
                    self._interact()  # d on the last frame returns
                    print(f"[{self.name}] end of recording; press q to close")
            except StopPipeline:
                pass
        self._close()
        return []
