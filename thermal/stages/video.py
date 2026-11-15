"""Step 10 – write the 8-bit frames to a video file with ffmpeg."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np

from ..frame import Domain, FrameData
from ..stage import Stage

# Output formats. Both keep the full grey range 0..255.
FORMATS = {
    # H.264 for viewing: plays in QuickTime, browsers, VLC, IINA.
    # Grey is stored as YUV with neutral colour; full range (0..255) is kept
    # and marked in the file (color_range = pc), so players do not
    # stretch or crush black and white.
    # tune "grain" keeps fine texture and the blue-noise dither of step 9
    # (default x264 tuning smooths it away); measured on the 640x512 output:
    # crf 14 → PSNR 45.9 dB, crf 10 + grain → 49.4 dB vs the lossless file.
    "h264": lambda crf, preset, tune: [
        *(["-tune", tune] if tune else []),
        "-vf", "scale=in_range=full:out_range=full,format=yuv420p",
        "-c:v", "libx264", "-preset", preset, "-crf", str(crf),
        "-profile:v", "high", "-color_range", "pc",
        "-x264-params", "fullrange=on",
        "-movflags", "+faststart", "-tag:v", "avc1",
    ],
    # FFV1: lossless archive / analysis copy, exactly the 8-bit frames.
    "ffv1": lambda crf, preset, tune: [
        "-c:v", "ffv1", "-level", "3", "-g", "1", "-slicecrc", "1",
        "-pix_fmt", "gray", "-color_range", "pc",
    ],
}


class VideoWriter(Stage):
    """Send 8-bit grey frames to ffmpeg and write one video file.

    Use one ``VideoWriter`` per file (e.g. one for .mp4, one for .mkv).
    The frame passes through unchanged. The file is opened at the first
    frame (its size is taken from the frame) and closed in ``flush`` – also
    when the run is stopped early.

    Formats: ``"h264"`` (viewing, .mp4, full range 0..255) and ``"ffv1"``
    (lossless, .mkv). ffmpeg messages go to ``<file>.log``.

    Parameters
    ----------
    path : output file
    fmt : "h264" or "ffv1"
    fps : frame rate written into the file
    crf : quality for H.264 (lower = better, 14 ≈ visually lossless)
    preset : x264 speed / compression trade-off ("slow" = good compression)
    tune : x264 tuning ("grain" = keep fine texture and dither; None = off)
    """

    name = "VideoWriter"
    run_when_frozen = True   # also shows / writes held (shutter) frames

    def __init__(self, path: str | Path, fmt: str = "h264", fps: float = 25.0,
                 crf: int = 10, preset: str = "slow", tune: str | None = "grain",
                 ffmpeg: str = "ffmpeg") -> None:
        super().__init__(name=f"VideoWriter[{fmt}]")
        if fmt not in FORMATS:
            raise ValueError(f"fmt must be one of {list(FORMATS)}")
        self.path = Path(path)
        self.fmt = fmt
        self.fps = fps
        self.crf = crf
        self.preset = preset
        self.tune = tune
        self.ffmpeg = ffmpeg
        self._proc: subprocess.Popen | None = None
        self._log = None
        self._size: tuple[int, int] | None = None
        self.frames_written = 0

    def _open(self, w: int, h: int) -> None:
        exe = shutil.which(self.ffmpeg)
        if exe is None:
            raise RuntimeError("ffmpeg not found (install it, e.g. 'brew install ffmpeg')")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        cmd = [exe, "-y", "-hide_banner", "-loglevel", "warning",
               "-f", "rawvideo", "-pix_fmt", "gray", "-s", f"{w}x{h}",
               "-r", str(self.fps), "-color_range", "pc", "-i", "-",
               *FORMATS[self.fmt](self.crf, self.preset, self.tune), str(self.path)]
        self._log = open(self.path.with_suffix(self.path.suffix + ".log"), "w")
        self._proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                                      stdout=subprocess.DEVNULL, stderr=self._log)
        self._size = (w, h)
        self.frames_written = 0

    def process(self, frame: FrameData) -> FrameData:
        frame.require(Domain.OUTPUT)
        img = frame.image
        if img.dtype != np.uint8 or img.ndim != 2:
            raise TypeError("VideoWriter needs 2-D uint8 frames (run Quantise first)")
        h, w = img.shape
        if self._proc is None:
            self._open(w, h)
        elif (w, h) != self._size:
            raise ValueError(f"frame size changed from {self._size} to {(w, h)}")
        self._proc.stdin.write(np.ascontiguousarray(img).tobytes())
        self.frames_written += 1
        return frame

    def _close(self) -> None:
        if self._proc is None:
            return
        self._proc.stdin.close()
        rc = self._proc.wait()
        self._log.close()
        self._proc = None
        if rc != 0:
            raise RuntimeError(f"ffmpeg failed (code {rc}), see {self.path}.log")
        print(f"[{self.name}] wrote {self.frames_written} frames → {self.path}")

    def flush(self) -> list[FrameData]:
        self._close()
        return []

    def reset(self) -> None:
        self._close()


def write_range_test(path: str | Path, fps: float = 25.0, seconds: float = 4.0) -> Path:
    """Short H.264 test clip to check that a player shows the full range.

    Top: a smooth ramp 0..255. Bottom: patches with grey levels
    0, 8, 16, 24 (left) and 231, 239, 247, 255 (right). In a player that
    shows full range correctly, all four dark patches differ from each
    other and all four light patches differ from each other. If 0/8/16 look
    the same black, or 239/247/255 the same white, the player clips the range.
    """
    w, h = 1280, 720
    img = np.zeros((h, w), np.uint8)
    img[: h // 2] = np.linspace(0, 255, w).astype(np.uint8)[None, :]
    levels = [0, 8, 16, 24, 231, 239, 247, 255]
    pw = w // len(levels)
    for i, v in enumerate(levels):
        img[h // 2:, i * pw:(i + 1) * pw] = v
    writer = VideoWriter(path, fmt="h264", fps=fps, crf=1, preset="medium")
    for k in range(int(fps * seconds)):
        writer.process(FrameData(index=k, image=img, domain=Domain.OUTPUT))
    writer.flush()
    return Path(path)
