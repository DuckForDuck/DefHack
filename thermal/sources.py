"""Frame sources: objects that produce ``FrameData`` from a recording."""

from __future__ import annotations

import os
import warnings
from typing import Iterator

import numpy as np

from .frame import Domain, FrameData


class RawFileSource:
    """Read frames from a headerless raw file of fixed-size frames.

    The file is memory-mapped, so only the frames that are used are read
    from disk. Each frame is returned as ``FrameData`` with ``raw`` set to a
    read-only uint16 array and ``domain = RAW``. Decoding is done by the
    ``Decode`` module, so that a partner can replace either part.

    Parameters
    ----------
    path : path to the .bin file
    width, height : frame size in pixels
    dtype : numpy dtype string of one pixel, default little-endian uint16
    header_bytes : bytes to skip at the start of the file
    frame_header_bytes : bytes to skip before every frame (per-frame header)
    start, stop, step : which frames to read (like ``range``)
    fps : frame rate, used for ``FrameData.timestamp`` and ``meta["fps"]``
    """

    def __init__(self, path: str, width: int = 640, height: int = 512,
                 dtype: str = "<u2", header_bytes: int = 0,
                 frame_header_bytes: int = 0, start: int = 0,
                 stop: int | None = None, step: int = 1,
                 fps: float | None = None) -> None:
        self.path = path
        self.width = width
        self.height = height
        self.dtype = np.dtype(dtype)
        self.header_bytes = header_bytes
        self.frame_header_bytes = frame_header_bytes
        self.start, self.stop, self.step = start, stop, step
        self.fps = fps

        self.pixel_bytes = width * height * self.dtype.itemsize
        self.frame_bytes = self.pixel_bytes + frame_header_bytes
        payload = os.path.getsize(path) - header_bytes
        self.total_frames = payload // self.frame_bytes
        rest = payload - self.total_frames * self.frame_bytes
        if rest:
            warnings.warn(f"{path}: {rest} trailing bytes ignored "
                          f"(not a whole frame)")
        if self.total_frames == 0:
            raise ValueError(f"{path}: smaller than one frame")

        self._mm = np.memmap(path, dtype=np.uint8, mode="r",
                             offset=header_bytes,
                             shape=(self.total_frames * self.frame_bytes,))

    @property
    def indices(self) -> range:
        return range(*slice(self.start, self.stop, self.step)
                     .indices(self.total_frames))

    def __len__(self) -> int:
        return len(self.indices)

    def read(self, index: int) -> FrameData:
        """Read one frame by its absolute index in the file."""
        if not 0 <= index < self.total_frames:
            raise IndexError(f"frame {index} out of range 0..{self.total_frames - 1}")
        offset = index * self.frame_bytes + self.frame_header_bytes
        buf = self._mm[offset: offset + self.pixel_bytes]
        raw = buf.view(self.dtype).reshape(self.height, self.width)
        meta = {"source.path": self.path, "source.total_frames": self.total_frames}
        if self.fps:
            meta["fps"] = self.fps
        return FrameData(
            index=index,
            raw=raw,  # read-only view into the memory map
            domain=Domain.RAW,
            timestamp=index / self.fps if self.fps else None,
            meta=meta,
        )

    def __iter__(self) -> Iterator[FrameData]:
        for i in self.indices:
            yield self.read(i)

    def __repr__(self) -> str:
        return (f"RawFileSource({self.path!r}, {self.width}x{self.height}, "
                f"{self.dtype.str}, frames={len(self)}/{self.total_frames})")
