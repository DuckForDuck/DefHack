"""Step 0 – decode: mask to the valid bit depth and convert to float32."""

from __future__ import annotations

import numpy as np

from ..frame import Domain, FrameData
from ..stage import Stage


class Decode(Stage):
    """Turn the raw integer frame into the linear float32 working signal.

    Actions:
    1. Count pixels that use bits above ``bit_depth`` (status/flag bits or a
       wrong data format) and store the count in
       ``meta["decode.overflow_pixels"]``.
    2. Keep only the lower ``bit_depth`` bits (``raw & (2**bit_depth - 1)``).
    3. Convert to float32 and store as ``image``; set ``domain = LINEAR``.
    4. Optionally record simple frame statistics for later modules and checks.

    Parameters
    ----------
    bit_depth : number of valid bits in the raw signal (14 for this sensor)
    on_overflow : what to do if pixels have bits above ``bit_depth``:
        "mask" – mask them silently (default; the count is kept in meta),
        "warn" – mask them and print a warning,
        "error" – raise ``ValueError``.
    keep_raw : keep the original uint16 array in ``frame.raw``. Set to False
        to save memory when no later module needs it.
    stats : store min / max / mean of the decoded frame in ``meta``.
    """

    name = "Decode"

    def __init__(self, bit_depth: int = 14, on_overflow: str = "mask",
                 keep_raw: bool = True, stats: bool = True) -> None:
        super().__init__()
        if not 1 <= bit_depth <= 16:
            raise ValueError("bit_depth must be in 1..16")
        if on_overflow not in ("mask", "warn", "error"):
            raise ValueError("on_overflow must be 'mask', 'warn' or 'error'")
        self.bit_depth = bit_depth
        self.on_overflow = on_overflow
        self.keep_raw = keep_raw
        self.stats = stats
        self._mask = np.uint16((1 << bit_depth) - 1)

    def process(self, frame: FrameData) -> FrameData:
        frame.require(Domain.RAW)
        raw = frame.raw
        if raw is None:
            raise ValueError(f"frame {frame.index}: no raw data to decode")
        if raw.dtype.kind != "u":
            raise TypeError(f"frame {frame.index}: raw must be unsigned, got {raw.dtype}")

        masked = np.bitwise_and(raw, self._mask, dtype=np.uint16)
        overflow = int(np.count_nonzero(masked != raw))
        if overflow:
            msg = (f"frame {frame.index}: {overflow} pixels use bits above "
                   f"bit {self.bit_depth - 1}")
            if self.on_overflow == "error":
                raise ValueError(msg)
            if self.on_overflow == "warn":
                print(f"[{self.name}] warning: {msg}; masked")

        image = masked.astype(np.float32)
        frame.image = image
        frame.domain = Domain.LINEAR
        frame.meta["bit_depth"] = self.bit_depth
        frame.meta["decode.overflow_pixels"] = overflow
        if self.stats:
            frame.meta["decode.min"] = float(image.min())
            frame.meta["decode.max"] = float(image.max())
            frame.meta["decode.mean"] = float(image.mean())
        if not self.keep_raw:
            frame.raw = None
        return frame
