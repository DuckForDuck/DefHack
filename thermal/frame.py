"""Shared data container that travels through the pipeline.

Every module receives a ``FrameData`` object and returns it (changed or new).
Different algorithms need different data, so the container has a fixed core
and three open dictionaries:

- ``image``   the main working signal (float32, H x W). Most modules read and
              write only this field.
- ``layers``  additional images of the same frame, e.g. ``"base"`` and
              ``"detail"`` (step 5) or ``"noise_sigma_map"``.
- ``masks``   boolean or uint8 maps, e.g. ``"bad_pixels"`` (step 1),
              ``"motion"`` (step 3), ``"saturated"`` (step 9).
- ``meta``    scalars and small objects, e.g. ``"noise_sigma"``,
              ``"agc.lo"``, ``"agc.hi"``, ``"motion.shift"``.

Key naming rule: ``"<module>.<value>"`` for anything a module adds to
``meta`` (for example ``"decode.overflow_pixels"``), so modules of different
team members do not overwrite each other. Shared, well-known keys without a
prefix are listed in ``KNOWN_KEYS``.

Temporal modules (denoising, NUC, AGC smoothing) keep their history inside
the module object, not in ``FrameData``. This keeps the container small and
lets each module decide how many past frames it needs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np


class Domain(str, Enum):
    """Which kind of values ``FrameData.image`` holds right now.

    The physical order of processing (see docs) depends on this: radiometric
    corrections are valid only in the LINEAR domain, display operations only
    after tone mapping.
    """

    RAW = "raw"          # undecoded integers from the file
    LINEAR = "linear"    # float32, proportional to the sensor signal (DN)
    DISPLAY = "display"  # float32, tone-mapped, nominal range 0..1
    OUTPUT = "output"    # uint8, final 8-bit frame


# Shared keys that several modules are expected to read or write.
KNOWN_KEYS = {
    "meta": {
        "noise_sigma": "float, std of random noise of `image` in its current units",
        "bit_depth": "int, number of valid bits of the raw signal",
        "fps": "float, frame rate of the recording",
    },
    "layers": {
        "base": "float32, edge-preserving low-pass part of the image (step 5)",
        "detail": "float32, image - base (step 5)",
        "noise_sigma_map": "float32, remaining noise std per pixel, DN (step 3)",
    },
    "masks": {
        "bad_pixels": "bool, True where the sensor pixel is invalid (step 1)",
    },
}


@dataclass
class FrameData:
    """One frame and everything the modules know about it."""

    index: int
    raw: np.ndarray | None = None
    image: np.ndarray | None = None
    domain: Domain = Domain.RAW
    timestamp: float | None = None
    layers: dict[str, np.ndarray] = field(default_factory=dict)
    masks: dict[str, np.ndarray] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def shape(self) -> tuple[int, int]:
        src = self.image if self.image is not None else self.raw
        if src is None:
            raise ValueError(f"frame {self.index} has no pixel data")
        return src.shape[:2]

    def copy(self) -> "FrameData":
        """Deep copy of all arrays; use it when a module must keep a frame."""
        return FrameData(
            index=self.index,
            raw=None if self.raw is None else self.raw.copy(),
            image=None if self.image is None else self.image.copy(),
            domain=self.domain,
            timestamp=self.timestamp,
            layers={k: v.copy() for k, v in self.layers.items()},
            masks={k: v.copy() for k, v in self.masks.items()},
            meta=dict(self.meta),
        )

    def require(self, domain: Domain) -> None:
        """Raise if the frame is not in the domain a module expects."""
        if self.domain is not domain:
            raise ValueError(
                f"frame {self.index}: expected domain '{domain.value}', "
                f"got '{self.domain.value}'"
            )
