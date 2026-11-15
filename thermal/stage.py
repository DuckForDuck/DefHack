"""Base class for all processing modules.

A module is an object with one main method::

    frame_out = module.process(frame_in)

Rules for every module:

1. All parameters go to ``__init__``. ``process`` takes only the frame.
2. ``process`` returns a ``FrameData``, or ``None`` if the module needs more
   frames before it can output one (for example a filter that uses future
   frames). Frames held back this way are returned later by ``flush``.
3. State between frames (history, running averages) lives in the object.
   ``reset`` clears it, so the same object can process a new recording.
4. A module checks ``frame.domain`` if it relies on it and sets the new
   domain if it changes it.
5. ``enabled = False`` switches a module off: the pipeline skips it
   (used by the tuner to compare with / without an algorithm).

Partner modules do not have to inherit from ``Stage``. Any object with a
``process(frame) -> FrameData | None`` method works in the pipeline.
For a plain function on a numpy array use ``ArrayStage``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, Protocol, runtime_checkable

import numpy as np

from .frame import Domain, FrameData


@runtime_checkable
class StageLike(Protocol):
    """What the pipeline accepts: anything with ``process``."""

    def process(self, frame: FrameData) -> FrameData | None: ...


class Stage(ABC):
    """Convenience base class with default ``reset``, ``flush`` and naming."""

    name: str = ""
    enabled: bool = True   # False = the pipeline skips this module

    def __init__(self, name: str | None = None) -> None:
        self.name = name or self.name or type(self).__name__

    @abstractmethod
    def process(self, frame: FrameData) -> FrameData | None:
        """Process one frame."""

    def reset(self) -> None:
        """Clear state kept between frames. Override if the module has state."""

    def flush(self) -> list[FrameData]:
        """Return frames still held back at the end of the stream."""
        return []

    def __call__(self, frame: FrameData) -> FrameData | None:
        return self.process(frame)

    def __repr__(self) -> str:
        params = {k: v for k, v in vars(self).items()
                  if not k.startswith("_") and k != "name"}
        args = ", ".join(f"{k}={v!r}" for k, v in params.items())
        return f"{type(self).__name__}({args})"


class ArrayStage(Stage):
    """Wrap a function ``f(image: np.ndarray) -> np.ndarray`` as a module.

    Example::

        pipeline.add(ArrayStage(lambda img: cv2.GaussianBlur(img, (3, 3), 0),
                                name="blur"))
    """

    def __init__(self, func: Callable[[np.ndarray], np.ndarray],
                 name: str | None = None,
                 required_domain: Domain | None = None,
                 output_domain: Domain | None = None) -> None:
        super().__init__(name or getattr(func, "__name__", "ArrayStage"))
        self.func = func
        self.required_domain = required_domain
        self.output_domain = output_domain

    def process(self, frame: FrameData) -> FrameData:
        if self.required_domain is not None:
            frame.require(self.required_domain)
        frame.image = self.func(frame.image)
        if self.output_domain is not None:
            frame.domain = self.output_domain
        return frame
