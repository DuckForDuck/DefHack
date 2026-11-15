"""Chain of modules that processes a stream of frames."""

from __future__ import annotations

import time
from collections import defaultdict
from typing import Iterable, Iterator

from .frame import FrameData
from .stage import StageLike


class StopPipeline(Exception):
    """Raise from a stage to stop the run early (e.g. user pressed q).

    Frames already held by stages are still flushed, so writers can close
    their files properly.
    """


def _stage_name(stage: StageLike) -> str:
    return getattr(stage, "name", "") or type(stage).__name__


def stage_runs(stage: StageLike, frame: FrameData) -> bool:
    """True if ``stage`` must process ``frame``.

    A stage is skipped when it is switched off (``enabled = False``) or when
    the frame is *frozen* (``meta["frozen"]``, set by ``ShutterGuard`` for a
    shutter / FFC frame) and the stage does not declare
    ``run_when_frozen = True``. So stateful modules (NUC, denoising, AGC) do
    not learn from a flat shutter frame; ``ShutterHold`` and the output
    modules still run and show the last good image.
    """
    if not getattr(stage, "enabled", True):
        return False
    if frame.meta.get("frozen") and not getattr(stage, "run_when_frozen", False):
        return False
    return True


class Pipeline:
    """Ordered list of modules.

    Example::

        pipe = Pipeline([Decode(bit_depth=14), BadPixels(...), ...])
        pipe.insert_after("Decode", PartnerModule(param=1))
        for frame in pipe.run(source):
            ...
    """

    def __init__(self, stages: Iterable[StageLike] = (), profile: bool = True) -> None:
        self.stages: list[StageLike] = []
        self.profile = profile
        self.timings: dict[str, float] = defaultdict(float)
        for s in stages:
            self.add(s)

    # ----- building -------------------------------------------------------
    @staticmethod
    def _check(stage: StageLike) -> StageLike:
        if not callable(getattr(stage, "process", None)):
            raise TypeError(
                f"{stage!r} has no process(frame) method; "
                "wrap plain array functions with ArrayStage"
            )
        return stage

    def add(self, stage: StageLike) -> "Pipeline":
        self.stages.append(self._check(stage))
        return self

    def index_of(self, name: str) -> int:
        for i, s in enumerate(self.stages):
            if _stage_name(s) == name:
                return i
        raise KeyError(f"no stage named '{name}' in {self.names}")

    def insert_after(self, name: str, stage: StageLike) -> "Pipeline":
        self.stages.insert(self.index_of(name) + 1, self._check(stage))
        return self

    def insert_before(self, name: str, stage: StageLike) -> "Pipeline":
        self.stages.insert(self.index_of(name), self._check(stage))
        return self

    def replace(self, name: str, stage: StageLike) -> "Pipeline":
        self.stages[self.index_of(name)] = self._check(stage)
        return self

    def remove(self, name: str) -> "Pipeline":
        del self.stages[self.index_of(name)]
        return self

    @property
    def names(self) -> list[str]:
        return [_stage_name(s) for s in self.stages]

    # ----- running --------------------------------------------------------
    def reset(self) -> None:
        self.timings.clear()
        for s in self.stages:
            if callable(getattr(s, "reset", None)):
                s.reset()

    def _run_from(self, start: int, frame: FrameData) -> FrameData | None:
        for stage in self.stages[start:]:
            if not stage_runs(stage, frame):  # switched off / frozen frame
                continue
            t0 = time.perf_counter()
            frame = stage.process(frame)
            if self.profile:
                self.timings[_stage_name(stage)] += time.perf_counter() - t0
            if frame is None:  # stage is buffering
                return None
        return frame

    def process(self, frame: FrameData) -> FrameData | None:
        """Run one frame through all stages (no flushing)."""
        return self._run_from(0, frame)

    def run(self, source: Iterable[FrameData]) -> Iterator[FrameData]:
        """Process a whole stream, including frames held back by stages."""
        self.reset()
        try:
            for frame in source:
                out = self._run_from(0, frame)
                if out is not None:
                    yield out
        except StopPipeline as e:
            print(f"[Pipeline] {e}")
        # Flush buffered frames stage by stage, in order, so that frames held
        # by stage i still pass through stages i+1 .. n.
        for i, stage in enumerate(self.stages):
            flush = getattr(stage, "flush", None)
            if not callable(flush):
                continue
            for frame in flush():
                try:
                    out = self._run_from(i + 1, frame)
                except StopPipeline:
                    continue
                if out is not None:
                    yield out

    def timing_report(self) -> str:
        total = sum(self.timings.values()) or 1.0
        lines = [f"{n:<24} {t:8.3f} s  {100 * t / total:5.1f} %"
                 for n, t in self.timings.items()]
        return "\n".join(lines)

    def __repr__(self) -> str:
        inner = ",\n  ".join(repr(s) for s in self.stages)
        return f"Pipeline([\n  {inner}\n])"
