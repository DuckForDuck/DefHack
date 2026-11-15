"""Speed check: how many frames per second the pipeline processes.

    python bench.py                                  # preset michael-features, all frames
    python bench.py --params recommended --frames 100
    python bench.py --fps 25                         # also check a 25 fps camera

Frames are fed one by one, as from a camera (streaming). Measured: the time
of steps 0–9 per frame (reading the file and writing video are NOT counted:
in a camera both are done by other hardware). The NUC is not calibrated in
advance (``nuc_fit_step = 0``), as in a real stream.

Output: per-frame time (mean, p50, p95, max), the frame rate it allows, the
share of each module, and the result for the camera frame rate ``--fps``.
"""

from __future__ import annotations

import argparse
import time
import types
from pathlib import Path

import numpy as np

import main as M
from thermal import FrameData, RawFileSource
from thermal.pipeline import stage_runs
from thermal.stages import Quantise
from thermal.tuning import apply_params, load_params, resolve_preset


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("input", nargs="?", default=str(M.DEFAULT_INPUT))
    ap.add_argument("--params", default="michael-features")
    ap.add_argument("--frames", type=int, default=0, help="0 = all")
    ap.add_argument("--fps", type=float, default=25.0, help="camera frame rate to check")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=512)
    a = ap.parse_args()

    args = types.SimpleNamespace(input=a.input, width=a.width, height=a.height,
                                 nuc_fit_step=0)
    stages = M.build_front(args) + M.build_back() + [Quantise(dither="blue")]
    if a.params:
        apply_params(stages, load_params(resolve_preset(M.PRESETS_DIR, a.params)))
    src = RawFileSource(a.input, width=a.width, height=a.height,
                        stop=a.frames or None)

    per_stage: dict[str, float] = {}
    times = []
    for s in stages:
        if callable(getattr(s, "reset", None)):
            s.reset()
    for frame in src:
        frame.raw = np.ascontiguousarray(frame.raw)   # file read is not timed
        t0 = time.perf_counter()
        f: FrameData | None = frame
        for s in stages:
            if f is None or not stage_runs(s, f):
                continue
            ts = time.perf_counter()
            f = s.process(f)
            per_stage[s.name] = per_stage.get(s.name, 0.0) + time.perf_counter() - ts
        times.append(time.perf_counter() - t0)

    t = np.array(times[1:]) * 1000.0          # first frame = warm-up, not counted
    n = len(times)
    budget = 1000.0 / a.fps
    print(f"preset: {a.params}   frames: {n}   file: {Path(a.input).name}")
    print(f"time per frame, ms: mean {t.mean():.1f}  p50 {np.percentile(t, 50):.1f}  "
          f"p95 {np.percentile(t, 95):.1f}  max {t.max():.1f}")
    print(f"=> {1000.0 / t.mean():.1f} frames per second on average "
          f"({1000.0 / np.percentile(t, 95):.1f} fps in the slowest 5 % of frames)")
    over = int((t > budget).sum())
    print(f"camera at {a.fps:g} fps: budget {budget:.1f} ms/frame, "
          f"{over} of {len(t)} frames over budget -> "
          f"{'keeps up' if t.mean() < 0.9 * budget and over <= 0.01 * len(t) else 'does NOT keep up'}")
    total = sum(per_stage.values()) or 1.0
    print("\nper module (ms per frame, share):")
    for name, s in sorted(per_stage.items(), key=lambda kv: -kv[1]):
        print(f"  {name:<18} {1000.0 * s / n:7.1f} ms  {100.0 * s / total:5.1f} %")


if __name__ == "__main__":
    main()
