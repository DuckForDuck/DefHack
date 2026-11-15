"""Interactive tuning: re-render single frames with changed parameters.

The pipeline is split in two parts:

- **front** (steps 0–3): heavy and stateful (NUC, temporal denoising). It is
  run once over the whole recording; its output (linear image + noise map)
  is cached in memory.
- **back** (steps 5–7 and later display steps): the parts that are tuned
  by eye. They are re-run for the requested frame.

Some back modules keep state between frames (``ToneMap`` smooths its range
and histogram over time). Before a frame is rendered, the back modules are
"warmed up" on the ``warmup`` previous frames. A module may implement
``warmup(frame)`` to update only its state cheaply; otherwise ``process``
is used. When frames are requested in order (playback), the state simply
continues and no warm-up is needed.

Parameters are addressed as ``"<StageName>.<attribute>"`` and set with
``setattr`` on the module object, so modules of partners can be tuned too
(add a ``Param`` for them).
"""

from __future__ import annotations

import json
import re
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np

from .frame import Domain, FrameData
from .pipeline import Pipeline, stage_runs
from .stage import StageLike


@dataclass
class Param:
    """One tunable value of one module."""

    key: str            # "<StageName>.<attribute>", e.g. "ToneMap.gamma"
    min: float
    max: float
    step: float
    label: str
    help: str = ""
    group: str = "display"   # section in the tuner page
    integer: bool = False
    toggle: bool = False     # on/off switch (bool value)
    rebuild: bool = False    # True: changes the cached steps 0–3

    @property
    def stage(self) -> str:
        return self.key.split(".", 1)[0]

    @property
    def attr(self) -> str:
        return self.key.split(".", 1)[1]


def Toggle(key: str, label: str, help: str = "", rebuild: bool = False) -> Param:
    """An on/off switch, e.g. ``Toggle("Destripe.enabled", "Destripe")``."""
    return Param(key, 0, 1, 1, label, help, group="algorithms",
                 toggle=True, rebuild=rebuild)


def _stage_name(stage: StageLike) -> str:
    return getattr(stage, "name", "") or type(stage).__name__


def find_stage(stages: Iterable[StageLike], name: str) -> StageLike:
    for s in stages:
        if _stage_name(s) == name:
            return s
    raise KeyError(f"no stage named '{name}'")


def apply_params(stages: Sequence[StageLike], values: dict) -> list[str]:
    """Set ``{"Stage.attr": value}`` on the stage objects; return unknown keys."""
    unknown = []
    for key, value in values.items():
        name, _, attr = key.partition(".")
        try:
            stage = find_stage(stages, name)
        except KeyError:
            unknown.append(key)
            continue
        if not hasattr(stage, attr):
            unknown.append(key)
            continue
        setattr(stage, attr, value)
    return unknown


def load_params(path: str | Path) -> dict:
    """Read a preset file and return ``{"Stage.attr": value}``.

    Accepts the preset format (``{"values": {...}, "saved_at": ..., "note": ...}``)
    and a plain ``{"Stage.attr": value}`` dictionary.
    """
    path = Path(path)
    if not path.exists():
        return {}
    data = json.loads(path.read_text())
    if isinstance(data, dict) and isinstance(data.get("values"), dict):
        return data["values"]
    return data


def save_params(path: str | Path, values: dict, note: str = "") -> None:
    data = {"saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "note": note, "values": values}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, indent=2, sort_keys=True,
                                     ensure_ascii=False) + "\n")


# ----- presets (named versions of the settings) -----------------------------
_NAME_OK = re.compile(r"^[\w\- .]{1,60}$", re.UNICODE)


def preset_path(presets_dir: str | Path, name: str) -> Path:
    """Path of preset ``name``; the name may contain letters, digits, - _ . and spaces."""
    name = name.strip()
    if name.endswith(".json"):
        name = name[:-5]
    if not _NAME_OK.match(name) or name.startswith("."):
        raise ValueError("preset name: 1–60 characters, letters, digits, "
                         "space, '-', '_' or '.'")
    return Path(presets_dir) / f"{name}.json"


def resolve_preset(presets_dir: str | Path, name_or_path: str) -> Path:
    """A path to an existing file, or the name of a preset in ``presets_dir``."""
    p = Path(name_or_path)
    if p.suffix == ".json" and p.exists():
        return p
    p = preset_path(presets_dir, name_or_path)
    if not p.exists():
        raise FileNotFoundError(f"preset not found: {p}")
    return p


def list_presets(presets_dir: str | Path) -> list[dict]:
    """Presets in ``presets_dir``, newest first: name, saved_at, note."""
    out = []
    for p in Path(presets_dir).glob("*.json"):
        try:
            data = json.loads(p.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        meta = data if isinstance(data, dict) and "values" in data else {}
        out.append({"name": p.stem, "saved_at": meta.get("saved_at", ""),
                    "note": meta.get("note", ""), "mtime": p.stat().st_mtime})
    out.sort(key=lambda d: d["mtime"], reverse=True)
    for d in out:
        d.pop("mtime")
    return out


def to_uint8(img: np.ndarray, domain: Domain) -> np.ndarray:
    if img.dtype == np.uint8:
        return img
    if domain is Domain.DISPLAY:
        return np.clip(np.round(img * 255.0), 0, 255).astype(np.uint8)
    lo, hi = np.percentile(img, [0.5, 99.5])
    return np.clip((img - lo) / max(hi - lo, 1e-6) * 255, 0, 255).astype(np.uint8)


class TuningSession:
    """Cache of the front part + on-demand rendering of the back part.

    Parameters
    ----------
    source : frame source (e.g. ``RawFileSource``)
    front, back : lists of module objects (steps 0–3 and 5–7)
    params : tunable parameters (``Param`` list)
    reference : module/pipeline for the right panel, takes a RAW frame
    warmup : number of previous frames used to warm up stateful back modules
    """

    def __init__(self, source, front: list[StageLike], back: list[StageLike],
                 params: list[Param], reference: StageLike | None = None,
                 warmup: int = 40) -> None:
        self.source = source
        self.front = front
        self.back = back
        self.params = params
        self.reference = reference
        self.warmup = warmup
        self.n = len(source)
        self.indices = list(source.indices)
        self.lock = threading.Lock()
        self.defaults = self.values()
        self._img: np.ndarray | None = None
        self._noise: np.ndarray | None = None
        self._meta: list[dict] = []
        self._built: dict | None = None     # rebuild-values used for the cache
        self._last: int | None = None      # last rendered position
        self.status = "not built"

    # ----- parameters ------------------------------------------------------
    def _stages(self) -> list[StageLike]:
        return self.front + self.back

    def values(self) -> dict:
        out = {}
        for p in self.params:
            stage = find_stage(self._stages(), p.stage)
            out[p.key] = getattr(stage, p.attr)
        return out

    def set_values(self, values: dict) -> None:
        with self.lock:
            spec = {p.key: p for p in self.params}
            for key, v in values.items():
                p = spec.get(key)
                if p is None:
                    continue
                if p.toggle:
                    v = bool(v)
                else:
                    v = int(round(v)) if p.integer else float(v)
                    v = min(max(v, p.min), p.max)
                stage = find_stage(self._stages(), p.stage)
                if getattr(stage, p.attr) == v:
                    continue
                setattr(stage, p.attr, v)
            self._last = None  # state must be rebuilt

    def spec(self) -> list[dict]:
        return [asdict(p) for p in self.params]

    # ----- front cache -----------------------------------------------------
    def _rebuild_values(self) -> dict:
        return {p.key: getattr(find_stage(self._stages(), p.stage), p.attr)
                for p in self.params if p.rebuild}

    @property
    def front_dirty(self) -> bool:
        """True if a value of steps 0–3 differs from the one the cache was built with."""
        return self._built is None or self._built != self._rebuild_values()

    def build_front(self, progress: Callable[[int, int], None] | None = None) -> float:
        """Run steps 0–3 over the whole recording and keep the result."""
        with self.lock:
            t0 = time.perf_counter()
            self.status = "building"
            pipe = Pipeline(self.front, profile=False)
            h, w = self.source.height, self.source.width
            img = np.empty((self.n, h, w), np.float32)
            noise = np.empty((self.n, h, w), np.float16)
            meta = []
            for k, f in enumerate(pipe.run(self.source)):
                img[k] = f.image
                nm = f.layers.get("noise_sigma_map")
                noise[k] = nm if nm is not None else f.meta.get("noise_sigma", 0.0)
                meta.append({key: v for key, v in f.meta.items()
                             if isinstance(v, (int, float, str, bool))})
                if progress and k % 10 == 0:
                    progress(k, self.n)
            self._img, self._noise, self._meta = img, noise, meta
            self._built = self._rebuild_values()
            self._last = None
            dt = time.perf_counter() - t0
            self.status = f"steps 0–3 cached ({self.n} frames, {dt:.1f} s)"
            return dt

    def _front_frame(self, k: int) -> FrameData:
        meta = dict(self._meta[k])
        return FrameData(index=self.indices[k], image=self._img[k].copy(),
                         domain=Domain.LINEAR,
                         layers={"noise_sigma_map": self._noise[k].astype(np.float32)},
                         meta=meta)

    # ----- rendering -------------------------------------------------------
    def _reset_back(self) -> None:
        for s in self.back:
            if callable(getattr(s, "reset", None)):
                s.reset()

    def _warm(self, k: int) -> None:
        f = self._front_frame(k)
        for s in self.back:
            if not stage_runs(s, f):
                continue
            fn = getattr(s, "warmup", None)
            f = fn(f) if callable(fn) else s.process(f)
            if f is None:
                return

    def render(self, k: int) -> dict:
        """Render position ``k`` (0..n-1); return images and statistics."""
        if self._img is None:
            raise RuntimeError("front cache not built")
        k = int(min(max(k, 0), self.n - 1))
        with self.lock:
            t0 = time.perf_counter()
            if self._last is not None and k == self._last + 1:
                warmed = 0  # playback: state continues
            else:
                self._reset_back()
                start = max(0, k - self.warmup)
                # a frozen (shutter) frame shows the last good output, so that
                # frame is rendered fully instead of only warmed up
                last_good = None
                if self._meta[k].get("frozen"):
                    good = [j for j in range(start, k) if not self._meta[j].get("frozen")]
                    last_good = good[-1] if good else None
                for j in range(start, k):
                    if j == last_good:
                        f = self._front_frame(j)
                        for s in self.back:
                            if stage_runs(s, f):
                                f = s.process(f)
                    else:
                        self._warm(j)
                warmed = k - start
            f = self._front_frame(k)
            for s in self.back:
                if stage_runs(s, f):
                    f = s.process(f)
            self._last = k
            left = to_uint8(f.image, f.domain)

            right = None
            if self.reference is not None:
                raw = self.source.read(self.indices[k])
                r = self.reference.process(raw)
                right = to_uint8(r.image, r.domain)
            dt = time.perf_counter() - t0

        info = {
            "position": k,
            "frame": self.indices[k],
            "render_ms": round(dt * 1000),
            "warmup_frames": warmed,
            "white_pct": round(float(np.mean(left == 255)) * 100, 3),
            "black_pct": round(float(np.mean(left == 0)) * 100, 3),
            "mean": round(float(left.mean()), 1),
        }
        if right is not None:
            info["ref_white_pct"] = round(float(np.mean(right == 255)) * 100, 3)
            info["ref_black_pct"] = round(float(np.mean(right == 0)) * 100, 3)
        return {"left": left, "right": right, "info": info}
