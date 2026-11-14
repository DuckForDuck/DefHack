"""All tuning knobs in one place, each with a one-line explanation."""

from dataclasses import dataclass

WIDTH, HEIGHT = 640, 512
MAX_VALUE = 16383            # 14-bit sensor
AGC_MODES = ("absolute", "plateau", "percentile", "mean")


@dataclass
class Config:
    # --- which stages are switched on --------------------------------------
    badpix: bool = True
    stripes: bool = True
    temporal: bool = True     # also switches the offset stabiliser
    dde: bool = True          # detail enhancement (base/detail split)

    # --- 1. bad pixels ------------------------------------------------------
    spike_sigmas: float = 6.0       # spike = this many sigmas outside all 8 neighbours

    # --- 2. stripes ---------------------------------------------------------
    column_learn_rate: float = 0.05 # column offsets are static -> learn slowly

    # --- 3. offset stabiliser + temporal denoise ----------------------------
    offset_release_rate: float = 0.03  # how fast an offset jump fades back in
    still_weight: float = 0.25      # weight of the new frame in still areas
    motion_compensation: bool = True  # align previous frame along motion before averaging

    # --- 4. base / detail split ---------------------------------------------
    guided_radius: int = 4          # 9x9 window
    edge_sigmas: float = 6.0        # edges stronger than this stay in the base layer

    # --- 5. AGC -------------------------------------------------------------
    agc_mode: str = "absolute"      # "absolute" (v2) | "plateau" (v1) | "percentile" | "mean"
    # "absolute" mode
    black_percentile: float = 0.01  # black point = coolest 1 % of the scene
    white_percentile: float = 0.999 # white point = hottest 0.1 % (animals, people), not big roofs
    peak_percentile: float = 0.99995  # hottest pixels: top of the knee
    black_rate: float = 0.02        # black point follows slowly (~2 s) -> no pumping
    white_attack: float = 0.1       # white point rises fairly fast when something hotter appears
    white_release: float = 0.005    # ...and falls back slowly (~8 s), so it doesn't pump
    white_floor_sigmas: float = 80.0  # white point at least this many noise sigmas above black
    knee: float = 0.9               # scene gets this share of grey levels; hotter objects share the rest
    eq_share: float = 0.35          # share of v1 plateau equalisation blended in (0 = pure linear)
    gamma: float = 1.0              # < 1 brightens cool parts, > 1 spreads warm parts
    warm_suppress: bool = False     # darken LARGE warm areas (roofs); removes real thermal info
    warm_suppress_strength: float = 0.5  # 0..1 share of the large-area excess removed
    warm_suppress_size: int = 64    # px; warm objects narrower than this are never darkened
    # shared by the modes
    mean_std_factor: float = 3.0    # "mean" mode: range = mean +- 3 std
    clip_low: float = 0.002         # darkest 0.2 % ignored
    clip_high: float = 0.998        # brightest 0.2 % ignored
    plateau_factor: float = 3.0     # histogram bins capped at 3x the average bin
    linear_share: float = 0.10      # share of grey levels given out linearly
    min_range_sigmas: float = 40.0  # never stretch less than 40 sigma to full scale
    lut_smoothing: float = 0.08     # LUT update per frame (lower = steadier)
    lut_smoothing_fast: float = 0.25  # used on a real scene change
    scene_change_share: float = 0.2   # share of pixels outside LUT range = scene change
    out_low: int = 8                # leave headroom for detail
    out_high: int = 247

    # --- 6. detail ----------------------------------------------------------
    detail_gain: float = 2.0


# --- ranges for the tuning UI: name -> (min, max, step) ------------------------
PARAM_RANGES = {
    "black_percentile": (0.0, 0.2, 0.005),
    "white_percentile": (0.9, 1.0, 0.001),
    "peak_percentile": (0.99, 1.0, 0.0001),
    "black_rate": (0.001, 1.0, 0.001),
    "white_attack": (0.01, 1.0, 0.01),
    "white_release": (0.001, 1.0, 0.001),
    "knee": (0.5, 1.0, 0.01),
    "white_floor_sigmas": (0.0, 400.0, 5.0),
    "eq_share": (0.0, 1.0, 0.05),
    "gamma": (0.4, 2.5, 0.05),
    "warm_suppress_strength": (0.0, 1.0, 0.05),
    "warm_suppress_size": (16, 320, 8),
    "spike_sigmas": (2.0, 20.0, 0.5),
    "column_learn_rate": (0.0, 0.5, 0.01),
    "offset_release_rate": (0.0, 0.3, 0.005),
    "still_weight": (0.05, 1.0, 0.05),
    "guided_radius": (1, 12, 1),
    "edge_sigmas": (1.0, 20.0, 0.5),
    "mean_std_factor": (1.0, 6.0, 0.1),
    "clip_low": (0.0, 0.05, 0.001),
    "clip_high": (0.95, 1.0, 0.001),
    "plateau_factor": (0.5, 20.0, 0.5),
    "linear_share": (0.0, 1.0, 0.01),
    "min_range_sigmas": (5.0, 200.0, 5.0),
    "lut_smoothing": (0.01, 1.0, 0.01),
    "lut_smoothing_fast": (0.01, 1.0, 0.01),
    "scene_change_share": (0.05, 0.9, 0.05),
    "out_low": (0, 64, 1),
    "out_high": (192, 255, 1),
    "detail_gain": (0.0, 6.0, 0.1),
}


def set_param(cfg, name, value):
    """Set one Config field from a string/number, with the right type and range."""
    import dataclasses
    fields = {f.name: f for f in dataclasses.fields(cfg)}
    if name not in fields:
        raise KeyError(f"unknown parameter '{name}'")
    current = getattr(cfg, name)
    if isinstance(current, bool):
        value = value if isinstance(value, bool) else str(value).lower() in ("1", "true", "yes", "on")
    elif isinstance(current, int):
        value = int(round(float(value)))
    elif isinstance(current, float):
        value = float(value)
    else:
        value = str(value)
        if name == "agc_mode" and value not in AGC_MODES:
            raise ValueError(f"agc_mode must be one of {AGC_MODES}")
    if name in PARAM_RANGES:
        low, high, _ = PARAM_RANGES[name]
        value = type(current)(min(max(value, low), high))
    setattr(cfg, name, value)
    return value


def save_config(cfg, path):
    import dataclasses, json
    with open(path, "w") as fh:
        json.dump(dataclasses.asdict(cfg), fh, indent=2)


def load_config(path):
    import json
    cfg = Config()
    with open(path) as fh:
        for name, value in json.load(fh).items():
            set_param(cfg, name, value)
    return cfg
