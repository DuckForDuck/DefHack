"""Entry point: build the pipeline from modules and run it on a recording.

The pipeline is described once here and used by both ``main.py`` (full run)
and ``tuner.py`` (interactive tuning in the browser):

- ``build_front``  – steps 0–3 (heavy, stateful; cached by the tuner)
- ``build_back``   – steps 5–7 (tuned by eye)
- ``TUNABLE``      – parameters shown as sliders in the tuner
- ``presets/``     – named versions of the settings saved by the tuner;
                     ``--params <name>`` applies one (without it: defaults)
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from thermal import Pipeline, RawFileSource
from thermal.stages import (BadPixels, BaseDetailSplit, Compare,
                            CompactHotBoost, Decode,
                            DetailEnhance, Destripe, HotSpotContrast,
                            LocalContrast, MultiScaleDetail, NaiveAGC,
                            Preview, Quantise, ResidualNUC, ShutterGuard,
                            ShutterHold, SpatialDenoise,
                            TemporalDenoise, ToneMap, Upscale, VideoWriter,
                            write_range_test)
from thermal.tuning import (Param, Toggle, apply_params, load_params,
                            resolve_preset)

ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = ROOT.parent / "normal_nuc_2.bin"
DEBUG_DIR = ROOT / "debug"
OUTPUT_DIR = ROOT / "output"
PRESETS_DIR = ROOT / "presets"

# Frames with known events (see docs, section 1).
KEY_FRAMES = [0, 3, 6, 100, 110, 118, 137, 164, 175, 200]

# Parameters that the tuner shows. Key = "<StageName>.<attribute>".
# Toggle = on/off switch. rebuild=True: changes the cached steps 0–3 (≈ 12 s).
TUNABLE = [
    # algorithms on / off
    Toggle("ShutterGuard.enabled", "0  Shutter / FFC guard (M)", "flat shutter frames: freeze output, no state update", rebuild=True),
    Toggle("BadPixels.enabled", "1  Bad pixels", "replace dead / hot / blinking pixels", rebuild=True),
    Toggle("ResidualNUC.enabled", "2a Residual NUC", "remove the fixed per-pixel pattern (static noise)", rebuild=True),
    Toggle("Destripe.enabled", "2b Destripe", "remove per-frame row / column stripes", rebuild=True),
    Toggle("TemporalDenoise.enabled", "3a Temporal denoise", "motion-compensated averaging (dynamic noise)", rebuild=True),
    Toggle("SpatialDenoise.enabled", "3b Spatial denoise", "guided filter on the remaining noise", rebuild=True),
    Toggle("BaseDetailSplit.enabled", "5  Base / detail split", "off: tone map the whole image, no detail layer"),
    Toggle("ToneMap.use_plateau", "6  Histogram equalisation", "plateau HE; off = linear curve"),
    Toggle("ToneMap.use_knee", "6  Soft knee / foot", "off = hard clipping at the scene range"),
    Toggle("ToneMap.temporal", "6  Temporal smoothing (AGC)", "off = range from the current frame only"),
    Toggle("ToneMap.anchor_black", "6  Anchored black point (M)", "black point moves slowly; fast only on a real scene change"),
    Toggle("ToneMap.white_hold", "6  White point peak hold (M)", "top of the range: fast up, very slow down"),
    Toggle("ToneMap.dde", "6  DDE detail (own gain)", "detail added with its own gain, not the slope of the tone curve"),
    Toggle("CompactHotBoost.enabled", "6c Compact hot boost / top-hat (M)", "small hot objects brighter than large warm areas"),
    Toggle("CompactHotBoost.soft", "6c Top-hat: soft (no clipping)", "off = add and clip as in Michael's code"),
    Toggle("HotSpotContrast.enabled", "6b Hot-object local contrast", "inner structure of hot objects in the knee zone"),
    Toggle("LocalContrast.enabled", "6+ CLAHE (local contrast)", "off by default: can break temperature order, halos"),
    Toggle("DetailEnhance.enabled", "7  Fine detail enhancement", "noise-gated sharpening, one fine scale"),
    Toggle("MultiScaleDetail.enabled", "7b Multi-scale detail", "3 scales, noise gate, no boost of strong edges; compare with 7"),

    # step 6 – tone mapping
    Param("ToneMap.gamma", 0.5, 2.5, 0.05, "Black level (gamma)",
          ">1 darker ground and mid-tones, <1 lighter"),
    Param("ToneMap.white", 0.70, 1.0, 0.01, "White level",
          "display value of the hottest pixels; <1 = less glare"),
    Param("ToneMap.knee", 0.40, 0.97, 0.01, "Knee start",
          "lower = more grey levels inside hot objects"),
    Param("ToneMap.foot", 0.0, 0.15, 0.01, "Foot (cold reserve)",
          "display range reserved below the scene range"),
    Param("ToneMap.plateau", 1.0, 10.0, 0.1, "Plateau (HE)",
          "lower = large uniform areas take fewer grey levels"),
    Param("ToneMap.linear_mix", 0.0, 1.0, 0.05, "Linear mix",
          "share of the linear curve; 1 = no histogram equalisation"),
    Param("ToneMap.detail_gain", 0.0, 4.0, 0.1, "Mid detail gain",
          "texture of the size of the split radius"),
    Param("ToneMap.p_high", 98.0, 100.0, 0.05, "Scene range top (pct)",
          "percentile above which values go to the knee"),
    Param("ToneMap.p_low", 0.0, 10.0, 0.1, "Scene range bottom (pct)",
          "percentile below which values go to the foot; ~5 = cold sky does not brighten the scene"),
    # step 6 – adaptation (AGC speed)
    Param("ToneMap.alpha_grow", 0.01, 1.0, 0.01, "Adapt: range grows",
          "speed when a hot / cold object enters (0.15 ≈ 6 frames; 1 = at once)"),
    Param("ToneMap.alpha_shrink", 0.005, 0.5, 0.005, "Adapt: range shrinks",
          "speed when it leaves (0.05 ≈ 20 frames); slow = no pumping"),
    Param("ToneMap.alpha_hist", 0.01, 1.0, 0.01, "Adapt: histogram",
          "speed of the tone curve change (1 = per frame, may flicker)"),
    Param("ToneMap.peak_attack", 0.05, 1.0, 0.05, "Peak hold: attack",
          "how fast a new hottest value is followed (1 = at once)"),
    Param("ToneMap.peak_hold", 0, 75, 1, "Peak hold: frames",
          "frames the hottest value is kept before release", integer=True),
    # step 5 – base/detail split
    Param("BaseDetailSplit.radius", 2, 24, 1, "Split radius (px)",
          "size of what counts as detail", integer=True),
    Param("BaseDetailSplit.edge_dn", 5.0, 100.0, 1.0, "Split edge (DN)",
          "contrast that stays in the base layer"),
    # step 7 – fine detail
    Param("DetailEnhance.gain", 0.0, 4.0, 0.1, "Fine detail gain",
          "sharpness of fine edges; also emphasises smeared streaks"),
    Param("DetailEnhance.coring_k", 0.0, 6.0, 0.1, "Noise gate",
          "higher = less noise and fewer smeared streaks are boosted"),
    Param("DetailEnhance.sigma", 0.6, 3.0, 0.1, "Fine detail scale (px)",
          "size of the boosted detail"),
    Param("DetailEnhance.max_amp", 0.01, 0.15, 0.005, "Halo limit",
          "maximum added brightness (0.06 ≈ 15 grey levels)"),
    # step 6b – local contrast inside hot objects
    Param("HotSpotContrast.gain", 0.0, 6.0, 0.1, "Hot objects: contrast",
          "extra local contrast inside hot objects (knee zone)"),
    Param("HotSpotContrast.sigma", 1.0, 12.0, 0.5, "Hot objects: scale (px)",
          "size of the structure that is raised"),
    Param("HotSpotContrast.edge", 0.005, 0.15, 0.005, "Hot objects: border",
          "contrast that counts as the object border (kept, not raised)"),
    Param("HotSpotContrast.width", 0.01, 0.15, 0.01, "Hot objects: fade-in",
          "display range above the knee over which the effect starts"),
    # step 7b – multi-scale detail (only when switched on)
    Param("MultiScaleDetail.fine", 0.0, 3.0, 0.1, "Multi-scale: fine (1-2 px)",
          "extra gain of the finest band"),
    Param("MultiScaleDetail.mid", 0.0, 3.0, 0.1, "Multi-scale: mid (2-5 px)",
          "extra gain of the middle band"),
    Param("MultiScaleDetail.coarse", 0.0, 3.0, 0.1, "Multi-scale: coarse (5-12 px)",
          "extra gain of the coarse band"),
    Param("MultiScaleDetail.edge", 0.01, 0.3, 0.005, "Multi-scale: edge protection",
          "lower = strong edges boosted less (fewer halos)"),
    Param("MultiScaleDetail.coring_k", 0.0, 6.0, 0.1, "Multi-scale: noise gate",
          "higher = less noise boosted"),
    # CLAHE (only when switched on)
    Param("LocalContrast.clip_limit", 1.0, 6.0, 0.1, "CLAHE clip limit",
          "contrast limit per tile"),
    Param("LocalContrast.tiles", 2, 16, 1, "CLAHE tiles per side",
          "fewer = larger regions", integer=True),
    Param("LocalContrast.amount", 0.0, 1.0, 0.05, "CLAHE amount",
          "blend with the global tone mapping"),
    # step 6 – Michael's options (only when the toggles are on)
    Param("ToneMap.lo_slew", 0.1, 50.0, 0.1, "Anchor: black point slew (DN/frame)",
          "max. change of the black point per frame (Michael: 1.5)", group="michael"),
    Param("ToneMap.fast_frac", 0.01, 0.6, 0.01, "Anchor: scene change share",
          "share of the frame below the black point that counts as a real scene change", group="michael"),
    Param("ToneMap.fast_gap", 0.05, 1.0, 0.05, "Anchor: warm-up gap",
          "empty share of the range at the bottom that counts as a real scene change", group="michael"),
    Param("ToneMap.hi_attack", 0.01, 1.0, 0.01, "White hold: attack",
          "how fast the top of the range follows a hotter scene", group="michael"),
    Param("ToneMap.hi_decay", 0.0, 20.0, 0.05, "White hold: release (DN/frame)",
          "how fast the top of the range comes down", group="michael"),
    Param("ToneMap.min_span_dn", 0.0, 1500.0, 10.0, "Min. range (DN)",
          "range never narrower; Michael 1000 (milky here), try 400-600", group="michael"),
    Param("ToneMap.shadow_lift", 0.0, 0.3, 0.01, "Shadow lift",
          "raises dark tones (Michael 0.12); 0 = off", group="michael"),
    # step 6 – DDE detail (only when "DDE detail" is on)
    Param("ToneMap.dde_gain", 0.0, 8.0, 0.1, "DDE: fixed gain",
          "detail gain in units of the linear tone-map slope (Michael 2.6)", group="michael"),
    Param("ToneMap.dde_adapt", 0.0, 1.0, 0.05, "DDE: adaptive share",
          "0 = fixed gain; 1 = every region gets the target detail contrast", group="michael"),
    Param("ToneMap.dde_target", 0.005, 0.1, 0.005, "DDE: target contrast",
          "local detail contrast for the adaptive part (0.03 ≈ 8 grey levels)", group="michael"),
    Param("ToneMap.dde_max", 1.0, 20.0, 0.5, "DDE: max gain",
          "limit of the adaptive gain, x linear slope (noise in flat areas)", group="michael"),
    Param("ToneMap.dde_radius", 2, 24, 1, "DDE: local window (px)",
          "size of the region for the local detail contrast", integer=True, group="michael"),
    Param("ToneMap.dde_coring_k", 0.0, 6.0, 0.1, "DDE: noise gate",
          "detail of the size of the noise is not boosted", group="michael"),
    Param("ToneMap.dde_limit", 0.02, 0.5, 0.01, "DDE: soft limit",
          "max. added detail (0.2 ≈ 50 grey levels)", group="michael"),
    Param("ToneMap.warm_boost", 1.0, 3.0, 0.05, "DDE: warm detail boost",
          "extra gain for warm (positive) detail (Michael 1.5)", group="michael"),
    # step 6c – compact hot boost (top-hat)
    Param("CompactHotBoost.gain", 0.0, 4.0, 0.1, "Top-hat: gain",
          "boost of compact hot objects (Michael 1.1)", group="michael"),
    Param("CompactHotBoost.threshold_dn", 0.0, 600.0, 5.0, "Top-hat: threshold (DN)",
          "how much hotter than the surroundings before the boost starts", group="michael"),
    Param("CompactHotBoost.size", 3, 21, 2, "Top-hat: window (px, half res)",
          "objects smaller than ≈ 2 x window count as compact", integer=True, group="michael"),
    # step 3 – denoising (slow: rebuild of steps 0–3)
    Param("TemporalDenoise.max_frames", 1, 16, 1, "Temporal memory (frames)",
          "more = less noise, more risk of smear", group="denoise", integer=True,
          rebuild=True),
    Param("TemporalDenoise.pixel_hi", 3.0, 10.0, 0.5, "Temporal tolerance",
          "higher = averages more, more risk of ghosts", group="denoise",
          rebuild=True),
    Param("SpatialDenoise.strength", 0.0, 4.0, 0.1, "Spatial denoise strength",
          "cleaning where temporal averaging was rejected", group="denoise",
          rebuild=True),
    Param("Destripe.temporal_alpha", 0.0, 1.0, 0.01, "Destripe: smoothing over time (M)",
          "0 = per frame; 0.1 = Michael (stripes fixed to the sensor)", group="denoise",
          rebuild=True),
]


def fit_nuc(nuc: ResidualNUC, args, step: int) -> None:
    """Calibrate the residual NUC on every `step`-th frame of the recording.

    The frames go through the same first steps as in the main pipeline
    (decode + bad pixels), so the offset is learned on clean data.
    """
    t0 = time.perf_counter()
    calib_source = RawFileSource(args.input, width=args.width, height=args.height,
                                 step=step)
    prep = Pipeline([Decode(bit_depth=14), BadPixels()], profile=False)
    nuc.fit(f.image for f in prep.run(calib_source))
    print(f"ResidualNUC fitted on {len(calib_source)} frames "
          f"in {time.perf_counter() - t0:.1f} s, offset std {nuc.offset.std():.2f} DN")


def build_front(args) -> list:
    """Steps 0–3 (linear domain). Order: docs/thermal_video_pipeline.md, section 3."""
    nuc = ResidualNUC(alpha=0.02, blur_sigma=2.0, clip_k=5.0, motion_k=3.0)
    if args.nuc_fit_step > 0:
        fit_nuc(nuc, args, args.nuc_fit_step)
    return [
        Decode(bit_depth=14, on_overflow="warn"),                    # step 0
        ShutterGuard(ratio=0.35, max_freeze=30),                     # step 0b
        BadPixels(k=8.0, k_replace=15.0),                            # step 1
        nuc,                                                         # step 2a
        Destripe(columns=True, rows=True, edge_k=0.5),               # step 2b
        TemporalDenoise(max_frames=8, flow_preset="medium"),         # step 3a
        SpatialDenoise(method="guided", strength=1.5),               # step 3b
        # step 4 (super-resolution / deblurring) postponed, see WORKLOG entry 12
    ]


def build_back() -> list:
    """Steps 5–7 (to display). Defaults; a preset (--params) overrides them."""
    return [
        BaseDetailSplit(radius=8, edge_dn=25.0),                     # step 5
        # Recommended values (WORKLOG entry 17): balance between a clean,
        # not "painted" look and stable contrast when the sky enters.
        ToneMap(plateau=3.0, linear_mix=0.6, foot=0.05, knee=0.75,   # step 6
                gamma=1.3, white=1.0, detail_gain=1.5),
        CompactHotBoost(gain=1.1, size=7, threshold_dn=150.0,        # step 6c,
                        soft=True, enabled=False),                   # off
        HotSpotContrast(gain=2.0, sigma=3.0, edge=0.03, width=0.05, # step 6b,
                        enabled=False),                              # off
        LocalContrast(clip_limit=2.0, tiles=8, amount=0.5,           # optional,
                      enabled=False),                                # off
        DetailEnhance(gain=1.0, sigma=1.2, coring_k=3.0),            # step 7
        MultiScaleDetail(fine=0.6, mid=1.0, coarse=0.6, edge=0.06,   # step 7b,
                         coring_k=3.0, enabled=False),               # off
        ShutterHold(),                                               # last good image for shutter frames
    ]


def build_output(args) -> list:
    """Steps 8–10: upscale, 8 bit with dither, video files (not in the tuner)."""
    stages = [
        Upscale(scale=args.upscale, method=args.upscale_method),     # step 8
        Quantise(dither="blue", amplitude=1.0),                      # step 9
    ]
    if not args.no_video:                                            # step 10
        stem = Path(args.input).stem + "__" + (Path(args.params).stem if args.params
                                                else "defaults")
        if args.upscale_method != "lanczos":
            stem += f"__{args.upscale_method}"
        out = Path(args.out)
        stages += [
            VideoWriter(out / f"{stem}.mp4", fmt="h264", fps=args.fps, crf=args.crf),
        ]
        if not args.no_mkv:  # lossless reference copy (not needed for submission)
            stages.append(VideoWriter(out / f"{stem}.mkv", fmt="ffv1", fps=args.fps))
    return stages


def apply_preset(stages: list, name: str | None) -> None:
    """Apply preset ``name`` (from presets/) or a .json path; None = defaults."""
    if not name:
        print("parameters: defaults from the code")
        return
    try:
        path = resolve_preset(PRESETS_DIR, name)
    except (FileNotFoundError, ValueError) as e:
        names = sorted(p.stem for p in PRESETS_DIR.glob("*.json"))
        raise SystemExit(f"{e}\navailable presets: {', '.join(names) or '(none)'}")
    values = load_params(path)
    unknown = apply_params(stages, values)
    print(f"parameters: preset '{path.stem}' ({len(values) - len(unknown)} values)"
          + (f", unknown keys ignored: {unknown}" if unknown else ""))


def build_pipeline(args, preview_frames, show: bool) -> Pipeline:
    front, back = build_front(args), build_back()
    apply_preset(front + back, args.params)
    pipe = Pipeline(front + back + build_output(args))

    for after, tag in [("Decode", "00_decode"), ("BadPixels", "01_badpixels"),
                       ("Destripe", "02_destripe"), ("SpatialDenoise", "03_denoise"),
                       ("ToneMap", "06_tonemap"), ("DetailEnhance", "07_detail"),
                       ("Quantise", "09_output")]:
        pipe.insert_after(after, Preview(tag=tag, frames=preview_frames,
                                         out_dir=DEBUG_DIR))

    # Live side-by-side view: pipeline (left) vs. naive script.py (right).
    # With the window: all frames, live. Without it: key frames saved
    # as side-by-side PNG files in debug/compare/.
    pipe.add(Compare(reference_label="naive (script.py)", show=show,
                     save=not show, frames=None if show else preview_frames,
                     scale=args.scale, out_dir=DEBUG_DIR))
    return pipe


def build_naive_pipeline(args, preview_frames, show: bool) -> Pipeline:
    """Baseline for comparison: the processing of the original script.py."""
    return Pipeline([
        Decode(bit_depth=14),
        NaiveAGC(low=2, high=98),
        Preview(tag="naive", frames=preview_frames, show=show,
                out_dir=DEBUG_DIR),
    ])


def add_source_args(ap: argparse.ArgumentParser) -> None:
    """Arguments shared by main.py and tuner.py."""
    ap.add_argument("input", nargs="?", default=str(DEFAULT_INPUT))
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=512)
    ap.add_argument("--fps", type=float, default=25.0)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--stop", type=int, default=None)
    ap.add_argument("--nuc-fit-step", type=int, default=3,
                    help="calibrate ResidualNUC on every N-th frame before "
                         "the run; 0 = no calibration (learn during the run)")
    ap.add_argument("--params", metavar="NAME", default=None,
                    help="preset from presets/ (name without .json) or a path "
                         "to a .json file; without it the defaults in the code are used")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_source_args(ap)
    ap.add_argument("--naive", action="store_true",
                    help="run the naive baseline (script.py) instead")
    ap.add_argument("--frames", type=str, default="key",
                    help="frames to save as previews: 'key', 'all' or e.g. '0,100,110'")
    ap.add_argument("--no-show", action="store_true",
                    help="do not open the comparison window (batch run)")
    ap.add_argument("--scale", type=float, default=1.0,
                    help="size factor of the comparison window, e.g. 0.75")
    ap.add_argument("--out", default=str(OUTPUT_DIR),
                    help="folder for the video files")
    ap.add_argument("--no-mkv", action="store_true",
                    help="write only the MP4 (no lossless .mkv reference copy)")
    ap.add_argument("--no-video", action="store_true",
                    help="do not write video files")
    ap.add_argument("--upscale", type=float, default=1.0,
                    help="upscaling factor of step 8 (1 = none, video = sensor size 640x512)")
    ap.add_argument("--upscale-method", default="lanczos",
                    choices=["lanczos", "cubic", "linear", "nearest"],
                    help="interpolation of step 8: lanczos = smooth (default), "
                         "nearest = square sensor pixels (use an integer --upscale)")
    ap.add_argument("--crf", type=int, default=10,
                    help="H.264 quality (lower = better, larger file)")
    ap.add_argument("--range-test", action="store_true",
                    help="only write output/range_test.mp4 (check full range in a player)")
    args = ap.parse_args()

    if args.range_test:
        path = write_range_test(Path(args.out) / "range_test.mp4", fps=args.fps)
        print(f"open {path}: top = ramp 0..255; bottom patches 0, 8, 16, 24 | "
              "231, 239, 247, 255 – all must look different")
        return

    if args.frames == "key":
        frames = KEY_FRAMES
    elif args.frames == "all":
        frames = None
    else:
        frames = [int(x) for x in args.frames.split(",")]

    source = RawFileSource(args.input, width=args.width, height=args.height,
                           fps=args.fps, start=args.start, stop=args.stop)
    build = build_naive_pipeline if args.naive else build_pipeline
    pipeline = build(args, frames, not args.no_show)
    print(source)
    print(pipeline)

    n = 0
    last = None
    for frame in pipeline.run(source):
        n += 1
        last = frame
        if args.naive and (frames is None or frame.index in frames):
            m = frame.meta
            print(f"#{frame.index:4d}  lo={m['naive.lo']:6.0f}  hi={m['naive.hi']:6.0f}  "
                  f"black={100 * m['naive.black']:5.2f} %  "
                  f"white={100 * m['naive.white']:5.2f} %")

    print(f"\nprocessed frames: {n}")
    if last is not None:
        print(f"last frame: index={last.index}, domain={last.domain.value}, "
              f"dtype={last.image.dtype}, shape={last.image.shape}")
        if "noise_sigma" in last.meta:
            print(f"noise sigma (last frame): {last.meta['noise_sigma']:.2f} DN")
    print(f"previews: {DEBUG_DIR}")
    print("\ntiming per stage:")
    print(pipeline.timing_report())


if __name__ == "__main__":
    main()
