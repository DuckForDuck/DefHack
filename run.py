"""
run.py - one command: raw .bin -> two videos, BEFORE (raw min/max) and AFTER (ISP white-hot).

  python run.py data.bin                       # out/data_before.mp4 + out/data_after.mp4 (+ data_compare.mp4)
  python run.py data.bin --config tuned.json   # parameters saved from tune.py
  python run.py data.bin --set detail_gain=3 --set plateau_factor=5
  python run.py data.bin --no-compare          # skip the side-by-side video
  python run.py data.bin --ablation            # extra videos, each with one stage switched off
  python run.py data.bin --agc percentile      # plateau (default) | percentile | mean
  python run.py data.bin --agc-compare         # extra video: all AGC modes side by side
  python run.py data.bin --no-motion-comp      # temporal denoise without motion compensation (faster)
  python run.py --list-params                  # show every parameter and its range
"""
import argparse
import dataclasses
import os
import time

import numpy as np
import cv2

from isp import (Config, ThermalISP, naive_scale, open_frames, Mp4Writer, WIDTH, HEIGHT,
                 AGC_MODES, PARAM_RANGES, set_param, load_config)

STAGES = ("badpix", "stripes", "temporal", "dde")


def put_label(image, text):
    image = image.copy()
    cv2.putText(image, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 0, 4)
    cv2.putText(image, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 255, 2)
    return image


def print_timing(name, seconds):
    ms = np.array(seconds[1:] or seconds) * 1000
    print(f"[{name}] {len(seconds)} frames  mean {ms.mean():.1f} ms  "
          f"p95 {np.percentile(ms, 95):.1f} ms  max {ms.max():.1f} ms  "
          f"-> {1000 / ms.mean():.0f} fps (budget 40 ms)")


def export_videos(path, out_dir, fps, cfg, compare=True, suffix="", progress=None,
                  width=WIDTH, height=HEIGHT):
    """Write <name>_before.mp4 and <name>_after.mp4 (and optionally <name>_compare.mp4).

    `progress(done, total)` is called after each frame (used by tune.py)."""
    os.makedirs(out_dir, exist_ok=True)
    frames = open_frames(path, width, height)
    name = os.path.splitext(os.path.basename(path))[0] + suffix
    isp = ThermalISP(dataclasses.replace(cfg), width, height)

    before = Mp4Writer(os.path.join(out_dir, f"{name}_before.mp4"), width, height, fps)
    after = Mp4Writer(os.path.join(out_dir, f"{name}_after.mp4"), width, height, fps)
    side_by_side = None
    if compare:
        side_by_side = Mp4Writer(os.path.join(out_dir, f"{name}_compare.mp4"), 2 * width, height, fps)

    seconds = []
    for i, raw in enumerate(frames):
        raw = np.array(raw)
        start = time.perf_counter()
        image = isp.process(raw)
        seconds.append(time.perf_counter() - start)

        raw8 = naive_scale(raw)
        before.write(raw8)
        after.write(image)
        if side_by_side:
            side_by_side.write(np.hstack([put_label(raw8, "BEFORE: raw min/max"),
                                          put_label(image, "AFTER: ISP")]))
        if i in (0, len(frames) // 2, len(frames) - 1):
            cv2.imwrite(os.path.join(out_dir, f"{name}_frame{i:04d}.png"), image)
        if progress:
            progress(i + 1, len(frames))

    for writer in (before, after, side_by_side):
        if writer:
            writer.close()
    print_timing(name, seconds)
    return [os.path.join(out_dir, f"{name}_{kind}.mp4")
            for kind in ("before", "after") + (("compare",) if compare else ())]


def compare_agc_modes(path, out_dir, fps, base_cfg):
    """One video with every AGC mode side by side (same frames, same other stages)."""
    frames = open_frames(path)
    name = os.path.splitext(os.path.basename(path))[0]
    pipelines = [ThermalISP(dataclasses.replace(base_cfg, agc_mode=mode)) for mode in AGC_MODES]
    video = Mp4Writer(os.path.join(out_dir, f"{name}_agc_modes.mp4"),
                      WIDTH * len(AGC_MODES), HEIGHT, fps)
    for raw in frames:
        raw = np.array(raw)
        panels = [put_label(isp.process(raw), f"AGC: {mode}")
                  for isp, mode in zip(pipelines, AGC_MODES)]
        video.write(np.hstack(panels))
    video.close()
    print(f"[{name}] AGC comparison written ({', '.join(AGC_MODES)})")


def list_params():
    cfg = Config()
    for field in dataclasses.fields(cfg):
        value = getattr(cfg, field.name)
        extra = ""
        if field.name in PARAM_RANGES:
            low, high, step = PARAM_RANGES[field.name]
            extra = f"range {low} .. {high}"
        elif field.name == "agc_mode":
            extra = " | ".join(AGC_MODES)
        elif isinstance(value, bool):
            extra = "true | false"
        print(f"  {field.name:<22} default {value!s:<10} {extra}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("bins", nargs="*", help="one or more raw .bin sequences")
    parser.add_argument("--out", default="out")
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--config", help="JSON file with parameters (saved from tune.py)")
    parser.add_argument("--set", action="append", default=[], metavar="NAME=VALUE",
                        help="override one parameter, can be repeated")
    parser.add_argument("--agc", choices=AGC_MODES, help="AGC type")
    parser.add_argument("--agc-compare", action="store_true",
                        help="also write a video with all AGC modes side by side")
    parser.add_argument("--no-motion-comp", action="store_true",
                        help="temporal denoise without motion compensation (faster)")
    parser.add_argument("--no-compare", action="store_true")
    parser.add_argument("--ablation", action="store_true")
    parser.add_argument("--list-params", action="store_true")
    args = parser.parse_args()

    if args.list_params or not args.bins:
        list_params()
        return

    base_cfg = load_config(args.config) if args.config else Config()
    if args.agc:
        base_cfg.agc_mode = args.agc
    if args.no_motion_comp:
        base_cfg.motion_compensation = False
    for item in args.set:
        name, _, value = item.partition("=")
        set_param(base_cfg, name.strip(), value.strip())

    for path in args.bins:
        for video in export_videos(path, args.out, args.fps, base_cfg, not args.no_compare):
            print("  wrote", video)
        if args.agc_compare:
            compare_agc_modes(path, args.out, args.fps, base_cfg)
        if args.ablation:
            for stage in STAGES:
                cfg = dataclasses.replace(base_cfg, **{stage: False})
                export_videos(path, args.out, args.fps, cfg, False, suffix=f"_no-{stage}")


if __name__ == "__main__":
    main()
