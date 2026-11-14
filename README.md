# Thermal ISP – 640×512 14-bit microbolometer → 8-bit white-hot video

Needs: Python 3.9+, numpy, opencv-python, ffmpeg on PATH. No other installs.

## 1. Tune parameters live (sliders + number boxes)
    python tune.py data.bin
Opens http://127.0.0.1:8765 : BEFORE (raw min/max) next to AFTER (ISP).
Every parameter has a slider and a number box; hover a name for its meaning.
Buttons: Play/Pause, Step (or ←/→, Space), seek bar, Save / Download / Load config,
Defaults, Restart pipeline, **Export before / after videos** (uses the current settings).
Exported to `out/`: `data_before.mp4`, `data_after.mp4`, `data_compare.mp4`, `used_config.json`.

## 2. Batch export from the command line
    python run.py data.bin                                   # defaults
    python run.py data.bin --config out/tuned.json           # settings saved from the tuner
    python run.py data.bin --set detail_gain=3 --set plateau_factor=5
    python run.py --list-params                              # all parameters + ranges
Other flags: --no-compare, --ablation, --agc-compare, --agc MODE, --no-motion-comp, --fps.

## AGC modes (agc_mode)
- **absolute** (default, v3): fixed counts-to-grey scale. Slow black point, peak-tracking white
  point with a floor (`white_floor_sigmas`), extreme hot objects share only the top 10 % (`knee`),
  plus a share of v1 equalisation (`eq_share`) for texture in large cool areas.
  Roofs keep a stable mid-grey, animals stay brightest, nothing pumps when objects enter.
- **plateau** (v1): histogram equalisation. Most local contrast, but brightness depends on
  frame content (large warm roofs go near-white).
- percentile, mean: simple linear stretches.
- `warm_suppress` (off by default): darkens LARGE warm areas (roofs) edge to edge; objects
  narrower than `warm_suppress_size` px (animals) keep full brightness. This removes real
  thermal information, so it is a switch, not a default.

## Other tools
    python make_farm.py farm.bin 200         # test scene with the measured levels (road/roofs/animals)
    python make_synthetic.py test.bin 200    # test sequence with all typical defects
    python explore.py data.bin               # statistics of the raw data
    python view.py data.bin                  # older OpenCV viewer (keys 1-5)

Input: raw little-endian uint16 frames, 14 bits used (top 2 bits zero), post-NUC.
