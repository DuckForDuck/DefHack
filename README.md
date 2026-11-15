# Thermal video pipeline

Converts raw 14-bit thermal data (`normal_nuc_2.bin`) into an 8-bit video.
The steps, their order and the reasons for the order are described in
`../docs/thermal_video_pipeline.md` (Ukrainian version: `thermal_video_pipeline.uk.md`).
What has been done so far, with results: `WORKLOG.md`.

## Setup

```
python -m venv .venv && source .venv/bin/activate     # or use ../.venv
pip install -r requirements.txt
```

## Tuner (interactive, in the browser)

```
python tuner.py                       # opens http://127.0.0.1:8765, starts from the defaults
python tuner.py --params dark-ground  # starts from a saved preset
```

- Steps 0–3 are computed once for the whole recording (≈ 12 s) and kept in memory.
- Every slider change re-renders the shown frame at once (steps 5–7, ≈ 0.15 s;
  ≈ 30 ms per frame during playback).
- Left: pipeline; right: naive `script.py`. White/black share is shown above each image.
- **Algorithms (on / off)**: every processing algorithm can be switched off to
  see what it contributes (bad pixels, residual NUC, destripe, temporal and
  spatial denoise, base/detail split, histogram equalisation, soft knee,
  temporal AGC smoothing, CLAHE, fine detail). A switched-off module is
  skipped by the pipeline (`enabled = False`).
- Values marked ↻ change the cached steps 0–3: press **Rebuild steps 0–3** (≈ 12 s).
- A dot after a value marks a change against the default.

Keyboard (physical keys – works with any keyboard layout, also Ukrainian):

| Key | Action |
|---|---|
| `A` / `D` | previous / next frame (`Shift`: 10 frames) |
| `Space` | play / pause |
| `↑` / `↓` | select the previous / next parameter (highlighted) |
| `←` / `→` | change the selected parameter by one step (`Shift`: 10 steps) |
| `R` | reset the selected parameter to its default |

The mouse still works: frame slider, frame number field, key-frame buttons
(0, 3, 6, 100, 110, 118, 137, 164, 175, 200), click on a parameter to select it.

### Presets (versions of the settings)

- **Save**: enter a name (and an optional note), press *Save* →
  `presets/<name>.json` (values, date, note). An existing name asks before
  it is overwritten. Names: letters, digits, space, `-`, `_`, `.`.
- **Load**: choose a preset in the list (newest first, with date) → *Load*.
- The header shows the active preset and "(modified)" after changes.
- **Reset all to defaults** returns to the values in the code (= `recommended`).
- Loading a preset first resets all values to the defaults, so values
  missing in an older preset do not come from the previous one.
- `presets/recommended.json` – the recommended start (WORKLOG entry 17).
  Use it as a base, change, save under a new name.
- Use a preset for the full run: `python main.py --params <name>`.
  Without `--params`, `main.py` uses the defaults in the code.
- The server listens on 127.0.0.1 only. Options: `--port`, `--no-browser`,
  `--warmup` (see below), `--start` / `--stop`.

How a single frame can be re-rendered although `ToneMap` keeps state
between frames: before the frame, the back modules are "warmed up" on the
40 previous frames (`ToneMap.warmup` updates only its range and histogram).
Difference to a full run: on average 0.1–0.4 grey levels, max 1–4 levels.
During playback the state simply continues.

To make a partner's module tunable: add a `Param("StageName.attribute", min,
max, step, "Label", "help")` to `TUNABLE` in `main.py`. Optionally give the
module a `warmup(frame)` method if it keeps state and can update it cheaply.

## Run

All commands are run from the `project/` folder.

| Command | What it does |
|---|---|
| `python main.py` | runs the pipeline and opens the comparison window: pipeline (left) vs. naive `script.py` (right) |
| `python main.py --start 95 --stop 130` | only frames 95–129 (e.g. the cold sky event) |
| `python main.py --scale 0.75` | smaller window (it is 1286 px wide at scale 1) |
| `python main.py --no-show` | batch run without a window; key frames are saved side by side to `debug/compare/` |
| `python main.py --frames 0,100,110` | which frames are saved as previews (`key`, `all` or a list) |
| `python main.py --nuc-fit-step 0` | no NUC calibration before the run (the offset is learned during the run only) |
| `python main.py --naive` | only the naive baseline, prints its per-frame range and clipping |
| `python main.py --params NAME` | full run with a preset from `presets/` (without it: defaults in the code) |
| `python main.py --no-show --params recommended` | batch run → `output/normal_nuc_2__recommended.mp4` and `.mkv` (≈ 25 s) |
| `python main.py --no-video` | no video files |
| `python main.py --upscale 2` | 2× upscaling (1280 × 1024); default is 1 = sensor size 640 × 512 |
| `python main.py --params recommended --upscale-method nearest` | square sensor pixels instead of smooth Lanczos; file name gets `__nearest` |
| `python main.py --range-test` | only `output/range_test.mp4`, to check a player (see "Video output") |
| `python main.py other.bin --width 640 --height 512 --fps 30` | another recording |

Key frames (default previews): 0, 3, 6 (hot car), 100, 110, 118 (cold sky),
137, 175, 200 (small hot spots).

### Comparison window

| Key | Action |
|---|---|
| `space` | play / pause (starts paused) |
| `d` | next frame |
| `a` | previous frame – from a cache of the last 300 views; the pipeline is not run again |
| `s` | save the current view to `debug/compare/frame_NNNNN.png` |
| `q` | stop the run |

Each panel shows the share of black (0) and white (255) pixels. After the
last frame the window stays open for browsing back; `q` closes it.

Since step 6 the left panel is the real pipeline output (display domain).
If modules after step 3 are removed, it falls back to linear data with a
view-only stretch (p0.5–p99.5); this is then marked in its caption.

### Debug output

| Folder | Content |
|---|---|
| `debug/00_decode/` | key frames after step 0 |
| `debug/01_badpixels/` | after step 1 |
| `debug/02_destripe/` | after step 2 (residual NUC + destriping) |
| `debug/03_denoise/` | after step 3 (temporal + spatial denoising) |
| `debug/06_tonemap/` | after steps 5–6 (tone mapping, display output) |
| `debug/07_detail/` | after step 7 (detail enhancement) |
| `debug/09_output/` | final 8-bit frames (after steps 8–9) |
| `output/` | video files (step 10) |
| `debug/compare/` | side-by-side views (batch run or key `s`) |

Open the same frame number in two folders to see what a step changed.

## Structure

```
main.py                  describes the pipeline (build_front, build_back, TUNABLE) and runs it
tuner.py                 interactive tuner in the browser (standard library server)
presets/                 saved versions of the settings (created by the tuner; main.py --params NAME)
thermal/tuning.py        TuningSession (cache of steps 0–3, re-render of steps 5–7), Param, params I/O
thermal/frame.py         FrameData – the shared data container
thermal/stage.py         Stage – base class of a module; ArrayStage – wrapper for plain functions
thermal/pipeline.py      Pipeline – ordered list of modules; StopPipeline – stop a run early
thermal/sources.py       RawFileSource – reads frames from a .bin file (memory-mapped)
thermal/utils.py         shared helpers (noise estimate, blur, neighbour max/min)
thermal/stages/          one file and one class per processing step
  decode.py              Decode        – step 0
  shutter.py             ShutterGuard  – step 0b (shutter / FFC frames); ShutterHold – shows the last good image
  badpixels.py           BadPixels     – step 1
  nuc.py                 ResidualNUC   – step 2a
  destripe.py            Destripe      – step 2b
  temporal.py            TemporalDenoise – step 3a
  spatial.py             SpatialDenoise  – step 3b
  base_detail.py         BaseDetailSplit – step 5
  tonemap.py             ToneMap         – step 6 (switches: use_plateau, use_knee, temporal)
  clahe.py               LocalContrast   – optional CLAHE (off by default)
  tophat.py              CompactHotBoost – step 6c, white top-hat for compact hot objects (off by default)
  hotspot.py             HotSpotContrast – optional local contrast inside hot objects (off by default)
  multiscale.py          MultiScaleDetail – step 7b, multi-scale detail (off by default; see preset recommended-multiscale)
  upscale.py             Upscale         – step 8
  quantise.py            Quantise        – step 9
  video.py               VideoWriter     – step 10 (+ write_range_test)
  detail.py              DetailEnhance   – step 7
  naive.py               NaiveAGC      – baseline from script.py (comparison only)
  preview.py             Preview       – save/show frames after any step, frame is not changed
  compare.py             Compare       – live side-by-side view: pipeline vs. a reference
```

## Modules

All parameters are given to the constructor; each class has a full
docstring (`help(BadPixels)`).

### Ideas taken from Michael's `thermal_isp` (switchable)

All of them use only the current frame and state from past frames
(streaming). Each one has a switch in the tuner (marked "(M)"); all are off
by default except `ShutterGuard`, so the old presets give the same result.
Preset **`michael-features`** = `recommended-multiscale` with all of them on.

| Module / option | What it does | Default |
|---|---|---|
| `ShutterGuard` (step 0b) + `ShutterHold` | Flat frame (spatial std < 0.35 × median of the last 16) = shutter/FFC: `meta["frozen"]`, all stateful modules are skipped, the last good image is shown (max 30 frames in a row). Modules that run on frozen frames declare `run_when_frozen = True` | on |
| `Destripe(temporal_alpha=…)` | Column/row offsets smoothed over time (Michael: 0.1). 0 = per frame | 0 |
| `ToneMap(anchor_black=True)` | Black point moves at most `lo_slew` DN/frame (1.5). Fast (EMA) only on a real scene change: > `fast_frac` (15 %) of the frame colder than the black point, or the bottom `fast_gap` (25 %) of the range empty. The cold sky (≤ 4.5 % of the frame here) goes into the soft foot instead of moving the scale | off |
| `ToneMap(white_hold=True)` | Top of the range = peak hold of `p_high` (up `hi_attack` 0.15/frame, down `hi_decay` 0.15 DN/frame) | off |
| `ToneMap(min_span_dn=…)` | Range never narrower than this (Michael 1000 → milky here; 500 in the preset) | 0 |
| `ToneMap(shadow_lift=…)` | Michael's lift of dark tones (0.12) | 0 |
| `ToneMap(dde=True)` | DDE detail: own gain instead of the tone-curve slope; noise gate from the noise map; adaptive part equalises local detail contrast (`dde_adapt`, `dde_target`, `dde_max`); warm detail × `warm_boost` (Michael 1.5); soft limit | off |
| `CompactHotBoost` (step 6c) | White top-hat (opening 7×7 at half res ≈ 14 px): compact objects hotter than their surroundings by > `threshold_dn` get brighter; `soft=True` approaches white without clipping, `soft=False` = Michael (add + clip) | off |

Measured (all 275 frames, front cached, `--nuc-fit-step 0`): grey level of
the same temperature (7372 DN) in frames 98 → 122 when the sky enters:
`recommended-multiscale` 75 → 115; with `anchor_black` + `white_hold`
65 → 67; `michael-features` 66 → 72. Largest frame-to-frame jump of the mean:
3.3 → 1.3 levels. Lone hot spot (frame 100), brightest value: 181 without
top-hat, 215 soft, 238 hard. Cost: `CompactHotBoost` ≈ 3 ms (soft), DDE
≈ +3 ms in `ToneMap`.

### Step 0 – `Decode(bit_depth=14, on_overflow="mask", keep_raw=True, stats=True)`

Masks the raw value to `bit_depth` bits, converts to float32, sets
`domain = LINEAR`. `on_overflow`: `"mask"`, `"warn"` or `"error"` if pixels
use higher bits. Keep `keep_raw=True` when `Compare` is used.

### Step 1 – `BadPixels(k=8, k_replace=15, static_mask=None, learn=True, ...)`

- Impulse test: excess of a pixel over the max (or below the min) of its 8
  neighbours, in units of the noise std.
- Excess > `k_replace` → replaced in this frame (blinking pixels).
- Excess > `k` in most recent frames, or no change while the scene moves →
  added to the permanent mask (hot / dead / stuck pixels).
- Replacement: median of valid neighbours (3×3, then 5×5).
- `save_mask(path)` stores the mask; reuse it with `static_mask="path.npy"`.
- Writes `masks["bad_pixels"]` and `meta["noise_sigma"]` (used by later steps).

Results on this recording: no bad pixels (0 detections in 275 frames). Thin
scene details (sky in 1-px gaps between leaves) reach 8–12 σ and are kept,
which is why `k_replace` is 15. Synthetic test (20 dead, 20 hot, 20 stuck,
20 blinking): all found, no false detections, replacement error 5.8 DN on
average (≈ noise level).

### Step 2a – `ResidualNUC(alpha=0.02, blur_sigma=2, clip_k=5, edge_k=3, motion_k=3, adapt=True)`

Removes the fixed per-pixel offset pattern left after the factory NUC.
Scene-based: the pattern stays at the same sensor pixel, the scene moves.

- `fit(images)` – calibration on many frames before the run (`main.py`
  uses every 3rd frame, ~2 s). `reset()` returns to the fitted state.
- During the run the offset keeps adapting, only when the scene moves
  (motion gate) and with less weight at strong edges (no burn-in).
- `save(path)` / `load(path)` for the offset map.

Results: the offset pattern is 4.6 DN std (up to ±28 DN), stronger than the
column stripes. Offset maps learned on the first and the second half of the
recording independently correlate at 0.95 (−0.03 if shifted by 1 px), so the
module learns the sensor, not the scene. Static pattern after correction:
4.60 → 0.74 DN.

### Step 2b – `Destripe(columns=True, rows=True, width=9, clip_k=2, edge_k=0.5, ...)`

Removes per-frame column and row offsets (read-out noise that changes from
frame to frame). Pixels on strong scene edges across the stripe direction
are not used, so long horizontal scene edges (tree line) are not taken as
stripes. Removes on average 1.0 DN (columns) and 1.3 DN (rows) per frame.
Updates `meta["noise_sigma"]` on the corrected frame (≈ 4.1 DN; before step 2
the fixed pattern was counted as noise, ≈ 6 DN).

### Step 3a – `TemporalDenoise(max_frames=8, window=5, flow_preset="medium", interp="cubic", ...)`

Motion-compensated recursive filter:

1. dense optical flow (OpenCV DIS) from the current frame to the previous output;
2. warp the previous output and its per-pixel frame count onto the current frame;
3. match test – where the warped history does not fit (occlusion, parallax,
   flow error, moving object), it is not used: a local test (mean of d² in
   a 5 × 5 box) and a pixel test (|d|), both relative to the expected noise;
4. running mean with memory `max_frames`.

Writes `layers["noise_sigma_map"]` (remaining noise per pixel),
`layers["denoise.frames"]` (frames averaged per pixel; view it with
`Preview(tag="frames", source="denoise.frames")`), `meta["noise_sigma"]`.

### Step 3b – `SpatialDenoise(method="guided", strength=1.5, radius=2)`

Self-guided filter; its strength follows `noise_sigma_map`, so pixels where
the temporal average was rejected (new content at the border, moving
objects) are cleaned more. `method="bilateral"` is the alternative.

Results of step 3:

| Test | Result |
|---|---|
| Synthetic sequence (real frame + known drone-like motion + moving hot object + noise σ 6) | error vs. truth 6.0 → 2.0 DN; hot object error not larger than in the input (no smear) |
| Variants on the same test | cubic warp 2.19 vs. linear 2.75 DN; memory 8 frames 2.19 vs. 12 frames 2.43 DN; spatial only (no temporal) 3.75–4.71 DN |
| Real recording | noise ≈ 4.1 → ≈ 1.5 DN; history rejected for ≈ 21 % of pixels (moving objects, parallax, motion blur) |

### `Preview(tag, frames=None, source="image", save=True, save_npy=False, show=False, ...)`

Saves/shows the frame at its position in the pipeline. `source` can be a
layer or a mask name, e.g. `Preview(tag="mask", source="bad_pixels")`.

### `Compare(reference=None, label=..., reference_label=..., show=True, save=False, ...)`

Side-by-side window, see above. `reference` is any module or `Pipeline`
that takes a RAW frame; default `Pipeline([Decode(14), NaiveAGC(2, 98)])`.

### Step 4 – postponed

Measurement (WORKLOG entry 12): scene detail ends at ≈ 0.23–0.25 cycles/px,
half the Nyquist frequency – no aliasing, so super-resolution has nothing to
recover; deblurring has a small possible gain and a high risk of ringing.

### Step 5 – `BaseDetailSplit(radius=8, edge_dn=25)`

Guided filter: `layers["base"]` (large temperature differences) and
`layers["detail"]` (texture). Edges are kept by local *contrast*, not by
size, so small hot objects with high contrast stay in `base` and go
through the soft knee of step 6. `image` is not changed.

### Step 6 – `ToneMap(plateau=3, linear_mix=0.3, foot=0.05, knee=0.88, detail_gain=1, ...)`

Linear DN → display 0..1, white-hot:

1. histogram of `base` on a fixed DN grid, smoothed over time (no flicker);
2. scene range `[lo, hi]` = p0.2 / p99.8, smoothed over time;
3. main curve on `[lo, hi]` → `[foot, knee]`: plateau histogram
   equalisation mixed with a linear curve – grey levels follow how many
   pixels have a temperature, so a hot car or the cold sky does not
   compress the rest of the scene;
4. soft knee above `hi` (→ `[knee, 1)`) and foot below `lo` (→ `(0, foot]`),
   no clipping, hotter is always brighter;
5. detail added back with the local slope of the curve (`detail_gain`);
6. final soft limiter at 0.97 / 0.03 so added detail cannot reach 0 or 1.

Adaptation (AGC): the scene range grows fast (`alpha_grow` 0.15 ≈ 6
frames) when a hot or cold object enters and shrinks slowly (`alpha_shrink`
0.05 ≈ 20 frames) when it leaves, so brightness does not pump. Everything
above the range goes into the soft knee, so nothing is clipped while the
range catches up. **Peak hold** for the knee scale: a new hottest value is
followed at once (`peak_attack` 1.0), kept for `peak_hold` 12 frames, then
released slowly. Measured when hot spots enter (frames 168–174): inner
structure of the hot spots 2.6× stronger than with plain smoothing (grey
level std 2.1 vs. 0.8); pixels at 255 ≤ 0.002 %.

Switches: `use_plateau` (off = linear curve), `use_knee` (off = hard
clipping at the scene range, for comparison), `temporal` (off = range from
the current frame only, like a per-frame percentile stretch).

Results (all 275 frames, vs. naive `script.py`):

| Measure | Naive | Pipeline |
|---|---|---|
| Pixels at 255 / at 0 | 2.00 % / 2.11 % | 0.00 % / 0.00 % |
| Largest frame-to-frame jump of mean brightness | 16.8 levels | 2.3 levels |
| Flicker (std of mean brightness around its trend) | 1.85 | 0.41 |
| Hot car, frames 0–6: grey-level std inside (0 = flat blob) | 0.0 | 7.9 |
| Contrast of the ground when the sky enters (frames 80–94 → 105–120) | 79 → 46 (−42 %) | 63 → 50 (−21 %) |
| Hotter = brighter (Spearman, frame 100) | 1.000 | 1.000 |

Additional parameters: `gamma` (black level of dark and middle tones),
`white` (display value of the hottest pixels). `layers["tonemap.slope"]`
(display units per DN) is written for step 7.

### Step 7 – `DetailEnhance(gain=1.5, sigma=1.2, coring_k=2, max_amp=0.06)`

Noise-gated unsharp masking of the fine band `y - gaussian(y, 1.2)`
(≈ 0.1–0.25 cycles/px, the upper part of the range where the scene has
detail). Band values of the order of the remaining noise (noise map ×
tone-map slope) are suppressed, so the noise removed in step 3 does not
come back; the added amount is limited to ±0.06 (≈ 15 grey levels) against
halos; the addition is limited smoothly by the distance to 0 / 1, so detail
never pushes a pixel to 0 or 1 (and pixels clipped by `ToneMap` with
`use_knee=False` stay visible as clipped). Mid-size detail is controlled by
`ToneMap(detail_gain=...)`.

### Step 7b – `MultiScaleDetail(fine=0.6, mid=1.0, coarse=0.6, edge=0.06, coring_k=3, enabled=False)`

Alternative to `DetailEnhance`: detail boosted on three scales (difference
of Gaussians at 1 / 2.5 / 6 px). Per band: noise gate (no boost of values of
the size of the remaining noise) and edge protection (the boost falls off
for band values above `edge`, so strong edges get no halos). Mainly texture
of moderate contrast is raised.

| Variant (every 5th frame) | Coarse 0.02–0.06 | Mid 0.06–0.12 | Fine 0.12–0.22 | Noise band 0.3–0.5 | Overshoot at strong edges |
|---|---|---|---|---|---|
| no detail step | 1411 | 77 | 4.0 | 0.039 | 0 |
| `DetailEnhance` (recommended) | 1456 | 93 | 6.5 | 0.090 | 0.3 levels |
| **`MultiScaleDetail` 0.6 / 1.0 / 0.6** | **2182** | **180** | **9.9** | **0.064** | 1.4 levels |
| `MultiScaleDetail` 1.0 / 1.5 / 1.0 | 2677 | 247 | 13.6 | 0.082 | 3.0 levels |

To compare: load the presets `recommended` and `recommended-multiscale`
in the tuner, or watch `output/normal_nuc_2__recommended.mp4` and
`…__recommended-multiscale.mp4`.

### Optional – `HotSpotContrast(gain=2, sigma=3, edge=0.03, width=0.05, enabled=False)`

Local contrast only in the knee zone (hot objects): inner structure =
image − guided filter, boosted with a weight that fades in above the knee.
The rest of the scene is not changed (measured: 0.00 grey levels). Off by
default, because on this recording it has little to show: the bright strip
in frame 100 is not in the knee at all, and the strip in frame 164 is an
almost even hot surface (72 DN inside, already shown with 37 grey levels).
For these strips use `White level`, `Knee start`, `gamma`, `Linear mix`.
Useful for hot objects with a wide inner temperature range (e.g. the hot
spots around frame 175: 364 DN in ≈ 42 grey levels).

### Optional – `LocalContrast(clip_limit=2, tiles=8, amount=0.5, enabled=False)`

CLAHE on a 16-bit copy of the display image, blended with the global tone
mapping. Off by default: it can make a cooler area brighter than a warmer
one (the temperature order is lost between tiles), it gives halos around
hot objects and amplifies noise in flat areas. Here for comparison and
mild use.

Results (every 10th frame, same denoised input):

| Variant | Detail energy 0.10–0.22 cycles/px | Noise (grey levels) | Ground p5 (darkness) |
|---|---|---|---|
| Naive p2/p98 | 13.3 | 0.80 | 41 |
| Step 6 only | 11.4 | 0.70 | 51 |
| + `DetailEnhance(gain=1)` | 19.3 | 0.96 | 50 |
| `ToneMap(detail_gain=1.5, gamma=1.3)` + `DetailEnhance(gain=1.5)` – first choice | 30.4 | 1.16 | 35 |

Current defaults (= `presets/recommended.json`, WORKLOG entry 17):
`ToneMap(linear_mix=0.6, knee=0.75, gamma=1.3, detail_gain=1.5)` +
`DetailEnhance(gain=1.0, coring_k=3.0)` – less "painted" look in blurred areas.

### Step 8 – `Upscale(scale=2, method="lanczos")`

Off by default (`--upscale 1`, video = sensor size 640 × 512). With `--upscale 2`:
640 × 512 → 1280 × 1024 on the float display image, before quantisation.
Adds no detail (the scene has detail only up to ≈ 0.23 cycles/px), but no
pixel blocks either.

### Step 9 – `Quantise(dither="blue", amplitude=1.0)`

Float 0..1 → uint8 with a fixed blue-noise threshold map (fine, even grain
of ±0.5 grey level, not moving between frames). Test on a very flat ramp
(6 grey levels over 1280 px):

| Dither | Error of the local mean | Largest visible step |
|---|---|---|
| none (rounding) | 0.24 levels | 0.11 |
| Bayer | 0.012 | 0.056 |
| **blue noise** | **0.015** | **0.028** |

### Step 10 – `VideoWriter(path, fmt="h264" | "ffv1", fps=25, crf=10, tune="grain")`

MP4 (H.264, crf 10, `-tune grain`) is the deliverable; the .mkv (FFV1) is a
lossless reference copy, skip it with `--no-mkv`. Measured vs the lossless
file (640 × 512): crf 14 default tuning 45.9 dB PSNR, 4.1 MB – fine dither
and texture smoothed away; crf 10 + grain 49.4 dB, 11.4 MB.

One writer per file; ffmpeg is fed raw 8-bit grey frames.

| File | Format | Use |
|---|---|---|
| `output/<recording>__<preset>.mp4` | H.264 High, full range 0..255 (`color_range=pc`), CRF 14, 25 fps, ≈ 8.4 Mbit/s | showing / viewing |
| `output/<recording>__<preset>.mkv` | FFV1, 8-bit grey, lossless | archive, analysis (identical to the pipeline frames) |

Checks: MKV frames are identical to the pipeline frames; MP4 differs by
0.6–0.7 grey levels on average (compression, mostly of the dither grain).

**Video output – choosing a player.** Full range (0..255) is marked in the
file, but not every player honours it. Open `output/range_test.mp4`
(`python main.py --range-test`): top = smooth ramp 0..255; bottom = 8
patches 0, 8, 16, 24 | 231, 239, 247, 255. In a correct player all four dark
patches look different and all four light patches look different. If
0 / 8 / 16 look the same black (or 239 / 247 / 255 the same white), that
player clips the range – use another one for the demo.

The frame rate of the sensor is not known; 25 fps is assumed (`--fps`).

## Tuning

Easiest: `python tuner.py` (see above). The defaults are in `build_back` in
`main.py`; a preset (`--params NAME`) overrides them. Look at the result with
`python main.py --params NAME` (frames 0–6 car, 95–130 sky, 160–170 fast turn, 169–212 hot spots).

| What you see | Knob |
|---|---|
| Ground / dark areas not black enough | `ToneMap(gamma=...)` up (1.3 → 1.5); or `foot` down (0.05 → 0.02) |
| Image too dark | `gamma` down |
| Hot objects glare (white strip in frames 100, 164) | `ToneMap(white=0.92)`; or `knee` down (0.88 → 0.82) for more grey levels inside hot objects |
| Not sharp enough | `DetailEnhance(gain=...)` up; `ToneMap(detail_gain=...)` up |
| Grainy / "painted" texture | `DetailEnhance(gain=...)` down or `coring_k` up (2 → 3) |
| Halos around hot objects | `DetailEnhance(max_amp=0.04)` |
| Large areas (sky, field) take too many grey levels | `ToneMap(plateau=...)` down (3 → 2) |
| Contrast should follow temperature more linearly | `ToneMap(linear_mix=...)` up (0.3 → 0.5) |
| Brightness adapts too slowly / too fast | tuner "Adapt: range grows / shrinks" (`alpha_grow`, `alpha_shrink`) |
| Tone curve changes too nervously | "Adapt: histogram" (`alpha_hist`) down |
| A new hot object looks flat in its first frames | "Peak hold: attack" up (1 = at once), "Peak hold: frames" |
| Scene gets darker when hot objects take a larger part of the frame | "Scene range top (pct)" up (99.8 → 99.95): more hot pixels go to the knee |

## Data model (`FrameData`)

| Field | Content |
|---|---|
| `index`, `timestamp` | position of the frame in the recording |
| `raw` | original uint16 frame |
| `image` | main working signal, float32 H×W |
| `domain` | `RAW` → `LINEAR` → `DISPLAY` → `OUTPUT` (what `image` holds now) |
| `layers` | extra images of the same frame, e.g. `"base"`, `"detail"` |
| `masks` | maps, e.g. `"bad_pixels"` |
| `meta` | scalars, e.g. `"noise_sigma"`, `"nuc.offset_std"` |

Rule for keys a module adds: `"<module>.<value>"` (e.g. `"badpix.dynamic"`).
Shared keys without a prefix are listed in `thermal.frame.KNOWN_KEYS`.

State between frames (history, running averages) stays inside the module object.

## Writing a module

```python
from thermal import Stage, FrameData, Domain

class MyDenoise(Stage):
    name = "MyDenoise"

    def __init__(self, strength: float = 1.0):  # all parameters here
        super().__init__()
        self.strength = strength
        self._prev = None                        # state between frames

    def process(self, frame: FrameData) -> FrameData:
        frame.require(Domain.LINEAR)             # check the domain you rely on
        sigma = frame.meta["noise_sigma"]        # set by BadPixels
        ...                                      # change frame.image
        frame.meta["mydenoise.sigma_out"] = 1.2
        return frame                             # or None while buffering

    def reset(self):                             # clear state
        self._prev = None

    def flush(self):                             # frames still held at the end
        return []
```

Inheriting from `Stage` is optional: any object with `process(frame)` works.
A plain function on a numpy array: `ArrayStage(func, name="...")`.
To stop a run from a module, raise `thermal.StopPipeline`.

## Plugging modules in

In `main.py`, `build_pipeline` lists the modules in order. Or in code:

```python
pipe = Pipeline([Decode(bit_depth=14), BadPixels()])
pipe.add(PartnerModule(param=1))                       # at the end
pipe.insert_after("BadPixels", Preview(tag="01_badpixels", frames=[0, 100]))
pipe.replace("PartnerModule", OtherModule())
pipe.remove("PartnerModule")
pipe.add(Compare(reference=PartnerPipeline(), reference_label="partner"))
for frame in pipe.run(source):
    ...
print(pipe.timing_report())
```

## Status

| Step | Module | Status |
|---|---|---|
| 0 | `Decode` – mask to 14 bits, float32 | done |
| 0b | `ShutterGuard` / `ShutterHold` – shutter / FFC frames (from Michael) | done |
| 1 | `BadPixels` – detection and replacement | done |
| 2 | `ResidualNUC` + `Destripe` – fixed pattern and stripes | done |
| 3 | `TemporalDenoise` + `SpatialDenoise` – motion-compensated denoising | done |
| 4 | super-resolution / deblurring | postponed (WORKLOG entry 12) |
| 5 | `BaseDetailSplit` | done |
| 6 | `ToneMap` – hot-object-aware AGC, plateau HE, soft knee | done |
| 7 | `DetailEnhance` – noise-gated sharpening | done |
| 6+ | `LocalContrast` – optional CLAHE (off by default) | done |
| 6 | `ToneMap` options from Michael (anchored black point, white peak hold, min. range, shadow lift) + DDE detail mode | done |
| 6c | `CompactHotBoost` – white top-hat for compact hot objects (from Michael) | done |
| 6b | `HotSpotContrast` – local contrast inside hot objects (off by default) | done |
| 7b | `MultiScaleDetail` – multi-scale detail (off by default; preset `recommended-multiscale`) | done |
| 8 | `Upscale` – Lanczos 2× | done |
| 9 | `Quantise` – 8 bit with blue-noise dither | done |
| 10 | `VideoWriter` – H.264 .mp4 (full range) + FFV1 .mkv (lossless) | done |
| – | `Preview`, `Compare`, `NaiveAGC` – tools | done |
| – | `tuner.py` – interactive tuning in the browser | done |

Speed (Apple silicon Mac, 275 frames): steps 0–10 with both video files
≈ 23 s (≈ 12 fps); steps 0–7 alone ≈ 19 s, of which
`TemporalDenoise` ≈ 7 s (optical flow; `flow_preset="fast"` is quicker) and
`ToneMap` ≈ 4.7 s (not optimised yet).
