# Work log

What was done, why, and what came out of it. Newest work is at the bottom.
Each entry has the same parts: **Goal**, **What was done**, **Result**, **Files**.

Recording: `normal_nuc_2.bin` · 275 frames · 640 × 512 · 14-bit in little-endian uint16.

## Overview

| # | Date | Entry | Status |
|---|---|---|---|
| 1 | 2026-11-13 | Analysis of the raw data | done |
| 2 | 2026-11-13 | Pipeline description (docs) | done |
| 3 | 2026-11-13 | Analysis of `script.py` | done |
| 4 | 2026-11-13 | Analysis of frame 100 | done |
| 5 | 2026-11-13 | TODO list in the docs | done |
| 6 | 2026-11-13 | Project framework + step 0 `Decode` | done |
| 7 | 2026-11-13 | Tools: `Preview`, `NaiveAGC` | done |
| 8 | 2026-11-13 | Tool: `Compare` (live side-by-side window) | done |
| 9 | 2026-11-13 | Step 1 `BadPixels` | done |
| 10 | 2026-11-13 | Step 2 `ResidualNUC` + `Destripe` | done |
| 11 | 2026-11-13 | Step 3 `TemporalDenoise` + `SpatialDenoise` | done |
| 12 | 2026-11-13 | Decision on step 4 (super-resolution / deblurring) + measurement | postponed |
| 13 | 2026-11-13 | Steps 5–6 `BaseDetailSplit` + `ToneMap` | done |
| 14 | 2026-11-14 | Feedback on step 6, frame 164 check, step 7 `DetailEnhance` | done |
| 15 | 2026-11-14 | Interactive tuner in the browser | done |
| 16 | 2026-11-14 | Tuner: presets (versions of the settings), keyboard control | done |
| 17 | 2026-11-14 | Recommended preset, algorithm switches, CLAHE, mapping of common terms | done |
| 18 | 2026-11-14 | Steps 8–10: upscale, dither, video files (full range) | done |
| 19 | 2026-11-14 | Adaptation (AGC) check, peak hold, adaptation sliders; tuner Load bug fixed | done |
| 20 | 2026-11-14 | Multi-scale detail and local contrast for hot objects (new modules) | done |
| 21 | 2026-11-14 | Ideas from Michael's `thermal_isp` (shutter guard, anchored AGC, top-hat, destripe smoothing) + DDE detail mode | done |
| – | – | Open items and next steps | see the end |

---

## 1. Analysis of the raw data

**Goal:** know the data before choosing algorithms.

**What was done:** statistics per frame and over time, noise estimate,
search for bad pixels and stripes, visual check of key frames.

**Result:**

| Property | Value |
|---|---|
| Value range | 6158 … 8654 DN (only ≈ 2 500 of 16 384 codes used) |
| Typical frame range (p1…p99) | ≈ 500 DN |
| Polarity | higher DN = hotter → white-hot without inversion |
| Noise | σ ≈ 6 DN |
| Bad pixels | none |
| Scene | drone over an orchard, strong motion, motion blur |

Key events:

| Frames | Event |
|---|---|
| 0–6 | hot car, lower left |
| 100–125 | cold sky enters at the top |
| 169–184, 196–212 | small hot spots, top right |

---

## 2. Pipeline description

**Goal:** a step-by-step plan, with algorithms, and the answer to the
question "is there a physically justified order of steps?".

**What was done:** wrote the description in English and in Ukrainian.

**Result:**

- The order **is** justified by physics (signal model
  `raw = G·L(T)⊛PSF + O + n`): first the corrections in the linear domain
  (bad pixels → fixed pattern → noise), then the non-linear display steps
  (tone mapping → detail → upscaling → 8-bit → encoding).
- Two key rules: remove the fixed pattern before temporal denoising, and
  both before any contrast stretch; never clip hot objects hard – use a
  smoothed scene range with a soft-knee headroom (≈ 217–255).

**Files:** `../docs/thermal_video_pipeline.md`, `../docs/thermal_video_pipeline.uk.md`

---

## 3. Analysis of `script.py`

**Goal:** understand what the existing viewer does with the data.

**Result:** it is a viewer only, it does not write a video. Per frame:
read → percentiles 2 % / 98 % of this frame → hard clipping → scale to
0–255 → truncate to uint8 → draw the frame number → show.

Problems: hot objects become flat white blobs; flicker (the range changes
every frame); a brightness jump when the sky enters; noise and stripes are
amplified.

---

## 4. Analysis of frame 100

**Goal:** check whether all objects in frame 100 are shown correctly.

**Result:** the **data** is correct (no glitch, polarity consistent, smooth
motion); the **display** by `script.py` is not:

| Object | Problem with the p2/p98 mapping |
|---|---|
| Small hot spot (≈ 10 × 6 px at row 215, column 517, +390 DN) | clipped to a flat white blob |
| Warm strip below the trees (rows ≈ 150–170) | 8.3 % of its pixels at 255 |
| Cold tail (sky in gaps, rows 0–18) | clipped to solid black |
| Tree belt (≈ 23 % of the frame) | pale, levels 160–247 |
| Noise | σ 6 DN becomes ≈ 3 grey levels |

---

## 5. TODO list in the docs

**Goal:** record the gaps in the plan that frame 100 showed (as TODO, not
as changes to the text).

**What was added:** small hot objects bypass the soft knee (they go into the
detail layer); limit `g · detail` everywhere; a formula for the cold "foot";
recommend plateau histogram equalisation; a check for small hot objects.

**Files:** TODO section at the end of both docs.

---

## 6. Project framework + step 0 `Decode`

**Goal:** a modular structure for team work: every step is a class, its
object is created in `main.py` with parameters, partners plug in their
modules by one method call.

**What was done:**

| Part | Purpose |
|---|---|
| `FrameData` | shared container: `image` (float32), `raw`, `domain`, plus open dictionaries `layers`, `masks`, `meta` for data that only some algorithms need |
| `Stage` | base class: `process(frame)`, `reset()`, `flush()` |
| `Pipeline` | ordered list: `add`, `insert_after`, `insert_before`, `replace`, `remove`, `run` |
| `ArrayStage` | wraps a plain function on a numpy array |
| `RawFileSource` | reads the .bin file frame by frame (memory-mapped) |
| `Decode` | step 0: mask to 14 bits, convert to float32, `domain = LINEAR` |

Rules: key names `"<module>.<value>"`; temporal history stays inside the
module; a partner module does not have to inherit from `Stage` – any object
with `process(frame)` works.

**Result:** 275 frames in 0.09 s; output identical to a direct `np.fromfile`
read; no pixels with bits above bit 13.

**Files:** `thermal/frame.py`, `thermal/stage.py`, `thermal/pipeline.py`,
`thermal/sources.py`, `thermal/stages/decode.py`, `main.py`, `README.md`

---

## 7. Tools: `Preview` and `NaiveAGC`

**Goal:** see the data after any step and have a baseline for comparison.

**What was done:**

- `Preview` – saves (or shows) frames at its position in the pipeline;
  does not change the frame. Can show layers and masks too.
- `NaiveAGC` – the processing of `script.py` as a module (without the text
  overlay).

**Result:** `NaiveAGC` is pixel-identical to `script.py` (checked on frame
100). Its numbers already show the problem: in frame 100 the lower limit is
7110 DN, in frame 110 it is 6631 DN (sky enters) → visible brightness jump;
≈ 2 % of pixels black and ≈ 2 % white in every frame.

**Files:** `thermal/stages/preview.py`, `thermal/stages/naive.py`

---

## 8. Tool: `Compare`

**Goal:** the evaluation is subjective, so the pipeline and the naive
result must be visible together, live, frame by frame.

**What was done:** one window, left = pipeline, right = naive `script.py`.
Keys: `space` play/pause, `d` next, `a` back (cache of 300 views, no
re-processing), `s` save view, `q` stop. Each panel shows the share of
black and white pixels. Batch mode (`--no-show`) saves key frames side by
side. The reference can be any module or pipeline (e.g. a partner's).
Added `StopPipeline` so a module can stop a run cleanly.

**Result:** navigation tested with simulated key presses.

**Files:** `thermal/stages/compare.py`, `thermal/pipeline.py`

---

## 9. Step 1 `BadPixels`

**Goal:** find and replace dead, stuck, hot/cold and blinking pixels
before any spatial filter.

**What was done:**

- Impulse test: excess of a pixel over the max (or below the min) of its 8
  neighbours, in units of the noise σ.
- Two thresholds: > 15 σ → replaced at once; > 8 σ in most recent frames →
  permanent mask. Stuck pixels: no change while the scene moves.
- Replacement: median of valid neighbours (3 × 3, then 5 × 5).
- Writes `masks["bad_pixels"]` and `meta["noise_sigma"]`.

**Result:**

| Test | Outcome |
|---|---|
| Synthetic: 20 dead, 20 hot, 20 stuck, 20 blinking | all found, 0 false detections, replacement error 5.8 DN (≈ noise) |
| Real recording | 0 bad pixels |

Finding: with one threshold of 8 σ there were 37 detections on the real
data. All were scene details (sky in 1-px gaps between leaves, 8–12 σ,
frames 102–123), never twice at the same place. This is why the immediate
replacement threshold is 15 σ.

**Files:** `thermal/stages/badpixels.py`, `thermal/utils.py`

---

## 10. Step 2 `ResidualNUC` + `Destripe`

**Goal:** remove the fixed pattern and the stripes that the factory NUC
left, without touching the scene.

**What was done:**

- `ResidualNUC` (2a): per-pixel offset learned from the moving scene
  (scene-based NUC). Calibrated before the run on every 3rd frame (≈ 2 s),
  then keeps adapting. Protection: no update when the scene does not move,
  less weight at strong edges, clipping.
- `Destripe` (2b): per-frame column and row offsets; pixels on strong
  scene edges are not used.

**Result:**

| Measurement | Value |
|---|---|
| Fixed per-pixel pattern before → after | 4.60 → 0.74 DN |
| Same map learned on 1st half vs 2nd half of the recording | correlation 0.95 (−0.03 if shifted by 1 px) → the module learns the sensor, not the scene |
| Stripes removed per frame | columns 1.0 DN, rows 1.3 DN |
| Leak of scene edges into row correction | 0.14 → 0.04 after edge exclusion |

Findings:

- The main residual noise is **not** stripes but a per-pixel pattern of
  4.6 DN (up to ±28 DN). It is the fine "mesh" visible on the right panel
  of `Compare`.
- Clipping the residual at 2.5 σ under-estimated this pattern (770 pixels
  above 14 DN) → `clip_k = 5`.
- Per-frame row noise is real: in differences of consecutive frames, row
  offsets from the left and the right half agree (corr. 0.43).
- Without edge exclusion, the tree line (frames 100–125) partly counted as
  a stripe → `edge_k = 0.5`.

**Files:** `thermal/stages/nuc.py`, `thermal/stages/destripe.py`, `main.py`

**Speed:** steps 0–2 together ≈ 4.5 s for 275 frames (≈ 60 fps).

---

## 11. Step 3 `TemporalDenoise` + `SpatialDenoise`

**Goal:** remove the random noise without smearing moving objects and
without ghosts from the strong drone motion.

**What was done:**

- `TemporalDenoise` (3a): optical flow (OpenCV DIS) from the current frame
  to the previous output → warp the history onto the current frame → match
  test (local 5 × 5 test and per-pixel test against the expected noise) →
  running mean over up to 8 frames. Where the history does not fit, the
  current frame is used. Tracks the remaining noise per pixel
  (`layers["noise_sigma_map"]`).
- `SpatialDenoise` (3b): self-guided filter whose strength follows the
  noise map – more cleaning where the temporal average was rejected.
- `Destripe` now estimates `noise_sigma` again after step 2.
- Noise estimate made 4× faster (median on every 2nd pixel, same value).

**How it was tested:** a synthetic sequence with a known clean image (a
real frame, moved like a drone: ≈ 7 px/frame + rotation, a moving hot
object of 6 × 10 px, +400 DN, noise σ 6), so the true error can be measured.

**Result:**

| Variant (synthetic test) | Error vs. truth |
|---|---|
| Input | 6.00 DN |
| Spatial only (guided / bilateral) | 4.71 / 3.75 DN |
| Temporal, linear warp | 2.75 DN |
| Temporal, memory 12 frames | 2.43 DN |
| Temporal, cubic warp, memory 8 | 2.19 DN |
| **Temporal + guided (strength 1.5) – chosen** | **2.00 DN** |

The moving hot object is not smeared: its error is not larger than in the
input. On the real recording: noise ≈ 4.1 → ≈ 1.5 DN; the history is
rejected for ≈ 21 % of pixels (moving objects, parallax, motion blur), and
those pixels get more spatial filtering. No ghosts or trails were seen
around the car (frames 0–6) or the moving hot spot (frames 95–105).

Findings:

- A first test with added noise suggested only ×1.5 reduction. It was
  misleading: the two runs took slightly different motion decisions, and
  that difference was counted as noise. The ground-truth test is the
  reliable one.
- The noise estimate of step 1 (≈ 6 DN) included the fixed pattern; the
  true random noise after step 2 is ≈ 4.1 DN. With the too high value the
  match test was too tolerant (10 % rejected instead of 21 %).
- Non-local means was not used: in OpenCV 5 (Python) it accepts only 8-bit
  input, which would lose the precision of the linear data.

**Files:** `thermal/stages/temporal.py`, `thermal/stages/spatial.py`,
`thermal/stages/destripe.py`, `thermal/utils.py`, `main.py`

**Speed:** steps 0–3 ≈ 12 s for 275 frames (≈ 22 fps); optical flow ≈ 7 s.

---

## 12. Decision on step 4: postponed

**Goal:** decide whether step 4 (multi-frame super-resolution or
deblurring) is worth doing now, based on the data.

### The trade-off

| | Do step 4 now | Postpone step 4 |
|---|---|---|
| Possible gain | sharper image: detail below pixel size (SR), less motion blur (deblurring) | – |
| Risks | ringing / halos around hot objects (deconvolution); noise amplified again; double edges and ghosts where sub-pixel registration fails (parallax of trees, moving objects – already ≈ 21 % of pixels fail the match test in step 3) | image stays as soft as the sensor delivers it |
| Effort | high: per-frame, per-region blur estimation; heavy computation | none |
| Output size 1280 × 1024 | – | still provided by step 8 (interpolation; adds no detail, but no artefacts either) |
| Main requirements (white-hot, no overexposure, hot object behaviour, no flicker) | not affected | handled by steps 5–6, which are needed to judge the result at all |

Physics reason for doubt: SR only helps when the image is **aliased**
(optics sharper than the pixel grid). At λ ≈ 10 µm the optics are
diffraction-limited, and the bolometer time constant and the drone motion
add more blur. Step 4 can be inserted later between steps 3 and 5 without
changes to other modules.

### Measurement

Frames after step 2 (before denoising), all 275 frames.

**1. Power spectrum (radially averaged):**

| Spatial frequency (cycles/px) | Signal / noise |
|---|---|
| 0.07 | ≈ 600 |
| 0.11 | ≈ 100 |
| 0.17 | ≈ 9 |
| 0.21 | ≈ 2 |
| 0.25 | ≈ 0.3 |
| 0.29 … 0.5 (Nyquist) | flat – noise only |

→ Scene detail ends at ≈ 0.23–0.25 cycles/px, about **half the Nyquist
frequency**. There is **no aliasing**, so multi-frame SR has nothing to
recover. The limit is blur, not the number of pixels. A 2× upscale by
interpolation (step 8) loses no information.

**2. Sharpness per frame** (energy in 0.15–0.30 cycles/px relative to 0.03–0.08):

| Measure | Value |
|---|---|
| Variation over the recording (max / min) | ≈ 6× |
| Frames below 50 % of the median sharpness | 2 of 274 |
| Correlation with motion speed (Spearman) | −0.41 – faster motion, softer frame (moderate) |
| Direction (x / y energy, median) | 0.42 – blur mostly horizontal, i.e. from motion |

### Decision

**Step 4 is postponed; we go to steps 5–6.**

- SR: no – no aliasing, nothing to gain.
- Deblurring: the only possible real gain, but the signal is above the
  noise only up to ≈ 0.22 cycles/px, so the possible boost is small and the
  risk of ringing around hot objects is high.
- Cheap alternatives with low risk, for later:
  - sharpness weighting in `TemporalDenoise` (blurred frames get a lower
    weight) – small gain, only 2 frames are strongly blurred;
  - moderate detail boost limited to < 0.22 cycles/px in step 7 (detail
    enhancement), which is planned anyway.

**Side finding:** the flat noise floor of the spectrum gives a white-noise
σ ≈ 3.4 DN, while the Laplacian estimate gives ≈ 4.1 DN – the Laplacian
also picks up some scene detail around 0.2 cycles/px. The match test in
step 3 is therefore slightly tolerant. See open items.

---

## 13. Steps 5–6 `BaseDetailSplit` + `ToneMap`

**Goal:** map the clean linear data to the display range: white-hot, no
overexposure, hot objects bright but with inner detail, the scene keeps its
contrast when a hot or cold object enters, no flicker.

**What was done:**

- `BaseDetailSplit` (5): guided filter → `base` + `detail`. Edges are kept
  by local contrast, not by size, so small hot objects with high contrast
  stay in `base`.
- `ToneMap` (6):
  1. histogram of `base` on a fixed DN grid, smoothed over time;
  2. scene range p0.2 / p99.8, smoothed over time (faster when it grows);
  3. main curve: plateau histogram equalisation (70 %) + linear (30 %);
  4. soft knee above the range (display 0.88–1) and soft foot below it
     (0–0.05) – no clipping;
  5. detail added back with the local slope of the curve;
  6. final soft limiter at 0.97 / 0.03.
- The left panel of `Compare` now shows the real output.

**Result** (all 275 frames, same denoised input for all variants):

| Measure | Naive p2/p98 | Linear curve + knee | **Chosen (plateau + knee)** |
|---|---|---|---|
| Pixels at 255 / 0 | 2.00 % / 2.11 % | 0 % / 0 % | **0 % / 0 %** |
| Largest jump of mean brightness | 16.8 levels | 6.1 | **2.3** |
| Flicker | 1.85 | 0.66 | **0.41** |
| Car (frames 0–6): grey-level std inside | 0.0 (flat blob) | 11.5 | **7.9** |
| Small hot spot (frames 95–105): max / std | 255 / 36 | 252 / 35 | **252 / 44** |
| Ground contrast when the sky enters | −42 % | −48 % | **−21 %** |
| Hotter = brighter (Spearman) | 1.000 | 1.000 | **1.000** |

TODO items from entry 5 closed by this step:

- [x] small hot objects bypass the knee → the guided split keeps them in
  `base`; the final limiter catches what is left;
- [x] limit `g · detail` everywhere → final soft limiter;
- [x] cold "foot" → exponential foot into 0–0.05;
- [x] plateau HE as the main curve.

Findings:

- Plateau HE is the main reason the ground keeps its contrast when the sky
  enters (−21 % instead of −48 % with a linear curve).
- The first version applied the final limiter at the knee (0.88), which
  compressed the knee twice; hot objects did not reach white. Moved to 0.97.
- Weak point: mid-tones are a bit flatter (greyer) than in the naive view.
  To be handled in step 7 and by tuning `plateau`, `linear_mix`,
  `detail_gain` while looking at `Compare`.

**Files:** `thermal/stages/base_detail.py`, `thermal/stages/tonemap.py`, `main.py`

**Speed:** steps 0–6 ≈ 18 s for 275 frames (≈ 15 fps); `ToneMap` ≈ 4.7 s.

---

## 14. Feedback on step 6, frame 164, step 7 `DetailEnhance`

**Feedback after step 6:** grey levels look acceptable, but the image is
not sharp, the ground is not black enough, and the white strip (frame 100)
and the strip in frame 164 are still quite bright. Hard to judge the grey
levels without sharpness.

**Checks:**

| Question | Answer |
|---|---|
| The rectangular shapes next to the strip in frame 164 – ghosts of the temporal filter? | **No.** They are in the raw frames 162–165 as well: real structures. Frame 164 is a fast turn with strong motion blur. |
| Why is the strip in frame 164 white? | It is the hottest object in the frame (≈ 0.3 % of pixels in the knee). In white-hot it must be the brightest; *how* bright is set by `knee` / `white`. |
| Why is the strip below the trees in frame 100 bright? | It is not in the knee. The main curve gives the top of the histogram (tree belt + strip) the light grey levels. |
| Why is the ground not black? | 5th percentile of the lower third = 31–45 levels: the foot (0.05) and the curve lift dark tones. |

**What was done:**

- `ToneMap`: new knobs `gamma` (black level of dark and middle tones,
  temperature order unchanged) and `white` (white level); it now writes
  `layers["tonemap.slope"]` for step 7.
- `DetailEnhance` (7): noise-gated unsharp masking of the fine band
  (σ = 1.2 px, ≈ 0.1–0.25 cycles/px). Noise gate from the noise map ×
  tone-map slope; added amount limited to ±0.06 against halos; final soft
  limiter.
- Tuning table in README.

**Result** (every 10th frame, same denoised input):

| Variant | Detail energy 0.10–0.22 cycles/px | Noise (grey levels) | Ground p5 |
|---|---|---|---|
| Naive p2/p98 | 13.3 | 0.80 | 41 |
| Step 6 only | 11.4 | 0.70 | 51 |
| + `DetailEnhance(gain=1)` | 19.3 | 0.96 | 50 |
| + `DetailEnhance(gain=2)` | 29.4 | 1.11 | 50 |
| `detail_gain=1.5` + `DetailEnhance(1.5)` | 33.3 | 1.17 | 48 |
| **`detail_gain=1.5`, `gamma=1.3` + `DetailEnhance(1.5)` – chosen** | **30.4** | **1.16** | **35** |

Chosen variant: ≈ 2.3× more fine detail than the naive view and ≈ 2.7×
more than after step 6; noise still ≈ 1 grey level; darker ground; the
strips in frames 100 and 164 dominate less. Still 0 % pixels at 255.

To watch: in areas with strong motion blur (lower part of frame 100,
frame 164) the sharpening also emphasises smeared streaks, which can look
"painted". If it disturbs: lower `DetailEnhance(gain)` or raise `coring_k`.

**Files:** `thermal/stages/detail.py`, `thermal/stages/tonemap.py`,
`main.py`, `README.md` (section "Tuning")

**Speed:** steps 0–7 ≈ 19 s for 275 frames (≈ 14 fps).

---

## 15. Interactive tuner in the browser

**Goal:** the smeared, "painted" texture in frame 100 disturbs a lot.
Tuning one parameter at a time through the chat is too slow. Needed: a
simple interface to change values on the fly and see the current frame,
also when moving forward and back.

**Design:**

- Problem: steps 2, 3 and 6 keep state between frames, so one frame cannot
  be processed alone.
- Solution: **steps 0–3** (heavy, stateful) run once for the whole
  recording and are cached in memory (≈ 12 s, ≈ 0.5 GB). **Steps 5–7** (the
  part that is tuned by eye) are re-run for the shown frame. Before it,
  the back modules are warmed up on the 40 previous frames
  (`ToneMap.warmup` updates only range and histogram). During playback the
  state continues, no warm-up.
- Browser page served by the Python standard library (Tkinter is not
  installed in this Python; no new packages needed). Listens on 127.0.0.1 only.
- `main.py` now describes the pipeline once: `build_front` (0–3),
  `build_back` (5–7), `TUNABLE` (slider list). `tuner.py` uses the same
  functions, so the tuner and the final run cannot differ.
- "Save" wrote one `params.json` (replaced by named presets in entry 16).

**Result:**

| Check | Result |
|---|---|
| Render after a jump (with warm-up of 40 frames) | ≈ 165 ms |
| Render during playback | ≈ 27 ms |
| Difference to a full sequential run (frames 100, 164, 250) | mean 0.09–0.44 grey levels, max 1–4 (warm-up 15: up to 1.2 mean, max 13) |
| Rebuild of steps 0–3 after a denoise change | ≈ 12 s |
| API (state, render, params, reset, rebuild) and page script | tested |

18 sliders: tone mapping (black level, white level, knee, foot, plateau,
linear mix, mid detail, range percentiles), base/detail split (radius,
edge), fine detail (gain, noise gate, scale, halo limit), and denoise
(temporal memory, temporal tolerance, spatial strength – need a rebuild).

**Files:** `tuner.py`, `thermal/tuning.py`, `main.py`,
`thermal/stages/tonemap.py` and `detail.py` (`warmup`), `README.md`

---

## 16. Tuner: presets and keyboard control

**Feedback:** one `params.json` that is always overwritten is not enough –
different versions of the settings must be kept. Sliders are hard to move
precisely with the mouse; frames should be controlled by letters and
parameters by arrow keys.

**What was done:**

- **Presets**: every save is a separate file `presets/<name>.json` with
  the values, date and an optional note. Load from a list (newest first).
  The header shows the active preset and "(modified)". Overwriting asks
  first; unsafe names (e.g. `../x`) are refused.
- `main.py --params <name>` (or a path to a .json file) applies a preset;
  without it the defaults in the code are used. An unknown name prints the
  list of available presets. `tuner.py --params <name>` starts from a preset.
  The automatic `params.json` was removed.
- **Keyboard** (physical keys, so it works with the Ukrainian layout too):

  | Key | Action |
  |---|---|
  | `A` / `D` (+`Shift`: 10) | frame back / forward |
  | `Space` | play / pause |
  | `↑` / `↓` | select parameter |
  | `←` / `→` (+`Shift`: 10 steps) | change the selected parameter |
  | `R` | reset the selected parameter |

  After a mouse click, buttons and checkboxes give the focus back, so the
  keys keep working.

**Tested:** save, refusal to overwrite without confirmation, refused unsafe
name, "modified" flag, load, `main.py --params` with an existing and an
unknown name, page script syntax (Node).

**Files:** `tuner.py`, `thermal/tuning.py` (preset functions), `main.py`, `README.md`

---

## 17. Recommended preset, algorithm switches, CLAHE

**Feedback:** for frame 100 the user chose the preset `knee-min-linear-max`
(`knee` 0.88 → 0.70, `linear_mix` 0.3 → 1.0, i.e. no histogram
equalisation). Questions: should there be a recommended version to start
from; can algorithms be switched on / off in the tuner; where are AGC,
CLAHE, gamma correction, ISP, percentile normalisation, static vs. dynamic
noise in our pipeline.

**Measurement** (all 275 frames, same denoised input):

| Variant | Fine noise (less = less "painted") | Ground contrast when the sky enters | Car structure (std) | Flicker | Largest brightness jump |
|---|---|---|---|---|---|
| Old defaults (knee 0.88, mix 0.3, detail 1.5 / gate 2) | 1.18 | −20 % | 8.7 | 0.42 | 2.3 |
| User preset (knee 0.70, mix 1.0) | **0.72** | −42 % (as naive) | **22.0** | 0.54 | 5.0 |
| **Recommended (knee 0.75, mix 0.6, fine detail 1.0 / gate 3)** | 0.75 | **−28 %** | 17.0 | **0.42** | 3.4 |

Finding: the "painted" look in frame 100 came mostly from histogram
equalisation – it raises the slope (and with it the detail) where the
histogram is dense. A linear curve removes it but loses the protection of
the scene contrast when the cold sky enters. The recommended values keep
most of both.

**What was done:**

- `presets/recommended.json` with a note; the defaults in the code are the
  same values (Reset all = recommended).
- Loading a preset first resets to the defaults (no mixing of presets).
- **Algorithm switches** in the tuner (11): bad pixels, residual NUC,
  destripe, temporal / spatial denoise (these need a rebuild), base/detail
  split, histogram equalisation, soft knee, temporal AGC smoothing, CLAHE,
  fine detail. Generic mechanism: `enabled = False` → the pipeline skips
  the module (also for partner modules).
- `ToneMap` switches `use_plateau`, `use_knee`, `temporal`.
- New optional module `LocalContrast` (CLAHE), off by default.
- `DetailEnhance`: the soft limiter was replaced by a smooth limit based on
  the distance to 0 / 1. Reason: with the old limiter, hard clipping
  (`use_knee` off) was hidden (0 % white). Now: recommended 0.000 % white;
  knee off 0.09 % white, 0.56 % black – the clipping is visible as it should be.

**Common terms and where they are in the pipeline:**

| Term | In our pipeline |
|---|---|
| ISP (image signal processing) | the whole chain: steps 0–10 |
| AGC (automatic gain control) | `ToneMap`: scene range from percentiles, smoothed over time (switch "Temporal smoothing") |
| Percentile normalisation | `ToneMap` `p_low` / `p_high` (p0.2 / p99.8, smoothed); naive per-frame version = `NaiveAGC` (p2 / p98) |
| Histogram equalisation | `ToneMap` plateau HE (global) |
| CLAHE | `LocalContrast`, optional, off by default |
| Gamma correction | `ToneMap` `gamma` (applied to the tone curve, keeps the temperature order). No extra display gamma needed: the output is already display-referred 8-bit |
| Static noise (fixed pattern) | `ResidualNUC` (per pixel, 4.6 → 0.7 DN) + `Destripe` (rows / columns) |
| Dynamic noise (temporal) | `TemporalDenoise` + `SpatialDenoise` (≈ 3.4–4.1 → ≈ 1.5 DN) |
| Detail enhancement (DDE) | `BaseDetailSplit` + `ToneMap(detail_gain)` + `DetailEnhance` |

**Files:** `presets/recommended.json`, `main.py`, `tuner.py`,
`thermal/pipeline.py`, `thermal/stage.py`, `thermal/tuning.py`,
`thermal/stages/tonemap.py`, `clahe.py`, `detail.py`, `README.md`

---

## 18. Steps 8–10: upscale, dither, video files

**Decision:** H.264 with the **full range 0..255** (the video is shown on
our own laptop, so the player can be chosen), plus a lossless copy.

**What was done:**

- `Upscale` (8): Lanczos 2×, 640 × 512 → 1280 × 1024, on float data
  before quantisation.
- `Quantise` (9): 8 bit with a fixed blue-noise dither (±0.5 grey level).
- `VideoWriter` (10): one module per file, frames piped to ffmpeg:
  `.mp4` H.264 High, full range, CRF 14; `.mkv` FFV1 lossless grey.
  The file is closed in `flush`, also after an early stop.
- `write_range_test` / `main.py --range-test`: test clip to check that a
  player shows the full range.
- `main.py`: `build_output` (steps 8–10, not used by the tuner), options
  `--out`, `--no-video`, `--upscale`, `--crf`, `--range-test`; file names
  `<recording>__<preset>.mp4 / .mkv`. `Compare` fits its window to 1600 px.

**Result:**

| Check | Result |
|---|---|
| Dither on a very flat ramp: error of the local mean | 0.24 (rounding) → 0.015 grey levels (blue noise) |
| Range test after H.264 encode + decode | patches 0, 8, 16, 24, 231, 239, 247, 255 decoded exactly; ramp 0..255 |
| `.mkv` vs. pipeline frames | identical |
| `.mp4` vs. pipeline frames | mean 0.6–0.7 grey levels, max 6–12 |
| File sizes (275 frames, 11 s) | `.mp4` 11.6 MB (≈ 8.4 Mbit/s), `.mkv` 86.7 MB |
| Final video (recommended preset) | white 0.001 %, black 0.004 %, flicker 0.42, largest brightness jump 3.4 levels |
| Full run, steps 0–10 + both files | ≈ 23 s |

**Files:** `thermal/stages/upscale.py`, `quantise.py`, `video.py`,
`thermal/stages/compare.py`, `main.py`, `README.md`

---

## 19. Adaptation when a hot object enters; peak hold; Load bug

**Question:** do we adapt the levels when a bright object enters, so that
there is no overexposure?

**Answer – yes, on three levels (`ToneMap`):** adaptive scene range
(percentiles, fast growth / slow release), soft knee for everything above
the range, plateau HE (a hot object takes few grey levels).

**Measurement – hot spots enter at frame 168** (frame max 7779 → 8043 DN):

| | Frame 165 | 168 | 174 | 190 |
|---|---|---|---|---|
| Frame max (DN) | 7779 | 8043 | 8165 | 8216 |
| Knee scale max, plain smoothing (DN) | 7733 | 7831 (lags) | 8008 | 8161 |
| Brightest output pixel | 254 | 254 | 254 | 252 |
| Median brightness of the scene | 107 | 107 | 101 | 76 |

No overexposure (pixels at 255 in 26 frames, ≤ 0.01 % each); largest
change of the median 3 levels per frame. Weak points: (1) the knee scale
lagged ≈ 6 frames, so a new hot object had compressed inner detail; (2) the
scene darkens slowly when hot objects take a larger part of the frame
(partly a real scene change after the fast turn at frame 164).

**What was done:**

- **Peak hold** in `ToneMap` for the max / min that scale the knee and the
  foot: follow a new peak at once (`peak_attack` 1.0), hold it
  (`peak_hold` 12 frames), then release slowly.
  Result (frames 168–174): knee scale on the true max from the first frame
  (8031 instead of 7831); inner structure of the hot spots grey-level std
  0.8 → 2.1; pixels at 255 0.0098 % → 0.0015 %; flicker unchanged.
- Tuner sliders: "Adapt: range grows / shrinks", "Adapt: histogram",
  "Peak hold: attack / frames". `recommended.json` updated (37 values);
  `output/normal_nuc_2__recommended.mp4 / .mkv` re-written.
- **Bug fixed – Load did not work** (also Reset all, Rebuild, and the
  sliders after Save): after a server answer the page replaced its list of
  parameters by a plain copy without the links to the sliders, so the next
  line failed. Now the list is kept.
- "Needs rebuild" is now computed from the values (switching a step 0–3
  module off and on again no longer asks for a rebuild).
- **UI test** with a browser emulator (jsdom, outside the project):
  initial render, A / D, ↑ / ↓ / ← / →, R, switches, Load, Save, Reset,
  Load own preset, keys after Load – 15 / 15 passed.

**Files:** `thermal/stages/tonemap.py`, `thermal/tuning.py`, `tuner.py`,
`main.py`, `presets/recommended.json`, `README.md`

---

## 20. Multi-scale detail and local contrast for hot objects

**Question:** are these presets, sliders or separate algorithms?
**Answer:** separate algorithms (modules), each with an on/off switch and
its own sliders in the tuner; presets only store combinations.

### `MultiScaleDetail` (step 7b)

Detail on three scales (DoG 1 / 2.5 / 6 px), per band a noise gate and edge
protection (no boost of strong edges). Measured every 5th frame:

| Variant | Coarse | Mid | Fine | Noise band | Overshoot at strong edges |
|---|---|---|---|---|---|
| no detail step | 1411 | 77 | 4.0 | 0.039 | 0 |
| `DetailEnhance` (recommended) | 1456 | 93 | 6.5 | 0.090 | 0.3 levels |
| **`MultiScaleDetail` 0.6 / 1.0 / 0.6** | **2182** | **180** | **9.9** | **0.064** | 1.4 levels |
| `MultiScaleDetail` 1.0 / 1.5 / 1.0 | 2677 | 247 | 13.6 | 0.082 | 3.0 levels |

≈ 1.5–1.9× more detail on the coarse and middle scales than `DetailEnhance`,
less noise in the finest band; price: a little more overshoot at strong
edges. Visually: clearly crisper crowns and orchard, fields slightly more
mottled. Off by default; preset `recommended-multiscale` and video
`output/normal_nuc_2__recommended-multiscale.mp4` for comparison.

### `HotSpotContrast` (step 6b)

- First version (local mean of the hot pixels): inner structure std
  12.4 → 23.6, but it drew a **dark outline** around hot objects (seen in
  frame 164). Cause: pixels in the fade-in zone at the border are darker
  than the hot core and were pushed down.
- Second version (structure = image − guided filter): no outline (change
  in the border zone +0.35 levels), the scene is not changed (0.00 levels
  without `DetailEnhance`; with it, mean 0.8 levels next to hot objects).
  But almost no visible effect on the bright strips.
- **Why – measured in the data:**

  | Strip | Data | Conclusion |
  |---|---|---|
  | Frame 100, below the trees | not in the knee: median 147, p95 182, knee at 191 | bright because it is the warmest band of the scene; the main tone curve, not the knee |
  | Frame 164 | interior range only 72 DN, shown with 37 grey levels (≈ 0.5 levels / DN, as the scene) | an almost even hot surface – no hidden structure |
  | Frame 175, hot spots | 364 DN in ≈ 42 grey levels | here the knee compresses; the module can help, but the spots are small |

- Decision: module kept, **off by default**. For the bright strips the
  level settings work: `White level`, `Knee start`, `gamma`, `Linear mix`.

**Also:** presets `recommended` and `recommended-multiscale` (48 values
each); tuner UI test 15 / 15 passed with the new modules.

**Files:** `thermal/stages/multiscale.py`, `thermal/stages/hotspot.py`,
`thermal/stages/tonemap.py` (`meta["tonemap.knee"]`), `detail.py`,
`main.py`, `presets/`, `README.md`

---

## 21. Ideas from Michael's `thermal_isp`, DDE detail mode

**Goal:** take the useful parts of the partner pipeline into this framework
as separate switches, to compare them one by one before optimising speed.
Constraint: streaming – only the current frame and state from past frames.

**What was done:**
- `ShutterGuard` (step 0b) + `ShutterHold`: shutter/FFC frames frozen;
  `Pipeline` and the tuner skip modules without `run_when_frozen` on such
  frames (`thermal.pipeline.stage_runs`). Tested on Michael's synthetic
  sequence (flat frames 90–92): frozen, output = frame 89, state not
  updated. No frozen frames in `normal_nuc_2.bin`.
- `Destripe(temporal_alpha)`: offsets smoothed over time.
- `ToneMap`: `anchor_black` (slew-limited black point with a scene-change
  rule by share of the frame), `white_hold`, `min_span_dn`, `shadow_lift`,
  and the DDE detail mode (`dde`, own gain, noise gate, adaptive local
  gain, warm boost). `layers["linear"]` and `meta["tonemap.lin_slope"]`
  are written for later modules.
- `CompactHotBoost` (step 6c): white top-hat, soft or hard.
- Preset `michael-features`; tuner switches marked "(M)".

**Results:** see README, section "Ideas taken from Michael's thermal_isp".
The jump when the sky enters (frames ~100–120) is gone with
`anchor_black` + `white_hold`; the cold sky goes into the foot.

**Speed (container, 275 frames, `michael-features`, no video):** total
≈ 34 s; `TemporalDenoise` 12.9 s (38 %), `ToneMap` 6.9 s (20 %),
`MultiScaleDetail` 3.2 s, `BadPixels` 2.5 s, `Destripe` 2.2 s,
`SpatialDenoise` 2.0 s, `BaseDetailSplit` 1.4 s, `ResidualNUC` 1.3 s,
`CompactHotBoost` 0.8 s.

## Open items and next steps

- [ ] Step 4 (postponed, see entry 12) – deblurring or sharpness-weighted
      temporal fusion, only if the result after steps 5–8 looks too soft.
- [ ] Check `output/range_test.mp4` in the player chosen for the demo.
- [ ] Real frame rate of the sensor (25 fps assumed).
- [ ] Final validation against the table in docs section 4 on the final video.
- [ ] Subjective tuning with `tuner.py`: black level, white level of hot
      strips, sharpness vs "painted" look in blurred areas (frame 100).
      Save as a preset; run `main.py --params <name>`.
- [ ] `ToneMap` is slow (≈ 4.7 s / 275 frames) – vectorise the curve lookup.
- [ ] TODO from entry 5 still open: add the small-hot-object check to the
      validation table in the docs.
- [ ] Docs, section 1: the fixed pattern is described as "stripes ≈ 2 DN";
      measurements show a per-pixel pattern of 4.6 DN. Update the table in
      both versions.
- [ ] Noise estimate: the Laplacian method gives ≈ 4.1 DN, the spectrum
      floor ≈ 3.4 DN (entry 12). Consider a more exact estimator.
- [ ] `TemporalDenoise` is the slowest step (≈ 7 s / 275 frames, optical
      flow) – `flow_preset="fast"` if speed matters.
