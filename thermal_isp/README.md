# Thermal ISP (v3.2): 14-bit post-NUC 640x512 -> stable 8-bit white-hot

Run (Python 3, numpy, opencv-python, ffmpeg in PATH):

    pip install -r requirements.txt
    python analyze.py data.bin            # what is still wrong after NUC (noise, stripes, shutter frames, bad pixels)
    python run.py data.bin -o out         # out/*_whitehot.mp4, *_compare.mp4 (naive >>6 | ISP), before/after PNG, timings
    python compare_versions.py data.bin out   # v3 vs v3.1 side by side

Flags: --row-fpn, --detail-gain G, --black-hot, --max-frames N. All parameters: `Params` in isp.py.

## Pipeline (v3.1)
| # | Stage | Fixes |
|---|-------|-------|
| 0 | Shutter/FFC guard: flat frame -> repeat last output, skip state update | gray/flashing frame during FFC |
| 1 | Bad pixels: static map + median of good 8-neighbours (real small targets are kept: a PSF raises the neighbours) | white/black dots |
| 2 | Residual column FPN: clipped horizontal high-pass averaged over rows, IIR over time | vertical stripes left by NUC |
| 3 | Motion-compensated recursive TNR: similarity transform (shift+roll+zoom) from LK-tracked corners + RANSAC (fallbacks: phase-correlation shift, then hold last), plus dense 1/4-res flow for parallax, one combined cubic warp of the state -> per-pixel k = 1/4 static ... 1 moving, vs spatially measured noise sigma | temporal noise, no ghosting on a panning camera |
| 4 | Absolute-anchored tone curve from frame t-1 statistics: black point = slow 1% percentile (slew-limited 1.5 counts/frame), white point = peak-hold of the per-frame 99.95% percentile (min stretch 1000 counts), gamma 1.15, shadow lift 0.12. No histogram equalisation | brightness stability / how bright large warm surfaces look |
| 5 | Fast guided-filter base/detail split; LUT on base; detail x2.6, warm detail x1.5 more with coring + soft clip; **5b** white top-hat (opening ~14 px) adds a boost to compact hot features above 150 counts | local contrast without halos or plastic texture |
| 6 | white-hot 8-bit | |

FPGA mapping: one frame buffer (TNR state) 640x512x16 bit = 640 KB; a few line buffers (bad pixels 3, column FPN 15-px shift register + 640 accumulators, guided filter ~5 lines at half res); histogram 2048x24 bit; LUT 16K x 8 bit. Statistics stages have a 1-frame latency, the pixel path ~20 lines.

## Test data (normal_nuc_2.bin, 275 frames, 11 s)
Moving camera (~1.6 px/frame pan, some roll), temporal noise sigma ~6 counts, scene range only ~870 counts (naive >>6 uses 14 of 256 levels), no shutter frames, 2 bad pixels, residual stripes ~1.4 counts.

## Known limitation
Bad-pixel statistics are computed over the whole clip. On very short or nearly static clips the 'stuck pixel' test (std < 0.15 x median std) can flag thousands of healthy pixels (measured: 4146 on the first 15 frames vs 2 on all 275). Use clips of ~100+ frames with camera motion or tighten that test.

## Version history
- **v1**: Basic pipeline: shutter guard, bad pixels, column FPN, recursive TNR with shift compensation (phase correlation), plateau histogram equalisation, guided-filter base/detail split.
  Result: Weakness: equalisation stretches the most populated levels, so road/ground/roofs drift to white and brightness wanders with frame content.
- **v2**: Absolute tone scale instead of equalisation: slow, slew-limited black point, peak-hold white point, gamma 1.15; detail x2.6, warm detail a further x1.5.
  Result: Road/ground mid-grey, animals brightest; max frame-to-frame mean-brightness jump 8.3 -> 2.0 levels. Weakness: temporal filter barely worked (k=0.72) because registration was poor.
- **v3**: Motion estimation: similarity transform (shift+rotation+scale) from LK corners + RANSAC, then dense local flow at 1/4 resolution for parallax, one cubic warp. Shadow lift (shadow_lift=0.12).
  Result: k 0.72 -> 0.40, residual 2.07 -> 1.04 sigma; vegetation shadows +21 levels, ground +8..13, animals unchanged. Cost: 12 -> ~23 ms/frame.
- **v3.1**: Boost of compact warm anomalies (white top-hat, ~14 px window, threshold 150 counts).
  Result: Lone spot 182 -> 240; road, ground, canopy, roofs unchanged (+0); at most 0.7% of the frame changes. Cost ~+1 ms.
- **v3.2**: Bad-pixel detector fixed for short clips: frozen-pixel test tied to the measured sensor noise (not to the sequence-wide std), sd-based tests need >= 8 frames, hot/cold threshold floored by sigma/sqrt(N).
  Result: false positives on real clips of 5/15/60 frames 7426/4146/1032 -> 19/15/2; full 275-frame file unchanged (identical mask, 2 pixels). Run `python test_bad_pixels.py`.

## Streaming check
`python stream_demo.py normal_nuc_2.bin --fps 25` replays the file as a camera (one frame at a time, bad-pixel map from the first 16 frames) and reports per-frame processing time, queueing latency and frames over budget.
Measured on 2 shared cores: 25 fps -> mean 25.7 ms, p99 35.0, max 39.3 ms, 0 of 275 frames over the 40 ms budget; 30 fps -> 6 of 275 over budget (max backlog 1 frame).

## Dataset for a learned enhancement model
`python make_dataset.py normal_nuc_2.bin -o dataset --full` builds noisy/clean pairs from the real file (no synthetic scene):
- input: the real raw frame (real noise, column FPN, bad pixels);
- target: offline, non-causal average of +-8 neighbouring real frames aligned by chained adjacent-frame motion (similarity + local flow) with per-pixel gating against moving objects;
- outputs: 128x128 patches in train/val/test .npz (time split with gaps, so neighbouring frames never leak across splits), a full-frame clean .bin in the layout of the input, a noise bank and a quality report.

Why chained motion: the camera pans 6-14 px per frame; direct LK between frames more than 1-2 apart locks on wrong points.
Target quality (12 sampled frames): raw sigma 6.08 counts, about 10 frames effectively averaged, noise of independent half-sets 2.42 vs ideal 2.58 (no misalignment excess), sharpness ratio vs raw 1.001. The target still contains about 2.4 counts of noise; it is a 2.5x cleaner reference, not a perfect one.

The sensor noise is not white (lag-1 correlation about 0.2, power concentrated at low frequencies), so a Gaussian noise model was tried and rejected. `NoiseBank` transplants real residuals (raw - clean) from training frames onto clean patches; on test frames it matches the real residual: sigma 5.93 vs 5.93, lag-1 correlation 0.20/0.23 vs 0.17/0.20.

`python train_demo.py dataset --minutes 8` (needs PyTorch) trains a 117k-parameter U-Net on CPU. Measured on 2 shared vCPUs: 2.45 s per batch of 16 patches (6.5 patches/s), about 3 epochs in 8 minutes; inference 127 ms per 640x512 frame (too slow for 25 fps on that CPU).
Real test patches from the last 15% of the clip (error vs clean target): raw 5.93 counts, Gaussian blur 3.73 (sharpness 0.92), network 3.37 (sharpness 0.98). Still improving when stopped; test frames come from the same scene, so generalisation to new scenes is untested.

