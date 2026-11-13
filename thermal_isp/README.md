# Thermal ISP (v1): 14-bit post-NUC 640x512 -> stable 8-bit white-hot

Run (Python 3, numpy, opencv-python, ffmpeg in PATH):

    pip install -r requirements.txt
    python analyze.py data.bin            # what is still wrong after NUC (noise, stripes, shutter frames, bad pixels)
    python run.py data.bin -o out         # out/*_whitehot.mp4, *_compare.mp4 (naive >>6 | ISP), before/after PNG, timings

Flags: --row-fpn, --detail-gain G, --black-hot, --max-frames N. All parameters: `Params` in isp.py.

## Pipeline (v1)
| # | Stage | Fixes |
|---|-------|-------|
| 0 | Shutter/FFC guard: flat frame -> repeat last output, skip state update | gray/flashing frame during FFC |
| 1 | Bad pixels: static map + median of good 8-neighbours (real small targets are kept: a PSF raises the neighbours) | white/black dots |
| 2 | Residual column FPN: clipped horizontal high-pass averaged over rows, IIR over time | vertical stripes left by NUC |
| 3 | Motion-adaptive recursive TNR with global-TRANSLATION compensation: phase-correlation shift (half res, smoothed, held when unreliable) -> cubic warp of the state -> per-pixel k = 1/4 static ... 1 moving, from 3x3-pooled |frame-state| vs spatially measured noise sigma (Laplacian MAD) | temporal noise, no ghosting on a panning camera |
| 4 | Tone curve from frame t-1 statistics: temporally smoothed histogram (2048 bins), percentile range 0.5%..99.95%, plateau-limited histogram equalisation blended 50/50 with linear | brightness stability / how bright large warm surfaces look |
| 5 | Fast guided-filter base/detail split; LUT on base; detail x1.5 with coring + soft clip | local contrast without halos or plastic texture |
| 6 | white-hot 8-bit | |

FPGA mapping: one frame buffer (TNR state) 640x512x16 bit = 640 KB; a few line buffers (bad pixels 3, column FPN 15-px shift register + 640 accumulators, guided filter ~5 lines at half res); histogram 2048x24 bit; LUT 16K x 8 bit. Statistics stages have a 1-frame latency, the pixel path ~20 lines.

## Test data (normal_nuc_2.bin, 275 frames, 11 s)
Moving camera (~1.6 px/frame pan, some roll), temporal noise sigma ~6 counts, scene range only ~870 counts (naive >>6 uses 14 of 256 levels), no shutter frames, 2 bad pixels, residual stripes ~1.4 counts.

## Known limitation
Bad-pixel statistics are computed over the whole clip. On very short or nearly static clips the 'stuck pixel' test (std < 0.15 x median std) can flag thousands of healthy pixels (measured: 4146 on the first 15 frames vs 2 on all 275). Use clips of ~100+ frames with camera motion or tighten that test.

## Version history
- **v1**: Basic pipeline: shutter guard, bad pixels, column FPN, recursive TNR with shift compensation (phase correlation), plateau histogram equalisation, guided-filter base/detail split.
  Result: Weakness: equalisation stretches the most populated levels, so road/ground/roofs drift to white and brightness wanders with frame content.
