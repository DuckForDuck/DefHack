# thermal_isp (C, vectorised)

C version of the thermal pipeline, preset **`michael-features`**: raw 14-bit
thermal frames (640 × 512, `uint16` little-endian) → 8-bit white-hot H.264
video. Same algorithm and parameter values as the C++ version (`cpp/`) and
the Python version (`project/`); it only runs faster.

- **No libraries.** Only a C11 compiler and pthreads; `ffmpeg` is needed at
  run time to write the `.mp4`. OpenCV is gone: every filter the C++ version
  took from OpenCV (Gaussian, box, Sobel, morphology, resize, remap and the
  DIS optical flow) is written here in C, with the same kernel sizes, borders
  and rounding.
- **Vectorised.** Hot loops use the GCC/Clang vector extensions (8 floats per
  vector): the same source gives NEON code on Apple Silicon and AVX code on
  x86. The other per-pixel loops are written so the compiler vectorises them.
- **Parallel.** A small thread pool runs the loops over rows. Results are
  identical for any number of threads.
- **Streaming.** Frames are read and processed one by one (also from `stdin`).
  Every stage uses only the current frame and the state it kept from earlier
  frames. The residual NUC is learned during the run.
- **Not included** (as in the C++ version): the tuner, the lossless `.mkv`,
  debug PNGs, presets as files. The preset values are in `src/params.h`, each
  one named after its preset key.

## Build

```bash
# macOS: xcode-select --install ; brew install ffmpeg
# Linux: sudo apt install build-essential ffmpeg
cd c
make                 # -> ./thermal_isp, optimised for this machine
```

`make PORTABLE=1` builds without `-march/-mcpu=native` (for another machine).
CMake works too: `cmake -S . -B build && cmake --build build -j`.

## Run

```bash
# video: output/normal_nuc_2__michael-features__c.mp4
./thermal_isp ../normal_nuc_2.bin

# side-by-side video, ours (left) vs the original script.py (right):
# output/normal_nuc_2__compare__c.mp4
./thermal_isp ../normal_nuc_2.bin --compare

# speed only (no video), time per stage
./thermal_isp ../normal_nuc_2.bin --no-video --profile

# streaming from another program
cat ../normal_nuc_2.bin | ./thermal_isp - --out output
```

| Option | Meaning |
|---|---|
| `--out DIR` | output folder (default `output`) |
| `--compare` | write the side-by-side video (ours vs `script.py`) instead of ours alone |
| `--no-video` | process only (for speed tests) |
| `--profile` | print the time of each stage |
| `--frames N` | stop after N frames |
| `--width`, `--height`, `--fps` | frame size (640 × 512) and video frame rate (25) |
| `--threads N` | worker threads. Default: all cores with `--no-video`, half of them when a video is written (the x264 encoder needs the rest) |
| `--flow medium\|fast\|ultrafast` | DIS optical-flow preset. `medium` = the preset (default). `fast` is not in the preset: quicker, less exact motion (check moving scenes for ghosting) |
| `--raw-out FILE` | also write the 8-bit frames as raw bytes (to compare with other versions) |

The printed frame rate counts only the processing (steps 0–9), not reading the
file or encoding the video. For the speed of the algorithm use `--no-video`.

Video settings are unchanged: H.264, CRF 10, preset `slow`, tune `grain`,
full range 0–255.

## Speed

Same 2-core cloud machine (x86, AVX-512), all 275 frames of
`normal_nuc_2.bin`, `--no-video`, time per frame (mean):

| | C++ (OpenCV) | C | |
|---|---|---|---|
| preset (`--flow medium`) | 62.6 ms (16 fps) | **32.4 ms (31 fps)** | 1.9× |
| `--flow fast` | 44.4 ms (23 fps) | **24.0 ms (42 fps)** | 1.85× |
| C on 1 thread, `medium` | | 57.9 ms | |

The sensor gives 25 fps (40 ms per frame); the C version keeps up with the
preset on two cores. With the video written (`.mp4`, x264 `slow`), the whole
run takes 20.3 s (C) vs 22.9 s (C++): there the encoder is the limit (x264
alone needs 10 s for this file on two cores). On a computer with more cores
both parts get faster: the stages run in parallel, DIS uses up to 8 threads.

Where the time went: half of it is the DIS optical flow (TemporalDenoise).
What made the C version faster than OpenCV:

- DIS patch search: one SIMD vector per 8-pixel patch row;
- DIS densification: patch by patch, 8 pixels at a time (same sums as OpenCV);
- variational refinement: red-black SOR on packed rows, no lanes wasted;
- medians of whole frames (BadPixels, SpatialDenoise): exact radix selection
  instead of sorting (0.5 ms instead of 3.5 ms each);
- a thread pool that starts a parallel loop in ~1 µs (a frame has a few
  hundred of them).

## Checked against the C++ version (14.11.2026)

All 275 frames, `--raw-out` of both versions:

- mean difference **0.12 grey levels**; 0.04 % of pixels differ by more than 2,
  0.0001 % by more than 8 (frames 0–2: at most 2 levels apart). This is the
  same size as the difference between the C++ and the Python version (0.10):
  tiny float rounding differences (order of sums, FMA) grow a little through
  the stages that keep state from frame to frame (NUC offset, temporal
  denoise).
- same output with 1, 2 or 3 threads (bit for bit);
- AddressSanitizer / UBSan: no errors (also with odd frame sizes such as
  457 × 333 and 131 × 101).

## Files

| File | What | C++ / OpenCV original |
|---|---|---|
| `src/main.c` | command line, reading frames, compare video, timing | `main.cpp` |
| `src/stages.c` | the 13 stages (steps 0–9), `naive_agc` for `--compare` | `stages.cpp` |
| `src/params.h` | preset values | `params.hpp` |
| `src/dis.c` | DIS optical flow + variational refinement | OpenCV `dis_flow.cpp`, `variational_refinement.cpp` |
| `src/filters.c` | Gaussian, box, Sobel, morphology, resize, remap, percentiles, noise estimate | OpenCV `imgproc`, `utils.cpp` |
| `src/parallel.c` | thread pool | `cv::parallel_for_` |
| `src/video.c` | ffmpeg pipe | `video.cpp` |
| `src/common.h` | SIMD vector type, small helpers | – |
| `src/blue_noise.h` | 64 × 64 dither tile (same as Python / C++) | `blue_noise.hpp` |
| `src/font_data.h` | DejaVu Sans glyphs for the captions of `--compare` | `cv::putText` |

Stages in order: Decode → ShutterGuard → BadPixels → ResidualNUC → Destripe →
TemporalDenoise (DIS) → SpatialDenoise → BaseDetailSplit → ToneMap (DDE) →
CompactHotBoost → MultiScaleDetail → ShutterHold → Quantise (blue noise).
