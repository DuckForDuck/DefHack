# thermal_isp (C++)

C++ version of the Python pipeline in `project/` with the preset
**`michael-features`**: raw 14-bit thermal frames (640 × 512, `uint16`
little-endian) → 8-bit white-hot H.264 video.

- **Streaming:** frames are read and processed one by one (also from `stdin`).
  Every stage uses only the current frame and the state it kept from earlier
  frames. The residual NUC is learned during the run (no calibration in advance,
  = Python `--nuc-fit-step 0`).
- **Not included** (compared with the Python version): the tuner, the lossless
  `.mkv` copy, debug PNGs, presets as files. The values of the preset are fixed
  in `src/params.hpp`, each one named after its preset key.

## Build (macOS)

```bash
brew install opencv cmake ffmpeg
cd project/cpp
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

Linux: `sudo apt install libopencv-dev cmake ffmpeg`, then the same two
`cmake` lines.

## Run

```bash
# video: output/normal_nuc_2__michael-features__cpp.mp4
./build/thermal_isp ../../normal_nuc_2.bin

# side-by-side video, ours (left) vs the original script.py (right):
# output/normal_nuc_2__compare__cpp.mp4
./build/thermal_isp ../../normal_nuc_2.bin --compare

# speed only (no video), time per stage
./build/thermal_isp ../../normal_nuc_2.bin --no-video --profile

# streaming from another program
cat ../../normal_nuc_2.bin | ./build/thermal_isp - --out output
```

| Option | Meaning |
|---|---|
| `--out DIR` | output folder (default `output`) |
| `--compare` | write the side-by-side video (ours vs `script.py`) instead of ours alone |
| `--no-video` | process only (for speed tests) |
| `--profile` | print the time of each stage |
| `--frames N` | stop after N frames |
| `--width`, `--height`, `--fps` | frame size (640 × 512) and video frame rate (25) |
| `--flow medium\|fast\|ultrafast` | DIS optical-flow preset. `medium` = the preset (default). `fast` is not in the preset: about 30 ms less per frame on a 2-core test machine, output differs by 0.7 grey levels on average (check moving scenes for ghosting) |
| `--raw-out FILE` | also write the 8-bit frames as raw bytes (to compare with the Python version) |

The printed frame rate counts only the processing (steps 0–9), not reading the
file or encoding the video. For the speed of the algorithm use `--no-video`:
ffmpeg runs on the same cores and slows the measurement down.

## Stages (in order) and their Python originals

| Step | C++ (`src/stages.cpp`) | Python (`project/thermal/stages/`) |
|---|---|---|
| 0 | `Decode` | `decode.py` |
| 0b | `ShutterGuard` | `shutter.py` |
| 1 | `BadPixels` | `badpixels.py` |
| 2a | `ResidualNUC` (online) | `nuc.py` |
| 2b | `Destripe` (temporal smoothing 0.1) | `destripe.py` |
| 3a | `TemporalDenoise` (DIS optical flow) | `temporal.py` |
| 3b | `SpatialDenoise` (guided) | `spatial.py` |
| 5 | `BaseDetailSplit` | `base_detail.py` |
| 6 | `ToneMap` (anchored black point, white peak hold, linear + gamma 1.3, knee/foot, DDE detail) | `tonemap.py` |
| 6c | `CompactHotBoost` (white top-hat, soft) | `tophat.py` |
| 7b | `MultiScaleDetail` | `multiscale.py` |
| – | `ShutterHold` | `shutter.py` |
| 9 | `Quantise` (blue-noise dither; same 64 × 64 tile, `src/blue_noise.hpp`) | `quantise.py` |
| 10 | `VideoWriter` (ffmpeg, H.264 CRF 10, full range) | `video.py` |
| – | `naive_agc` (for `--compare`) | `script.py` |


