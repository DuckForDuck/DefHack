"""
view.py - live demo: raw (left) vs ISP (right), switch stages on/off while playing.

  python view.py data.bin

Keys: SPACE play/pause   D / A next / previous frame
      1 bad pixels   2 stripes   3 temporal + offset   4 detail
      5 next AGC mode (plateau / percentile / mean)
      S save PNG     Q quit
Stepping backwards restarts the pipeline state from that frame (like a camera would).
"""
import sys

import numpy as np
import cv2

from isp import ThermalISP, naive_scale, open_frames, WIDTH, AGC_MODES

STAGE_KEYS = {"1": "badpix", "2": "stripes", "3": "temporal", "4": "dde"}


def draw_view(raw, image, index, total, cfg):
    view = np.hstack([naive_scale(raw), image])
    active = " ".join(key for key, stage in STAGE_KEYS.items() if getattr(cfg, stage)) or "none"
    cv2.putText(view, f"frame {index}/{total - 1}  RAW min/max", (10, 25), 0, 0.6, 255, 2)
    cv2.putText(view, f"ISP stages on: {active}  AGC: {cfg.agc_mode}", (WIDTH + 10, 25), 0, 0.6, 255, 2)
    return view


def main():
    frames = open_frames(sys.argv[1])
    total = len(frames)
    isp = ThermalISP()
    index, last_index, playing = 0, -1, True

    while True:
        if index != last_index + 1:                 # jumped: start fresh state
            isp = ThermalISP(isp.cfg)
        raw = np.array(frames[index])
        image = isp.process(raw)
        last_index = index

        cv2.imshow("thermal ISP", draw_view(raw, image, index, total, isp.cfg))
        key = cv2.waitKey(40 if playing else 0) & 0xFF
        char = chr(key).lower() if key != 255 else ""

        if char == "q" or key == 27:
            break
        if char == " ":
            playing = not playing
        elif char in STAGE_KEYS:
            stage = STAGE_KEYS[char]
            setattr(isp.cfg, stage, not getattr(isp.cfg, stage))
        elif char == "5":
            next_mode = (AGC_MODES.index(isp.cfg.agc_mode) + 1) % len(AGC_MODES)
            isp.cfg.agc_mode = AGC_MODES[next_mode]
            isp.agc.lut = None                      # rebuild LUT now, no 1 s fade
        elif char == "s":
            cv2.imwrite(f"view_{index:05d}.png", draw_view(raw, image, index, total, isp.cfg))
        elif char == "a":
            index = max(index - 1, 0)
            continue

        if playing or char == "d":
            index = (index + 1) % total
            if index == 0:
                last_index = -1

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
