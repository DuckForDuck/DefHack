"""
make_farm.py - synthetic scene with the raw levels measured on the real file:
road 7500-7650, sun-warmed roofs ~7790 (large areas), animals 7850-8350 (small, moving).
Used to check that roofs stay grey, animals stay brightest, and brightness does not
pump when animals or a hot object enter the frame.

Usage: python make_farm.py out.bin [n_frames]
"""
import sys
import numpy as np
import cv2

W, H = 640, 512


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "farm.bin"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 200
    rng = np.random.default_rng(3)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)

    scene = 7480 + 40 * (yy / H)                                  # field / ground
    scene[(yy > 330) & (yy < 400)] = 7500 + 150 * (xx[(yy > 330) & (yy < 400)] / W)   # road 7500..7650
    roofs = [(30, 260, 60, 250), (300, 620, 40, 200), (380, 600, 420, 500)]
    for x0, x1, y0, y1 in roofs:                                  # large warm roofs ~7790
        scene[y0:y1, x0:x1] = 7760 + 30 * ((xx[y0:y1, x0:x1] - x0) / (x1 - x0))
        scene[y0 + 8:y1 - 8:40, x0:x1] = 7720                     # roof seams (texture)
    scene += 5 * cv2.GaussianBlur(rng.normal(0, 1, (H, W)).astype(np.float32), (0, 0), 1.5)
    scene = cv2.GaussianBlur(scene, (0, 0), 0.8)
    col_fpn = rng.normal(0, 4, W).astype(np.float32)

    animals = [(-40, 290, 2.2, 8350), (700, 370, -1.6, 7900), (100, 300, 0.0, 8100)]
    with open(out, "wb") as fh:
        for i in range(n):
            f = scene.copy()
            for x0, y, speed, temp in animals:
                if i < 60 and speed == 0.0:
                    continue                                      # third animal appears at frame 60
                x = x0 + speed * i
                blob = np.exp(-(((xx - x) / 14) ** 2 + ((yy - y) / 8) ** 2))
                f = np.maximum(f, f + (temp - 7500) * blob * (f < 7700))  # on the ground only
            if 120 <= i < 160:                                    # hot exhaust passes for 40 frames
                f += 5000 * np.exp(-(((xx - 560) / 10) ** 2 + ((yy - 300) / 6) ** 2))
            f += 25 * np.sin(i / 50.0) + col_fpn[None, :]
            f += rng.normal(0, 2, H)[:, None] + rng.normal(0, 7, (H, W))
            fh.write(np.clip(f, 0, 16383).astype("<u2").tobytes())
    print(f"wrote {n} frames to {out}")


if __name__ == "__main__":
    main()
