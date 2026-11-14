"""
make_synthetic.py - generate a realistic post-NUC 14-bit test sequence (for development
when the real .bin is not at hand). Contains exactly the problems the ISP must solve:

  * narrow scene range (~400 counts out of 16384)  -> flat grey when scaled naively
  * a very hot object appearing mid-sequence       -> 'lost scene' test
  * a warm walking target                          -> ghosting test
  * residual column FPN + per-frame row noise      -> stripes
  * dead / stuck / blinking pixels                 -> speckles
  * temporal noise, slow drift and an FFC offset jump -> flicker test

Usage: python make_synthetic.py out.bin [n_frames] [--clean]
"""
import sys
import numpy as np
import cv2

W, H = 640, 512


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "synthetic.bin"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 200
    noise = 0.0 if "--clean" in sys.argv else 1.0   # --clean: same scene, no random noise
    rng = np.random.default_rng(1)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)

    # static scene: sky gradient, ground, buildings, trees (texture), road
    scene = 7000 + 120 * (yy / H)                          # sky -> ground gradient
    scene[yy > 300] = 7250 + 60 * ((yy[yy > 300] - 300) / 212)
    for x0, x1, y0, top in [(40, 160, 300, 180), (190, 260, 300, 120), (420, 600, 300, 200)]:
        scene[top:y0, x0:x1] = 7330
        for wy in range(top + 15, y0 - 15, 30):                  # windows, a bit warmer
            for wx in range(x0 + 10, x1 - 15, 25):
                scene[wy:wy + 12, wx:wx + 10] = 7390
    tex = cv2.GaussianBlur(rng.normal(0, 1, (H, W)).astype(np.float32), (0, 0), 2.5)
    trees = (xx > 270) & (xx < 410) & (yy > 210) & (yy < 300)
    scene[trees] = 7280 + 180 * tex[trees]
    scene[(yy > 420) & (yy < 450)] = 7200                     # cool road
    scene += 6 * cv2.GaussianBlur(rng.normal(0, 1, (H, W)).astype(np.float32), (0, 0), 1.2)
    scene = cv2.GaussianBlur(scene, (0, 0), 0.8)

    col_fpn = rng.normal(0, 5, W).astype(np.float32)
    dead = rng.random((H, W)) < 0.0004
    dead_val = rng.choice([0, 16383, 9000], dead.sum())
    blink = rng.random((H, W)) < 0.0002

    with open(out, "wb") as fh:
        for i in range(n):
            f = scene.copy()
            # walking person (warm), moving right
            px, py = 30 + i * 2.8, 330
            person = np.exp(-(((xx - px) / 9) ** 2 + ((yy - py) / 34) ** 2)) * 260
            f += person
            # hot engine / exhaust appears at frame 80 and stays
            if i >= 80:
                hot = np.exp(-(((xx - 520) / 18) ** 2 + ((yy - 380) / 12) ** 2))
                f += 6000 * hot
            drift = 40 * np.sin(i / 60.0)                      # sensor drift
            if i >= 140:
                drift += 35                                    # FFC offset jump
            f += drift
            f += col_fpn[None, :]
            f += noise * rng.normal(0, 2.5, H)[:, None]        # row noise
            f += noise * rng.normal(0, 8, (H, W))              # temporal noise
            f[dead] = dead_val
            bl = blink & (rng.random((H, W)) < 0.3)
            f[bl] += 2500
            fh.write(np.clip(f, 0, 16383).astype("<u2").tobytes())
    print(f"wrote {n} frames to {out}")


if __name__ == "__main__":
    main()
