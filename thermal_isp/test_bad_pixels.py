"""Self-test for find_bad_pixels: injected defects must be found, false positives must stay rare,
for short/long and static/moving clips. Usage: python test_bad_pixels.py"""
import sys
import numpy as np
from isp import find_bad_pixels

H, W = 512, 640
rng = np.random.default_rng(3)
yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
BASE = 7000 + 500 * np.sin(xx / 70) * np.cos(yy / 55) + 300 * (yy / H)
IDX = rng.choice(H * W, 100, replace=False)        # 40 hot, 40 cold, 20 stuck
TRUTH = np.zeros(H * W, bool)
TRUTH[IDX] = True
TRUTH = TRUTH.reshape(H, W)


def synth(n, move):
    out = np.empty((n, H, W), np.uint16)
    for t in range(n):
        f = np.roll(BASE, (3 * t, 2 * t) if move else (0, 0), (0, 1)) + rng.normal(0, 6, (H, W)).astype(np.float32)
        g = f.ravel()
        g[IDX[:40]] += 700
        g[IDX[40:80]] -= 700
        g[IDX[80:]] = 7300
        out[t] = np.clip(f, 0, 16383).astype(np.uint16)
    return out


def main():
    ok = True
    for n, move in ((5, False), (15, False), (60, False), (15, True), (60, True)):
        bad = find_bad_pixels(synth(n, move))
        recall = (bad & TRUTH).sum() / TRUTH.sum()
        fp = int((bad & ~TRUTH).sum())
        good = recall >= 0.95 and fp <= 150            # <= 0.05% of 327680 pixels
        ok &= good
        print("%3d frames %-6s recall %.2f  false positives %4d  %s" % (n, "moving" if move else "static", recall, fp, "ok" if good else "FAIL"))
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
