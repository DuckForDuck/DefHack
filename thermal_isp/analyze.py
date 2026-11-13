"""
Data exploration: python analyze.py data.bin
Prints what still stands in the way of a good image after NUC, and saves out/analysis.png.
Use it to decide: row_fpn on/off, shutter frames present?, noise level, dynamic range, bad-pixel count.
"""
import os
import sys

import cv2
import numpy as np

from isp import H, W, find_bad_pixels

path = sys.argv[1]
n = os.path.getsize(path) // (W * H * 2)
mm = np.memmap(path, dtype="<u2", mode="r", shape=(n, H, W))
print(f"{n} frames ({n / 25:.1f} s @25 fps)")

st = np.array([[f.mean(), f.std(), np.percentile(f[::4, ::4], 1), np.percentile(f[::4, ::4], 99.9)]
               for f in (np.asarray(mm[i]) for i in range(n))])
dm = np.abs(np.diff(st[:, 0]))
print(f"frame mean: min {st[:,0].min():.0f} max {st[:,0].max():.0f}; "
      f"frame std median {np.median(st[:,1]):.0f}, min {st[:,1].min():.0f}")
low = np.nonzero(st[:, 1] < 0.35 * np.median(st[:, 1]))[0]
print(f"suspected shutter/flat frames: {low.tolist()[:20]}{' ...' if len(low) > 20 else ''}")
print(f"largest frame-to-frame mean jumps at: {np.argsort(dm)[-5:][::-1].tolist()} (values {np.sort(dm)[-5:][::-1].round(0).tolist()})")
print(f"1%..99.9% range of frame 0: {st[0,2]:.0f}..{st[0,3]:.0f}  -> naive >>6 uses {(st[0,3]-st[0,2])/64:.0f} of 256 levels")

idx = np.linspace(0, n - 1, min(64, n)).astype(int)
stack = np.asarray(mm[idx]).astype(np.float32)
sd = stack.std(0)
pairs = [np.asarray(mm[i + 1]).astype(np.float32) - np.asarray(mm[i]) for i in idx[idx < n - 1][:16]]
sig_t = 1.4826 * float(np.median(np.abs(np.stack(pairs)))) / 1.414
print(f"temporal noise sigma (adjacent-frame diffs, robust): {sig_t:.2f} counts")
sd = np.full_like(sd, sig_t)   # used below as the noise reference (sequence-wide std is polluted by drift/motion)
bad = find_bad_pixels(stack.astype(np.uint16))
print(f"bad pixels (static map): {int(bad.sum())} ({100*bad.mean():.3f}%)")

m = stack.mean(0)
hp_c = m - cv2.blur(m, (15, 1))
hp_r = m - cv2.blur(m, (1, 15))
col = np.clip(hp_c, -20, 20).mean(0)
row = np.clip(hp_r, -20, 20).mean(1)
print(f"residual column stripe std {col.std():.2f} counts | row stripe std {row.std():.2f} counts "
      f"(noise sigma {np.median(sd):.2f})")
print("-> enable --row-fpn only if row stripe std is clearly above ~noise/sqrt(640)*3 and visible in the image")

os.makedirs("out", exist_ok=True)
f0 = np.asarray(mm[0]).astype(np.float32)
lo, hi = np.percentile(f0, [0.5, 99.5])
img = np.clip((f0 - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)
cv2.imwrite("out/analysis.png", np.hstack([img, np.clip(sd / np.median(sd) * 60, 0, 255).astype(np.uint8)]))
print("saved out/analysis.png (left: frame 0 percentile-stretched, right: temporal-std map)")
