"""Synthetic post-NUC test sequence (stand-in for the real .bin): python make_synth.py out.bin [frames]"""
import sys
import numpy as np

H, W = 512, 640
rng = np.random.default_rng(1)
n = int(sys.argv[2]) if len(sys.argv) > 2 else 150

yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
bg = 7000 + 700 * (yy / H) + 300 * np.sin(xx / 90.0)
bg[330:, :] += 500 + 120 * np.sin(xx[330:, :] / 13.0) * np.sin(yy[330:, :] / 9.0)   # textured ground
for x0, x1, y0, h in [(60, 200, 150, 180), (260, 330, 210, 120), (420, 600, 120, 210)]:   # buildings
    bg[y0:y0 + h, x0:x1] += 900
    for wy in range(y0 + 15, y0 + h - 10, 28):
        for wx in range(x0 + 10, x1 - 10, 22):
            bg[wy:wy + 10, wx:wx + 8] += 450
vig = -400 * (((xx - W / 2) / W) ** 2 + ((yy - H / 2) / H) ** 2)
col_fpn = rng.normal(0, 5, W).astype(np.float32)
bad = rng.choice(H * W, 160, replace=False)
bad_val = np.where(rng.random(160) < 0.5, 700, -700).astype(np.float32)
stuck = rng.choice(H * W, 20, replace=False)

out = np.empty((n, H, W), np.uint16)
for t in range(n):
    f = bg + vig + 0.6 * t
    # hot object (engine) + moving warm person-sized blob
    f[400:408, 300:310] += 5200
    px = int(40 + 3 * t) % (W - 40)
    f[360:420, px:px + 16] += 1100
    f += rng.normal(0, 5, (H, W)).astype(np.float32)           # temporal noise
    f += col_fpn[None, :] + 1.5 * np.sin(t / 20.0)             # residual column FPN
    f += rng.normal(0, 1.2, H).astype(np.float32)[:, None]      # row noise
    ff = f.ravel()
    ff[bad] += bad_val
    ff[stuck] = 0
    if t in (90, 91, 92):                                       # shutter event: flat frames
        f = np.full((H, W), f.mean(), np.float32) + rng.normal(0, 5, (H, W)).astype(np.float32) + col_fpn[None, :]
    if t >= 93:
        f += 40
    out[t] = np.clip(f, 0, 16383).astype(np.uint16)
out.tofile(sys.argv[1])
print("wrote", sys.argv[1], out.shape)
