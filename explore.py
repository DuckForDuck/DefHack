"""
explore.py - look at the raw data before designing anything.

Prints and saves the numbers that justify each ISP stage:
  value range actually used, temporal noise, column / row stripe strength,
  bad pixel count, frozen frames (FFC), per-frame mean drift.

Usage: python explore.py data.bin [out_dir]
"""
import os
import sys
import numpy as np
import cv2

W, H = 640, 512


def main():
    path = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else "explore_out"
    os.makedirs(out, exist_ok=True)
    fr = np.memmap(path, dtype="<u2", mode="r")
    n = fr.size // (W * H)
    fr = fr[: n * W * H].reshape(n, H, W)
    print(f"{path}: {n} frames of {W}x{H}")

    sub = fr[:: max(1, n // 64)].astype(np.float32)
    print(f"value range: min {sub.min():.0f}  p0.1 {np.percentile(sub, .1):.0f}  "
          f"median {np.median(sub):.0f}  p99.9 {np.percentile(sub, 99.9):.0f}  max {sub.max():.0f}")
    p1, p99 = np.percentile(sub, (1, 99))
    print(f"1..99% span = {p99 - p1:.0f} counts = {100 * (p99 - p1) / 16383:.1f}% of 14-bit range")

    k = min(n, 32)
    stack = fr[:k].astype(np.float32)
    tstd = stack.std(axis=0)
    print(f"temporal noise (median per-pixel std over {k} frames): {np.median(tstd):.2f} counts")
    mean = stack.mean(axis=0)
    hp = mean - cv2.blur(mean, (15, 1))
    col = np.clip(hp, -30, 30).mean(axis=0)
    hpr = mean - cv2.blur(mean, (1, 15))
    row = np.clip(hpr, -30, 30).mean(axis=1)
    print(f"static column stripe std: {col.std():.2f} counts   static row stripe std: {row.std():.2f}")
    rowt = [np.clip(f - cv2.blur(f, (1, 15)), -30, 30).mean(axis=1) for f in stack[:8]]
    print(f"per-frame row noise std: {np.std(np.array(rowt) - np.mean(rowt, 0)):.2f} counts")

    dev = np.abs(mean - cv2.medianBlur(mean, 3))
    med_std = np.median(tstd)
    bad = (dev > 10 * max(med_std, 1)) | (tstd < 0.1 * med_std) | (tstd > 6 * med_std)
    print(f"suspect bad pixels: {bad.sum()} ({100 * bad.mean():.3f}%)")

    means = fr.reshape(n, -1)[:, ::97].astype(np.float32).mean(axis=1)
    jumps = np.abs(np.diff(means))
    frozen = [i for i in range(1, n) if np.array_equal(fr[i, ::16, ::16], fr[i - 1, ::16, ::16])]
    print(f"frame mean: {means.min():.0f}..{means.max():.0f}, biggest jump {jumps.max():.1f} at "
          f"frame {int(jumps.argmax()) + 1}; frozen/duplicate frames: {len(frozen)}")

    def save(name, img):
        lo, hi = np.percentile(img, (1, 99))
        cv2.imwrite(os.path.join(out, name),
                    np.clip((img - lo) * 255 / max(hi - lo, 1e-6), 0, 255).astype(np.uint8))
    save("mean_frame.png", mean)
    save("temporal_std.png", tstd)
    save("highpass_mean.png", hp)
    np.savetxt(os.path.join(out, "frame_means.txt"), means, fmt="%.2f")
    print(f"diagnostic images in {out}/")


if __name__ == "__main__":
    main()
