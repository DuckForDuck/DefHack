"""
One-command run:
    python run.py data.bin [more.bin ...] -o out/

Writes out/<name>_whitehot.mp4 (ISP result), out/<name>_compare.mp4 (naive 14->8 shift | ISP),
out/<name>_before_after.png and prints speed + flicker numbers.
"""
import argparse
import os
import subprocess
import time

import cv2
import numpy as np

from isp import H, W, ISP, Params, find_bad_pixels


class Writer:
    def __init__(self, path, w, h, fps):
        self.p = subprocess.Popen(
            ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "gray", "-s", f"{w}x{h}",
             "-r", str(fps), "-i", "-", "-c:v", "libx264", "-crf", "12", "-preset", "medium",
             "-pix_fmt", "yuv420p", path], stdin=subprocess.PIPE)

    def write(self, img):
        self.p.stdin.write(np.ascontiguousarray(img).tobytes())

    def close(self):
        self.p.stdin.close()
        self.p.wait()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("-o", "--out", default="out")
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--no-compare", action="store_true")
    ap.add_argument("--row-fpn", action="store_true")
    ap.add_argument("--black-hot", action="store_true")
    ap.add_argument("--detail-gain", type=float, default=None)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    for path in a.inputs:
        name = os.path.splitext(os.path.basename(path))[0]
        nfr = os.path.getsize(path) // (W * H * 2)
        mm = np.memmap(path, dtype="<u2", mode="r", shape=(nfr, H, W))
        n = min(nfr, a.max_frames) if a.max_frames else nfr
        print(f"{name}: {nfr} frames, processing {n}")

        # bad-pixel map: sampled over the sequence (FPGA: collected once at start-up / from calibration)
        idx = np.linspace(0, n - 1, min(96, n)).astype(int)
        bad = find_bad_pixels(np.asarray(mm[idx]), Params().bp_k)
        print(f"  bad pixels found: {int(bad.sum())}")

        p = Params(row_fpn=a.row_fpn, white_hot=not a.black_hot)
        if a.detail_gain is not None:
            p.detail_gain = a.detail_gain
        isp = ISP(p, bad)
        w_out = Writer(f"{a.out}/{name}_whitehot.mp4", W, H, a.fps)
        w_cmp = None if a.no_compare else Writer(f"{a.out}/{name}_compare.mp4", 2 * W, H, a.fps)

        times, m_naive, m_isp, keep = [], [], [], {}
        for i in range(n):
            raw = np.asarray(mm[i])
            t0 = time.perf_counter()
            out = isp.process(raw)
            times.append((time.perf_counter() - t0) * 1e3)
            naive = (raw >> 6).astype(np.uint8)
            w_out.write(out)
            if w_cmp:
                w_cmp.write(np.hstack([naive, out]))
            m_naive.append(naive.mean())
            m_isp.append(out.mean())
            if i in (n // 4, n // 2, 3 * n // 4):
                keep[i] = np.hstack([naive, out])
        w_out.close()
        if w_cmp:
            w_cmp.close()
        if keep:
            cv2.imwrite(f"{a.out}/{name}_before_after.png", np.vstack(list(keep.values())))

        t = np.array(times[1:])
        fl = lambda m: float(np.mean(np.abs(np.diff(m))))
        print(f"  ISP time/frame: mean {t.mean():.1f} ms, p95 {np.percentile(t, 95):.1f} ms, "
              f"max {t.max():.1f} ms  (budget 40 ms)")
        print(f"  mean-brightness flicker (mean |dB| per frame, 8-bit levels): "
              f"naive {fl(m_naive):.3f} | ISP {fl(m_isp):.3f}")


if __name__ == "__main__":
    main()
