"""v2 vs v3 on the same raw frames.
python compare_versions.py data.bin out_dir"""
import os, subprocess, sys
import cv2
import numpy as np
from isp import H, W, ISP, Params, NB, BSH, MAXV, find_bad_pixels


class LegacyISP(ISP):
    """v1 tone curve: per-frame-adaptive plateau-limited histogram equalisation blended with linear."""
    def _build_lut(self, hist):
        Hs = self.Hs
        N = float(Hs.sum()) + 1e-6
        c = np.cumsum(Hs)
        edges = (np.arange(NB) + 1) << BSH
        lo = float(np.interp(0.005 * N, c, edges)) - 8
        hi = float(np.interp(0.9995 * N, c, edges))
        min_span = max(150.0, 50.0 * self.sigma_n)
        if hi - lo < min_span:
            mid = 0.5 * (hi + lo); lo, hi = mid - 0.5 * min_span, mid + 0.5 * min_span
        xs = (np.arange(NB) + 0.5) * (1 << BSH)
        inside = (xs >= lo) & (xs <= hi)
        plateau = 1.0 * float(Hs[inside].sum() + 1e-6) / max(1.0, (hi - lo) / (1 << BSH))
        cdf = np.cumsum(np.where(inside, np.minimum(Hs, plateau), 0.0)); cdf /= cdf[-1] + 1e-9
        lin = np.clip((xs - lo) / (hi - lo), 0, 1)
        lb = 0.5 * lin + 0.5 * cdf
        lb = np.where(xs < lo, 0.0, np.where(xs > hi, 1.0, lb))
        self.lut = (np.interp(np.arange(MAXV + 1), xs, lb) * 255.0).astype(np.float32)
        self.lo, self.hi = lo, hi


def label(img, text):
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(img, (0, 0), (W, 30), (0, 0, 0), -1)
    cv2.putText(img, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return img


path, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
n = os.path.getsize(path) // (W * H * 2)
mm = np.memmap(path, dtype="<u2", mode="r", shape=(n, H, W))
bad = find_bad_pixels(np.asarray(mm[np.linspace(0, n - 1, min(96, n)).astype(int)]))
old = ISP(Params(top_gain=0.0), bad)   # v3 without compact-anomaly boost
new = ISP(Params(), bad)                                                      # v3
ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{2*W}x{H}",
                       "-r", "25", "-i", "-", "-c:v", "libx264", "-crf", "12", "-pix_fmt", "yuv420p",
                       f"{out}/compare_v3_vs_v31.mp4"], stdin=subprocess.PIPE)
stills = {}
for i in range(n):
    raw = np.asarray(mm[i])
    a, b = old.process(raw), new.process(raw)
    fr = np.hstack([label(a, "v3: before compact-anomaly boost"), label(b, "v3.1: + compact hot-spot boost")])
    ff.stdin.write(fr.tobytes())
    if i in (100, 165, 195):
        stills[i] = fr
ff.stdin.close(); ff.wait()
cv2.imwrite(f"{out}/compare_v3_vs_v31_stills.png", np.vstack(list(stills.values())))
print("done", n, "frames")
