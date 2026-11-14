"""Build a supervised dataset for a learned image-enhancement model from a real raw .bin sequence.

Input  (noisy):  the real raw 14-bit post-NUC frame, untouched (real noise, column FPN, bad pixels).
Target (clean):  the same frame after bad-pixel / column-FPN removal and an offline, NON-causal, motion-compensated
                 average of +-R neighbouring real frames (global similarity + local flow alignment, per-pixel
                 gating so moving objects are not ghosted).

Because both come from the same real sensor, there is no synthetic-to-real gap in the noise.
Fresh noisy inputs for the same clean targets can be drawn with NoiseBank (real residuals transplanted onto clean patches).

Usage:
    python make_dataset.py data.bin -o dataset            # patches + splits + report
    python make_dataset.py data.bin -o dataset --full     # also write <name>_clean.bin (same layout as the input)
    python make_dataset.py data.bin -o dataset --max-frames 40 --radius 8
"""
import argparse
import json
import os
import time

import cv2
import numpy as np

from isp import ISP, Params, find_bad_pixels, H, W


# ----------------------------------------------------------------------------- preprocessing
def preprocess(raw, bad_mask):
    """Stateless-looking pass over the sequence using the ISP's own stage 1-2 (bad pixels, column FPN).
    Returns float32 frames, 8-bit LK feature images and a validity flag per frame (False = shutter/flat frame)."""
    isp = ISP(Params(), bad_mask=bad_mask)
    xs, feats, ok = [], [], []
    for i in range(len(raw)):
        x = np.asarray(raw[i]).astype(np.float32)
        if isp._shutter_guard(x):                      # flat frame during FFC: unusable as input or as neighbour
            xs.append(x), feats.append(None), ok.append(False)
            continue
        x = isp._fix_bad(x)
        sn = isp._est_noise(x)
        isp.sigma_n = sn if i == 0 or isp.sigma_n is None else isp.sigma_n + 0.1 * (sn - isp.sigma_n)
        x = isp._col_fpn(x)
        xs.append(x)
        feats.append(isp._features(x))
        ok.append(True)
    return isp, xs, feats, ok


# ----------------------------------------------------------------------------- clean target
def _smooth01(u):
    u = np.clip(u, 0.0, 1.0)
    return u * u * (3.0 - 2.0 * u)


def adjacent_motion(isp, feats, ok):
    """3x3 similarity matrices A[k]: frame k -> frame k+1 (adjacent frames only: LK is reliable at 6-14 px/frame,
    but fails for neighbours several frames apart). Falls back to phase correlation, then to the previous estimate."""
    A, last = [None] * (len(feats) - 1), np.eye(3, dtype=np.float64)
    for k in range(len(feats) - 1):
        if not (ok[k] and ok[k + 1]):
            continue
        M, _ = isp._motion_affine(feats[k], feats[k + 1])
        if M is None:
            Mt, resp = isp._motion_translate(feats[k], feats[k + 1])
            M = Mt if resp >= isp.p.min_resp else None
        if M is not None:
            last = np.vstack([M.astype(np.float64), [0, 0, 1]])
        A[k] = last.copy() if M is None else np.vstack([M.astype(np.float64), [0, 0, 1]])
    return A


def aligned_neighbours(isp, xs, feats, ok, A, t, offsets):
    """Yield (warped neighbour, weight map) for every usable neighbour j = t + offset, aligned by chaining adjacent motions."""
    n = len(xs)
    xt = xs[t]
    sig = float(isp._est_noise(xt))
    sig_d = sig * np.sqrt(2.0)                         # noise of a difference of two independent noisy frames
    ones = np.ones((H, W), np.uint8)
    P = {0: np.eye(3)}
    for sgn in (1, -1):
        cur = np.eye(3)
        for o in range(1, max(abs(x) for x in offsets) + 1):
            j = t + sgn * o
            if j < 0 or j >= n or not ok[j]:
                break
            k = j - 1 if sgn > 0 else j                # adjacent matrix between j and j-sgn
            if A[k] is None:
                break
            cur = cur @ (np.linalg.inv(A[k]) if sgn > 0 else A[k])
            P[sgn * o] = cur.copy()
    for o in offsets:
        if o not in P:
            continue
        j = t + o
        M = P[o][:2].astype(np.float32)
        if np.abs(M[:, 2]).max() > 6 * isp.p.max_shift:
            continue
        warped = isp._warp_state(xs[j], M, xt)
        valid = cv2.erode(cv2.warpAffine(ones, M, (W, H), flags=cv2.INTER_NEAREST), np.ones((9, 9), np.uint8)).astype(np.float32)
        a = cv2.blur(np.abs(warped - xt), (3, 3))      # pooled: motion is coherent, noise is not
        wgt = (1.0 - _smooth01((a / sig_d - 1.3) / 1.2)) * valid
        yield warped, wgt


def build_target(isp, xs, feats, ok, A, t, radius, parity=None, include_self=True):
    """Weighted average of frame t and its aligned neighbours. parity 0/1 keeps only even/odd offsets (split-half test)."""
    offs = [o for o in range(-radius, radius + 1) if o != 0 and (parity is None or (abs(o) % 2) == parity)]
    num = xs[t].copy() if include_self else np.zeros((H, W), np.float32)
    den = np.ones((H, W), np.float32) if include_self else np.zeros((H, W), np.float32)
    for warped, wgt in aligned_neighbours(isp, xs, feats, ok, A, t, offs):
        num += wgt * warped
        den += wgt
    return num / np.maximum(den, 1e-3), den


# ----------------------------------------------------------------------------- validation of the target
def robust_sigma(a):
    return 1.4826 * float(np.median(np.abs(a - np.median(a))))


def sharp_proxy(img):
    """Mean gradient magnitude on the strongest 2% edges after light smoothing (noise-insensitive)."""
    g = cv2.GaussianBlur(img, (0, 0), 1.5)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1)
    m = np.hypot(gx, gy)
    return float(np.mean(np.sort(m.ravel())[-int(0.02 * m.size):]))


def validate_target(isp, xs, feats, ok, A, frames, radius):
    rows = []
    for t in frames:
        tgt, den = build_target(isp, xs, feats, ok, A, t, radius)
        a, da = build_target(isp, xs, feats, ok, A, t, radius, parity=0, include_self=False)
        b, db = build_target(isp, xs, feats, ok, A, t, radius, parity=1, include_self=False)
        sel = (da >= 2.5) & (db >= 2.5)                # pixels where both halves have enough support
        s_raw = float(isp._est_noise(xs[t]))
        s_ab = robust_sigma((a - b)[sel]) / np.sqrt(2.0) if sel.any() else float("nan")
        pred = s_raw / np.sqrt(float(np.mean(da[sel]))) if sel.any() else float("nan")
        rows.append(dict(t=int(t), sigma_raw=s_raw, n_eff=float(den.mean()), sigma_half_measured=s_ab, sigma_half_ideal=pred,
                         sharp_ratio=sharp_proxy(tgt) / sharp_proxy(xs[t]), support=float(sel.mean())))
    return rows


# ----------------------------------------------------------------------------- noise bank (fresh noisy inputs)
class NoiseBank:
    """Real noise residuals (raw - clean) taken from training frames. The sensor noise here is NOT white: it is
    spatially correlated and dominated by low frequencies (measured, see the report), so a parametric Gaussian model
    does not reproduce it. Transplanting real residual patches onto clean patches does.
    add(clean_patch, rng) -> noisy uint16 patch (random residual, random crop, random flips)."""

    def __init__(self, residuals, frames):
        self.res = residuals.astype(np.int16)           # (K, H, W)
        self.frames = np.asarray(frames)

    @staticmethod
    def build(raw, clean, frames):
        res = np.stack([np.asarray(raw[t]).astype(np.int32) - np.asarray(clean[t]).astype(np.int32) for t in frames])
        return NoiseBank(np.clip(res, -2000, 2000), frames)

    def save(self, path):
        np.savez_compressed(path, residuals=self.res, frames=self.frames)

    @staticmethod
    def load(path):
        z = np.load(path)
        return NoiseBank(z["residuals"], z["frames"])

    def add(self, clean, rng, exclude_frame=None, min_gap=12):
        h, w = clean.shape
        ok = np.arange(len(self.frames)) if exclude_frame is None else np.nonzero(np.abs(self.frames - exclude_frame) >= min_gap)[0]
        r = self.res[int(rng.choice(ok))]
        y, x = int(rng.integers(0, r.shape[0] - h + 1)), int(rng.integers(0, r.shape[1] - w + 1))
        r = r[y:y + h, x:x + w]
        if rng.random() < 0.5:
            r = r[:, ::-1]
        if rng.random() < 0.5:
            r = r[::-1]
        return np.clip(clean.astype(np.int32) + r, 0, 16383).astype(np.uint16)


def noise_stats(r):
    """sigma, column-FPN std, lag-1 correlations and radial power spectrum (low -> high frequency, normalised to the highest band)."""
    r = r.astype(np.float32)
    h, w = r.shape
    f = np.abs(np.fft.fftshift(np.fft.fft2(r - r.mean()))) ** 2
    yy, xx = np.mgrid[0:h, 0:w]
    rad = np.hypot((yy - h // 2) / h, (xx - w // 2) / w)
    bands = [float(f[(rad >= lo) & (rad < hi)].mean()) for lo, hi in ((0, .02), (.02, .05), (.05, .1), (.1, .2), (.2, .35), (.35, .5))]
    return dict(sigma=robust_sigma(r), col_std=float(np.std(r.mean(0))),
                rho_x=float(np.corrcoef(r[:, :-1].ravel(), r[:, 1:].ravel())[0, 1]),
                rho_y=float(np.corrcoef(r[:-1].ravel(), r[1:].ravel())[0, 1]),
                psd=[round(v / bands[-1], 2) for v in bands])


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bin")
    ap.add_argument("-o", "--out", default="dataset")
    ap.add_argument("--radius", type=int, default=8, help="neighbours used on each side for the clean target")
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--patch", type=int, default=128)
    ap.add_argument("--per-frame", type=int, default=6, help="random patches per frame")
    ap.add_argument("--full", action="store_true", help="also write the full-frame clean sequence as a .bin")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    raw = np.memmap(a.bin, dtype="<u2", mode="r").reshape(-1, H, W)
    n = len(raw) if not a.max_frames else min(a.max_frames, len(raw))
    raw = raw[:n]
    name = os.path.splitext(os.path.basename(a.bin))[0]
    rng = np.random.default_rng(a.seed)
    t0 = time.time()

    bad = find_bad_pixels(np.asarray(raw[np.linspace(0, n - 1, min(96, n)).astype(int)]))
    isp, xs, feats, ok = preprocess(raw, bad)
    print("preprocessed %d frames (%d usable), %d bad pixels, %.1f s" % (n, sum(ok), bad.sum(), time.time() - t0))
    A = adjacent_motion(isp, feats, ok)
    print("adjacent motion: %d of %d links, %.1f s" % (sum(m is not None for m in A), len(A), time.time() - t0))

    # time-based split with gaps >= radius so that neighbouring frames never leak across splits
    gap = a.radius + 2
    c1, c2 = int(0.70 * n), int(0.85 * n)
    split_of = np.full(n, -1)
    split_of[: c1 - gap] = 0
    split_of[c1: c2 - gap] = 1
    split_of[c2:] = 2
    names = ["train", "val", "test"]

    P, ppf = a.patch, a.per_frame
    store = {s: dict(raw=[], clean=[], frame=[], y=[], x=[]) for s in names}
    clean_full = np.memmap(os.path.join(a.out, name + "_clean.bin"), dtype="<u2", mode="w+", shape=(n, H, W)) if a.full else None
    for t in range(n):
        if not ok[t]:
            continue
        tgt, den = build_target(isp, xs, feats, ok, A, t, a.radius)
        c16 = np.clip(np.rint(tgt), 0, 16383).astype(np.uint16)
        if clean_full is not None:
            clean_full[t] = c16
        if split_of[t] < 0:
            continue
        s = names[split_of[t]]
        for _ in range(ppf):
            y, x = int(rng.integers(0, H - P + 1)), int(rng.integers(0, W - P + 1))
            if den[y:y + P, x:x + P].mean() < 3.0:     # skip patches without enough temporal support (frame borders)
                continue
            st = store[s]
            st["raw"].append(np.asarray(raw[t, y:y + P, x:x + P]).copy())
            st["clean"].append(c16[y:y + P, x:x + P])
            st["frame"].append(t), st["y"].append(y), st["x"].append(x)
        if t % 20 == 0:
            print("  frame %d/%d  %.0f s" % (t, n, time.time() - t0), flush=True)
    if clean_full is not None:
        clean_full.flush()

    for s in names:
        st = store[s]
        if not st["raw"]:
            continue
        np.savez_compressed(os.path.join(a.out, "%s_%s.npz" % (name, s)), raw=np.stack(st["raw"]), clean=np.stack(st["clean"]),
                            frame=np.array(st["frame"]), y=np.array(st["y"]), x=np.array(st["x"]))
        print("%-5s %5d patches %dx%d" % (s, len(st["raw"]), P, P))

    # noise bank from TRAIN frames only; checked on TEST frames (bank residual transplanted onto a test clean frame
    # must look like the real test residual)
    train_frames = [t for t in range(n) if split_of[t] == 0 and ok[t]]
    test_frames = [t for t in range(n) if split_of[t] == 2 and ok[t]]
    bank = None
    noise_check = None
    if clean_full is not None and len(train_frames) >= 4 and test_frames:
        bank_frames = [train_frames[i] for i in np.linspace(0, len(train_frames) - 1, min(24, len(train_frames))).astype(int)]
        bank = NoiseBank.build(raw, clean_full, bank_frames)
        bank.save(os.path.join(a.out, name + "_noise_bank.npz"))
        real_s, syn_s = [], []
        for t in test_frames[:: max(1, len(test_frames) // 6)]:
            c = np.asarray(clean_full[t])
            real_s.append(noise_stats(np.asarray(raw[t]).astype(np.float32) - c))
            syn_s.append(noise_stats(bank.add(c, rng).astype(np.float32) - c))
        med = lambda L, k: float(np.median([x[k] for x in L]))
        noise_check = {k: (med(real_s, k), med(syn_s, k)) for k in ("sigma", "col_std", "rho_x", "rho_y")}
        noise_check["psd_real"] = np.median([x["psd"] for x in real_s], 0).tolist()
        noise_check["psd_bank"] = np.median([x["psd"] for x in syn_s], 0).tolist()
        print("noise bank (%d residual frames from train) vs real residual on test frames:" % len(bank_frames))
        for k in ("sigma", "col_std", "rho_x", "rho_y"):
            print("  %-8s real %.3f   bank %.3f" % (k, *noise_check[k]))
        print("  PSD low->high real", noise_check["psd_real"], " bank", noise_check["psd_bank"])
    else:
        print("noise bank skipped (needs --full and enough frames)")

    # validate the clean target on a sample of frames
    sample = [t for t in np.linspace(a.radius, n - a.radius - 1, min(12, n)).astype(int) if ok[t]]
    rows = validate_target(isp, xs, feats, ok, A, sample, a.radius)
    json.dump(dict(rows=rows, radius=a.radius, noise_check=noise_check), open(os.path.join(a.out, name + "_report.json"), "w"), indent=2)
    r = lambda k: np.nanmedian([x[k] for x in rows])
    print("target quality (median over %d frames, radius %d):" % (len(rows), a.radius))
    print("  raw noise sigma            %.2f counts" % r("sigma_raw"))
    print("  effective frames averaged  %.1f" % r("n_eff"))
    print("  half-set noise measured    %.2f  vs ideal %.2f   (excess = misalignment / correlated noise)" % (r("sigma_half_measured"), r("sigma_half_ideal")))
    print("  sharpness ratio target/raw %.3f  (1.0 = no blur)" % r("sharp_ratio"))
    print("done in %.0f s" % (time.time() - t0))


if __name__ == "__main__":
    main()
