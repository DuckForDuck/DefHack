"""
Streaming-friendly ISP for an uncooled 640x512 microbolometer (post-NUC 14-bit -> stable 8-bit white-hot).   [v3]

Stage order (each stage needs only a few lines of context or last-frame statistics, so it maps to an FPGA pixel stream):

  0  shutter / FFC frame guard        -> freeze the output instead of showing a flat gray frame
  1  bad-pixel replacement            -> static map + median of good neighbours (before any spatial filter)
  2  residual column (row) FPN removal-> vertical stripes the NUC did not remove
  3  motion-adaptive recursive TNR, similarity (shift+roll+zoom) motion from LK corners + dense local flow -> low noise, no ghosting
  4  absolute-anchored tone curve (frame t-1 stats): slew-limited black point, peak-hold white point, gamma -> no equalisation + shadow lift
  5  base / detail split (fast guided filter) -> compress the base with the LUT, keep + boost detail (no halos)
  6  white-hot 8-bit output
"""
from collections import deque
from dataclasses import dataclass
import warnings

import cv2
import numpy as np

H, W = 512, 640
NB = 2048            # histogram bins (14 bit >> 3)
BSH = 3
MAXV = 16383


@dataclass
class Params:
    # --- bad pixels
    bp_k: float = 6.0            # static outlier threshold in robust sigmas
    # --- shutter guard
    shutter_ratio: float = 0.35  # frame spatial std < ratio * recent median => shutter frame
    shutter_max_freeze: int = 30
    # --- residual FPN
    col_fpn: bool = True
    col_win: int = 15            # horizontal window of the high-pass used to see a stripe
    col_alpha: float = 0.1       # temporal IIR of the per-column estimate
    row_fpn: bool = False        # enable only if analyze.py shows row banding
    # --- temporal noise reduction
    kmin: float = 1 / 4          # static-pixel IIR weight  (noise std -> ~0.38x)
    motion: str = 'affine'       # 'affine' (v3: LK corners + RANSAC similarity) | 'translate' (v2)
    local_flow: bool = True      # residual dense flow for parallax
    flow_limit: float = 3.0
    min_resp: float = 0.15       # phase-correlation confidence below which TNR is bypassed
    max_shift: float = 24.0      # px/frame
    t0: float = 1.0              # motion gate: mean|d|/sigma_d below t0 -> pure noise
    t1: float = 2.2              #              above t1 -> motion, pass through (k=1)
    cut_ratio: float = 5.0       # global change => scene cut / FFC recovery, reset state
    # --- AGC (absolute-anchored, no equalisation)
    p_lo: float = 0.01           # slow low percentile -> black point
    hi_pct: float = 0.9995       # per-frame high percentile that feeds the peak-hold
    lo_slew: float = 1.5         # max black-point change, counts/frame
    hi_attack: float = 0.15      # peak-hold attack (fraction per frame)
    hi_decay: float = 0.15       # peak-hold release, counts per frame (very slow)
    hist_alpha: float = 0.03     # temporal smoothing of the low-percentile histogram
    shadow_lift: float = 0.12    # v3: raise dark vegetation without lifting road/roofs (0 = v2)
    gamma: float = 1.15          # >1 keeps big warm surfaces dark/mid, leaves the top for hot anomalies
    min_span_sigma: float = 50.0
    min_span_abs: float = 1000.0 # never stretch less than this many counts over 0..255
    # --- detail layer
    gf_radius: int = 8
    edge_frac: float = 0.05      # guided-filter eps = (edge_frac * span)^2
    detail_gain: float = 2.6
    hot_boost: float = 1.5       # extra gain for warm small detail (animals, hot spots)
    detail_limit: float = 60.0   # soft clip of the detail in 8-bit levels
    core_k: float = 2.0          # coring threshold in units of post-TNR noise sigma
    white_hot: bool = True


def find_bad_pixels(stack, k=6.0):
    """stack: (N,H,W) uint16 sampled over the sequence (FPGA: collected once at start-up)."""
    s = stack.astype(np.float32)
    sub = s[:, ::4, ::4].reshape(len(s), -1)
    fs = sub.std(1)
    keep = fs > 0.5 * np.median(fs)                  # drop shutter / flat frames
    s = s[keep]
    s -= np.median(sub[keep], axis=1)[:, None, None]  # remove global drift / FFC steps before taking stats
    m = s.mean(0)
    sd = s.std(0)
    med_sd = float(np.median(sd)) + 1e-3
    sd_loc = cv2.medianBlur(sd, 5)
    noisy = (sd > 3.0 * sd_loc) & (sd > 2.5 * med_sd)           # blinkers / popcorn
    stuck = (sd < 0.15 * med_sd) | (stack.max(0) == 0) | (stack.min(0) >= MAXV)   # dead / saturated
    med3 = cv2.medianBlur(m, 3)
    d = m - med3
    mad = 1.4826 * float(np.median(np.abs(d))) + 1e-3
    ker = np.ones((3, 3), np.uint8)
    ker[1, 1] = 0
    nb_max = cv2.dilate(d, ker)
    nb_min = cv2.erode(d, ker)
    # isolated single-pixel outliers only: a real small target has a PSF, so its neighbours are elevated too
    hot = (d > k * mad) & (nb_max < 0.35 * d)
    cold = (d < -k * mad) & (nb_min > 0.35 * d)
    return noisy | stuck | hot | cold


class ISP:
    def __init__(self, params=None, bad_mask=None):
        self.p = params or Params()
        self.set_bad_mask(bad_mask)
        self.reset()

    # ------------------------------------------------------------------ setup
    def set_bad_mask(self, bad):
        if bad is None or not bad.any():
            self.bp_y = None
            return
        ys, xs = np.nonzero(bad)
        offs = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        ny = np.stack([np.clip(ys + dy, 0, H - 1) for dy, dx in offs], 1)
        nx = np.stack([np.clip(xs + dx, 0, W - 1) for dy, dx in offs], 1)
        self.bp_y, self.bp_x, self.nb_y, self.nb_x = ys, xs, ny, nx
        self.nb_bad = bad[ny, nx]
        self.n_bad = len(ys)

    def reset(self):
        self.gx, self.gy = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
        self.y = None            # TNR frame buffer
        self.sig_d = None
        self.feat = None
        self.sm = (0.0, 0.0)
        self.M_last = np.float32([[1, 0, 0], [0, 1, 0]])
        self.src = 'none'
        self.diag = (0, 0, 0)
        self.shift = (0, 0, 0)
        self.prev_mean = None
        self.win = cv2.createHanningWindow((W // 2, H // 2), cv2.CV_32F)
        self.sigma_n = 8.0
        self.cs = None           # per-column offset estimate
        self.Hs = None           # smoothed histogram
        self.lut = None
        self.hi_hold = None
        self.lo, self.hi = 0.0, 16383.0
        self.hist_std = deque(maxlen=16)
        self.freeze = 0
        self.last_out = None
        self.cut = False

    # ------------------------------------------------------------------ stages
    def _shutter_guard(self, x):
        s = float(x[::4, ::4].std())
        if len(self.hist_std) >= 4 and s < self.p.shutter_ratio * float(np.median(self.hist_std)):
            self.freeze += 1
            if self.freeze <= self.p.shutter_max_freeze:
                return True
            self.hist_std.clear()          # the scene really changed: accept it
        self.freeze = 0
        self.hist_std.append(s)
        return False

    def _fix_bad(self, x):
        if self.bp_y is None:
            return x
        v = x[self.nb_y, self.nb_x]
        v[self.nb_bad] = np.nan
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            r = np.nanmedian(v, 1)
        r = np.where(np.isnan(r), x[self.bp_y, self.bp_x], r)
        x[self.bp_y, self.bp_x] = r
        return x

    def _col_fpn(self, x):
        p = self.p
        w = p.col_win
        hp = x - cv2.boxFilter(x, -1, (w, 1), borderType=cv2.BORDER_REFLECT)
        T = max(8.0, 5.0 * self.sigma_n)            # clip: edges can leak at most +-T into a column
        e = np.clip(hp, -T, T).mean(0) * (w / (w - 1.0))
        self.cs = e if self.cs is None else self.cs + p.col_alpha * (e - self.cs)
        return x - self.cs[None, :]

    def _row_fpn(self, x):
        hp = x - cv2.boxFilter(x, -1, (1, 15), borderType=cv2.BORDER_REFLECT)
        T = max(8.0, 5.0 * self.sigma_n)
        e = np.clip(hp, -T, T).mean(1) * (15 / 14.0)
        return x - e[:, None]

    def _est_noise(self, x):
        """Spatial noise sigma (Laplacian MAD). Unlike frame differencing it is not polluted by camera/scene motion."""
        l = cv2.Laplacian(x, cv2.CV_32F)[::2, ::2]
        return max(0.5, 1.4826 * float(np.median(np.abs(l))) / 4.4721)

    def _features(self, x):
        f = cv2.resize(x, (W // 2, H // 2), interpolation=cv2.INTER_AREA)
        return cv2.GaussianBlur(f, (0, 0), 1.0)

    def _motion_translate(self, prev, f):
        """v2: global translation by phase correlation, smoothed, held when unreliable (FPGA: projection correlation)."""
        p = self.p
        (dx, dy), resp = cv2.phaseCorrelate(prev - float(prev.mean()), f - float(f.mean()), self.win)
        dx, dy = 2.0 * dx, 2.0 * dy
        if abs(dx) > p.max_shift or abs(dy) > p.max_shift:
            resp = 0.0
        a_s = float(np.clip(resp / p.min_resp, 0.0, 1.0)) if resp < p.min_resp else 1.0
        if resp >= p.min_resp * 0.5 and a_s > 0:
            self.sm = (self.sm[0] + a_s * (dx - self.sm[0]), self.sm[1] + a_s * (dy - self.sm[1]))
        return np.float32([[1, 0, self.sm[0]], [0, 1, self.sm[1]]]), resp

    def _motion_affine(self, prev, f):
        """v3: similarity transform (shift + roll + zoom) from sparse corners tracked with pyramidal LK + RANSAC.
        Handles the camera roll / zoom that a pure translation cannot. Returns (M or None, inlier ratio)."""
        p = self.p
        lo, hi = np.percentile(f[::2, ::2], (1, 99.5))
        sc = 255.0 / max(float(hi - lo), 1.0)
        a = np.clip((prev - lo) * sc, 0, 255).astype(np.uint8)
        b = np.clip((f - lo) * sc, 0, 255).astype(np.uint8)
        pts = cv2.goodFeaturesToTrack(a, 400, 0.005, 6, blockSize=5)
        if pts is None or len(pts) < 30:
            return None, 0.0
        nxt, st, _ = cv2.calcOpticalFlowPyrLK(a, b, pts, None, winSize=(21, 21), maxLevel=3)
        ok = st.ravel() == 1
        if ok.sum() < 25:
            return None, 0.0
        M, inl = cv2.estimateAffinePartial2D(pts[ok], nxt[ok], method=cv2.RANSAC,
                                             ransacReprojThreshold=1.5, maxIters=300, confidence=0.99)
        if M is None:
            return None, 0.0
        ratio = float(inl.mean())
        A = M[:, :2]
        t = 2.0 * M[:, 2] + 0.5 * (np.eye(2) - A) @ np.ones(2)     # half-res -> full-res coordinates
        M2 = np.hstack([A, t[:, None]]).astype(np.float32)
        if inl.sum() < 20 or ratio < 0.35 or np.abs(M2[:, 2]).max() > p.max_shift:
            return None, ratio
        return M2, ratio

    def _warp_state(self, y, M, x):
        p = self.p
        if not p.local_flow:
            return cv2.warpAffine(y, M, (W, H), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        # residual local motion (parallax: trees vs ground) with dense flow at 1/4 res; one combined cubic remap
        q = (W // 4, H // 4)
        Mq = M.copy()
        Mq[:, 2] /= 4.0
        yq = cv2.warpAffine(cv2.resize(y, q, interpolation=cv2.INTER_AREA), Mq, q, borderMode=cv2.BORDER_REPLICATE)
        xq = cv2.resize(x, q, interpolation=cv2.INTER_AREA)
        lo, hi = np.percentile(xq, (1, 99.5))
        sc = 255.0 / max(float(hi - lo), 1.0)
        x8 = np.clip((xq - lo) * sc, 0, 255).astype(np.uint8)
        y8 = np.clip((yq - lo) * sc, 0, 255).astype(np.uint8)
        fl = cv2.calcOpticalFlowFarneback(x8, y8, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        fl = np.clip(cv2.resize(fl, (W, H), interpolation=cv2.INTER_LINEAR) * 4.0, -p.flow_limit, p.flow_limit)
        Ai = cv2.invertAffineTransform(M)
        sx = Ai[0, 0] * self.gx + Ai[0, 1] * self.gy + Ai[0, 2] + Ai[0, 0] * fl[..., 0] + Ai[0, 1] * fl[..., 1]
        sy = Ai[1, 0] * self.gx + Ai[1, 1] * self.gy + Ai[1, 2] + Ai[1, 0] * fl[..., 0] + Ai[1, 1] * fl[..., 1]
        return cv2.remap(y, sx.astype(np.float32), sy.astype(np.float32), cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    def _temporal(self, x):
        p = self.p
        sn = self._est_noise(x)
        self.sigma_n = sn if self.y is None else self.sigma_n + 0.1 * (sn - self.sigma_n)
        f = self._features(x)
        prev, self.feat = self.feat, f
        if self.y is None or prev is None:
            self.y = x.copy()
            return self.y
        M, ratio, src = None, 0.0, "hold"
        if p.motion == "affine":
            M, ratio = self._motion_affine(prev, f)
            src = "lk"
        if M is None:                                           # translation fallback / v2 mode
            Mt, resp = self._motion_translate(prev, f)
            if p.motion == "translate" or resp >= p.min_resp:
                M, ratio, src = Mt, resp, "pc"
        if M is None:
            M, src = self.M_last, "hold"                        # camera motion is smooth: keep the last estimate
        else:
            self.M_last = M
        self.shift, self.src = (float(M[0, 2]), float(M[1, 2]), ratio), src
        self.y = self._warp_state(self.y, M, x)
        sig_d = float(self.sigma_n * (1.0 + p.kmin / (2 - p.kmin)) ** 0.5)
        d = x - self.y
        a = cv2.blur(np.abs(d), (3, 3))                       # pooled: motion is coherent, noise is not
        t = np.clip((a / sig_d - p.t0) / (p.t1 - p.t0), 0.0, 1.0)
        k = p.kmin + (1.0 - p.kmin) * t * t * (3.0 - 2.0 * t)
        self.diag = (float(np.median(a[::4, ::4])) / sig_d, float((k > 0.5).mean()), float(k.mean()))
        self.y += k * d
        return self.y

    def _build_lut(self, hist):
        """Absolute-anchored tone curve. lo = slow low percentile; hi = peak-hold of the per-frame high percentile,
        so large warm surfaces (road, roofs) stay mid-grey and only real hot anomalies reach white."""
        p = self.p
        edges = (np.arange(NB) + 1) << BSH
        c = np.cumsum(self.Hs)
        lo_t = float(np.interp(p.p_lo * c[-1], c, edges)) - 8
        # black point is slew-limited: a cold region (sky) entering the frame must not re-expose the whole image
        self.lo_s = lo_t if self.lo_s is None else self.lo_s + float(np.clip(lo_t - self.lo_s, -p.lo_slew, p.lo_slew))
        lo = self.lo_s
        cf = np.cumsum(hist)
        hf = float(np.interp(p.hi_pct * cf[-1], cf, edges))
        if self.hi_hold is None:
            self.hi_hold = hf
        elif hf > self.hi_hold:
            self.hi_hold += p.hi_attack * (hf - self.hi_hold)
        else:
            self.hi_hold -= p.hi_decay                       # very slow release
        min_span = max(p.min_span_abs, p.min_span_sigma * self.sigma_n)
        hi = max(self.hi_hold, lo + min_span)
        u = np.clip((np.arange(MAXV + 1) - lo) / (hi - lo), 0.0, 1.0)
        v = u ** p.gamma
        if p.shadow_lift:
            v = np.clip(v + p.shadow_lift * 14.9 * v * (1.0 - v) ** 5, 0.0, 1.0)   # lifts the shadows, ~no change above mid-grey
        self.lut = (255.0 * v).astype(np.float32)
        self.lo, self.hi = lo, hi

    def _tonemap(self, y):
        p = self.p
        yi = np.clip(y, 0, MAXV)
        hist = np.bincount((yi.astype(np.int32) >> BSH).ravel(), minlength=NB).astype(np.float32)
        mu_now = float(yi[::8, ::8].mean())
        if self.prev_mean is not None and abs(mu_now - self.prev_mean) > 0.6 * (self.hi - self.lo):
            self.cut = True
        self.prev_mean = mu_now
        if self.Hs is None or self.cut:
            self.Hs = hist
            self.hi_hold = None
            self.lo_s = None
            self._build_lut(hist)
            self.cut = False
        # --- fast guided filter base / detail split (a,b at half res, applied at full res)
        span = self.hi - self.lo
        mu = float(yi[::8, ::8].mean())
        I = yi - mu
        ksz = (2 * (p.gf_radius // 2) + 1,) * 2
        box = lambda a: cv2.boxFilter(a, -1, ksz, borderType=cv2.BORDER_REFLECT)
        Is = cv2.resize(I, (W // 2, H // 2), interpolation=cv2.INTER_AREA)
        m = box(Is)
        var = np.maximum(box(Is * Is) - m * m, 0.0)
        eps = float(max(p.edge_frac * span, 3.0 * self.sigma_n) ** 2)
        a = var / (var + eps)
        b = m - a * m
        A = cv2.resize(box(a), (W, H), interpolation=cv2.INTER_LINEAR)
        B0 = cv2.resize(box(b), (W, H), interpolation=cv2.INTER_LINEAR)
        base = A * I + B0 + mu
        detail = yi - base
        B = np.take(self.lut, np.clip(base, 0, MAXV).astype(np.int32))
        # --- cored, soft-limited detail; warm (positive) small detail is boosted more than cold
        sf = float(self.sigma_n * (p.kmin / (2 - p.kmin)) ** 0.5)
        tc2 = float((p.core_k * sf) ** 2)
        d2 = detail * detail
        d8 = p.detail_gain * (255.0 / span) * detail * d2 / (d2 + tc2)
        d8 += (p.hot_boost - 1.0) * np.maximum(d8, 0.0)
        d8 = d8 / (1.0 + np.abs(d8) / p.detail_limit)
        out = np.clip(B + d8, 0, 255).astype(np.uint8)
        # --- statistics of this frame set the curve of the next one (1-frame latency, no pixel buffer)
        self.Hs += p.hist_alpha * (hist - self.Hs)
        self._build_lut(hist)
        return out

    # ------------------------------------------------------------------ API
    def process(self, raw):
        """raw: (H,W) uint16 post-NUC frame -> (H,W) uint8 white-hot."""
        x = raw.astype(np.float32)
        if self._shutter_guard(x):
            return self.last_out.copy() if self.last_out is not None else np.zeros((H, W), np.uint8)
        x = self._fix_bad(x)
        if self.p.col_fpn:
            x = self._col_fpn(x)
        if self.p.row_fpn:
            x = self._row_fpn(x)
        y = self._temporal(x)
        out = self._tonemap(y)
        if not self.p.white_hot:
            out = 255 - out
        self.last_out = out
        return out
