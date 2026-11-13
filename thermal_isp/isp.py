"""
Streaming-friendly ISP for an uncooled 640x512 microbolometer (post-NUC 14-bit -> stable 8-bit white-hot).   [v1]

Stage order (each stage needs only a few lines of context or last-frame statistics, so it maps to an FPGA pixel stream):

  0  shutter / FFC frame guard        -> freeze the output instead of showing a flat gray frame
  1  bad-pixel replacement            -> static map + median of good neighbours (before any spatial filter)
  2  residual column (row) FPN removal-> vertical stripes the NUC did not remove
  3  motion-adaptive recursive TNR with global-TRANSLATION compensation (phase correlation) -> low noise, no ghosting
  4  histogram statistics (frame t-1) -> temporally smoothed, percentile range + plateau-limited equalisation LUT
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
    min_resp: float = 0.15       # phase-correlation confidence below which TNR is bypassed
    max_shift: float = 24.0      # px/frame
    t0: float = 1.0              # motion gate: mean|d|/sigma_d below t0 -> pure noise
    t1: float = 2.2              #              above t1 -> motion, pass through (k=1)
    cut_ratio: float = 5.0       # global change => scene cut / FFC recovery, reset state
    # --- AGC
    p_lo: float = 0.005
    p_hi: float = 0.9995
    hist_alpha: float = 0.05     # temporal smoothing of the histogram (anti-flicker)
    plateau: float = 1.0         # plateau = plateau * mean occupancy inside the range
    lin_mix: float = 0.5         # 0 = pure equalisation, 1 = linear
    min_span_sigma: float = 50.0 # never stretch less than this many sigma_n over 0..255
    min_span_abs: float = 150.0
    # --- detail layer
    gf_radius: int = 8
    edge_frac: float = 0.05      # guided-filter eps = (edge_frac * span)^2
    detail_gain: float = 1.5
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
        self.y = None            # TNR frame buffer
        self.sig_d = None
        self.feat = None
        self.sm = (0.0, 0.0)
        self.shift = (0, 0, 0)
        self.prev_mean = None
        self.win = cv2.createHanningWindow((W // 2, H // 2), cv2.CV_32F)
        self.sigma_n = 8.0
        self.cs = None           # per-column offset estimate
        self.Hs = None           # smoothed histogram
        self.lut = None
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

    def _est_shift(self, x):
        """Global translation prev->cur in full-res pixels (phase correlation at half res; FPGA: row/col projections)."""
        f = cv2.resize(x, (W // 2, H // 2), interpolation=cv2.INTER_AREA)
        f = cv2.GaussianBlur(f, (0, 0), 1.0)
        f -= float(f.mean())
        if self.feat is None:
            self.feat, res = f, (0.0, 0.0, 0.0)
        else:
            (dx, dy), r = cv2.phaseCorrelate(self.feat, f, self.win)
            self.feat, res = f, (2.0 * dx, 2.0 * dy, r)
        return res

    def _temporal(self, x):
        p = self.p
        sn = self._est_noise(x)
        self.sigma_n = sn if self.y is None else self.sigma_n + 0.1 * (sn - self.sigma_n)
        dx, dy, resp = self._est_shift(x)
        if self.y is None:
            self.y = x.copy()
            return self.y
        # camera motion is smooth: low-confidence / absurd measurements are blended toward (or replaced by)
        # the previous estimate instead of dropping the temporal filter (which would alternate clean/noisy frames)
        if abs(dx) > p.max_shift or abs(dy) > p.max_shift:
            resp = 0.0
        a_s = float(np.clip(resp / p.min_resp, 0.0, 1.0)) if resp < p.min_resp else 1.0
        if resp >= p.min_resp * 0.5 and a_s > 0:
            self.sm = (self.sm[0] + a_s * (dx - self.sm[0]), self.sm[1] + a_s * (dy - self.sm[1]))
        dx, dy = self.sm
        self.shift = (dx, dy, resp)
        if abs(dx) > 0.02 or abs(dy) > 0.02:
            M = np.float32([[1, 0, dx], [0, 1, dy]])
            self.y = cv2.warpAffine(self.y, M, (W, H), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        sig_d = float(self.sigma_n * (1.0 + p.kmin / (2 - p.kmin)) ** 0.5)
        d = x - self.y
        a = cv2.blur(np.abs(d), (3, 3))                       # pooled: motion is coherent, noise is not
        t = np.clip((a / sig_d - p.t0) / (p.t1 - p.t0), 0.0, 1.0)
        k = p.kmin + (1.0 - p.kmin) * t * t * (3.0 - 2.0 * t)
        self.y += k * d
        return self.y

    def _build_lut(self):
        p = self.p
        Hs = self.Hs
        N = float(Hs.sum()) + 1e-6
        c = np.cumsum(Hs)
        edges = (np.arange(NB) + 1) << BSH
        lo = float(np.interp(p.p_lo * N, c, edges)) - 8
        hi = float(np.interp(p.p_hi * N, c, edges))
        min_span = max(p.min_span_abs, p.min_span_sigma * self.sigma_n)
        if hi - lo < min_span:
            mid = 0.5 * (hi + lo)
            lo, hi = mid - 0.5 * min_span, mid + 0.5 * min_span
        xs = (np.arange(NB) + 0.5) * (1 << BSH)
        inside = (xs >= lo) & (xs <= hi)
        n_in = max(1.0, (hi - lo) / (1 << BSH))
        plateau = p.plateau * float(Hs[inside].sum() + 1e-6) / n_in
        clipped = np.where(inside, np.minimum(Hs, plateau), 0.0)
        cdf = np.cumsum(clipped)
        cdf = cdf / (cdf[-1] + 1e-9)
        lin = np.clip((xs - lo) / (hi - lo), 0, 1)
        lut_b = p.lin_mix * lin + (1 - p.lin_mix) * cdf
        lut_b = np.where(xs < lo, 0.0, np.where(xs > hi, 1.0, lut_b))
        self.lut = (np.interp(np.arange(MAXV + 1), xs, lut_b) * 255.0).astype(np.float32)
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
            self._build_lut()
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
        # --- cored, soft-limited detail
        sf = float(self.sigma_n * (p.kmin / (2 - p.kmin)) ** 0.5)
        tc2 = float((p.core_k * sf) ** 2)
        d2 = detail * detail
        d8 = p.detail_gain * (255.0 / span) * detail * d2 / (d2 + tc2)
        d8 = d8 / (1.0 + np.abs(d8) / p.detail_limit)
        out = np.clip(B + d8, 0, 255).astype(np.uint8)
        # --- statistics of this frame set the curve of the next one (1-frame latency, no pixel buffer)
        self.Hs += p.hist_alpha * (hist - self.Hs)
        self._build_lut()
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
