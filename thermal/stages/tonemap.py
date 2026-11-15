"""Step 6 – tone mapping of the linear frame to the display range (AGC)."""

from __future__ import annotations

import numpy as np

import cv2

from ..frame import Domain, FrameData
from ..stage import Stage


def soft_top(y: np.ndarray, k0: float) -> np.ndarray:
    """Identity below ``k0``; above it, compress smoothly towards 1.

    Continuous with slope 1 at ``k0``, never reaches 1, monotonic.
    """
    span = 1.0 - k0
    over = np.maximum(y - k0, 0.0)
    return np.where(y > k0, k0 + span * (1.0 - np.exp(-over / span)), y)


def soft_bottom(y: np.ndarray, f0: float) -> np.ndarray:
    """Mirror of ``soft_top``: compress smoothly towards 0 below ``f0``."""
    return 1.0 - soft_top(1.0 - y, 1.0 - f0)


class ToneMap(Stage):
    """Hot-object-aware AGC: linear DN → display values 0..1 (white-hot).

    1. **Statistics** come from ``layers["base"]`` (step 5), or from
       ``image`` if there is no base. A histogram on a fixed DN grid
       (``bin_dn`` wide) is smoothed over time (EMA ``alpha_hist``), so the
       curve does not flicker.
    2. **Scene range** ``[lo, hi]`` = percentiles ``p_low`` / ``p_high`` of the
       base, smoothed over time (EMA; ``alpha_grow`` when the range gets
       wider, ``alpha_shrink`` when it gets narrower).
    3. **Main curve** on ``[lo, hi]`` → ``[foot, knee]``:
       ``(1 - linear_mix) * plateau-HE + linear_mix * linear``.
       Plateau histogram equalisation gives grey levels in proportion to
       how many pixels have a temperature (each bin clipped at
       ``plateau`` x the mean bin count). A hot car or the cold sky adds only
       a few pixels, so it takes few grey levels and the rest of the scene
       keeps its contrast; the empty gap between scene and object takes
       almost none. The linear part keeps a proportional link to temperature
       and gives some levels to sparse ranges.
    4. **Soft knee / foot**: values above ``hi`` go to ``[knee, 1)``, below
       ``lo`` to ``(0, foot]``, with an exponential curve scaled by the
       smoothed frame max / min. Nothing is clipped; hotter is always
       brighter (monotonic) – the inner structure of hot objects stays.
    5. **Detail** (step 5) is added back with the local slope of the curve:
       ``y = T(base) + detail_gain * T'(base) * detail``, slope limited to
       ``max_slope`` x the slope of a linear map (noise is not boosted in
       dense histogram ranges). ``detail_gain`` > 1 is detail enhancement
       (step 7 will refine it).
    Black and white level (subjective tuning):
    ``gamma`` > 1 makes dark and middle tones darker (deeper black ground),
    < 1 lighter; it is applied to the main curve, so the order of
    temperatures does not change. ``white`` is the display value of the
    hottest possible pixel (1 = full white); below 1 hot objects glare less.

    6. **Final soft limiter** on ``y`` (``soft_top`` at ``limit``,
       ``soft_bottom`` at ``1 - limit``), so detail added near the ends of
       the range cannot reach 0 or 1 (docs TODO: small hot objects / limit
       ``g · detail`` everywhere). It is placed above the knee, so the knee
       itself is not compressed twice.

    Adaptation speed: ``alpha_grow`` / ``alpha_shrink`` (scene range) and
    ``alpha_hist`` (histogram), per frame; 1 = no smoothing.
    **Peak hold** for the frame maximum / minimum that scale the knee and
    the foot: a new maximum is followed at once (``peak_attack`` = 1), kept
    for ``peak_hold`` frames, then released slowly (``alpha_shrink``). So a
    hot object that enters gets the full knee range from its first frame
    (no compressed inner detail while the smoothing catches up).

    Switches (to see what each algorithm contributes):
    ``use_plateau`` – False: no histogram equalisation (same as
    ``linear_mix = 1``); ``use_knee`` – False: hard clipping at ``lo`` /
    ``hi`` instead of the soft knee and foot (the range maps to 0..1);
    ``temporal`` – False: range and histogram from the current frame only
    (no smoothing over time, like a per-frame percentile stretch).

    Output: ``image`` in 0..1 (float32), ``domain = DISPLAY``;
    ``layers["tonemap.slope"]`` – display units per DN at each pixel (used by
    ``DetailEnhance`` to convert the noise map to display units).
    ``meta``: ``agc.lo``, ``agc.hi`` (DN, smoothed), ``agc.max``, ``agc.min``.

    Parameters
    ----------
    p_low, p_high : percentiles that define the scene range
    alpha_grow, alpha_shrink : EMA rates of the range (per frame)
    alpha_hist : EMA rate of the histogram
    plateau : plateau of the histogram, x mean bin count
    linear_mix : share of the linear curve in the main curve (0..1)
    foot, knee : display values reserved below ``lo`` / above ``hi``
    detail_gain : gain of the detail layer (1 = as in the scene)
    max_slope : limit of the curve slope for detail, x linear slope
    limit : start of the final soft limiter (and 1 - limit at the bottom)
    gamma : exponent of the main curve (black level of dark/middle tones)
    white : output value for the hottest pixels (white level)
    bin_dn : histogram bin width in DN
    use_plateau, use_knee, temporal : switches, see above
    peak_attack : how fast a new max / min is followed (1 = at once)
    peak_hold : frames a peak is kept before it is released

    **Options taken from Michael's ``thermal_isp``** (all off by default, so
    the old behaviour is unchanged; preset ``michael-features`` switches
    them on). They make the mapping DN → grey almost fixed over time, so the
    same temperature keeps the same grey level when a cold sky or a hot
    object enters the frame:

    anchor_black : the black point ``lo`` may move only ``lo_slew`` DN per
        frame (Michael: 1.5). Exception = a real scene change: if more than
        ``fast_frac`` of the frame is colder than ``lo`` (going down), or the
        new percentile is more than ``fast_gap`` x range above ``lo`` (going
        up), ``lo`` follows at the normal EMA speed. A strip of sky (≈ 4 % of
        the frame in this recording) is not a scene change; it goes into the
        soft foot instead of moving the whole scale.
    white_hold : the top of the range ``hi`` is a peak hold of the
        ``p_high`` percentile: up with ``hi_attack`` (fraction per frame),
        down only ``hi_decay`` DN per frame.
    min_span_dn : the range ``hi - lo`` is never narrower than this (DN).
        Michael uses 1000, which makes this recording look milky (the
        scene spans ≈ 600 DN); 400–600 is a better start.
    shadow_lift : raises dark tones (Michael: 0.12, max lift at 1/6 of the
        range, almost none above mid grey). 0 = off.

    **DDE mode** (``dde = True``, Digital Detail Enhancement): the detail
    layer is added in display units with its *own* gain instead of the slope
    of the tone curve, so texture is not compressed together with the base:

    ``add = g · w · detail``, ``w = d² / (d² + (dde_coring_k · noise)²)``
    (noise gate from the noise map of step 3), with
    ``g = (1 - dde_adapt) · dde_gain · lin_slope + dde_adapt · g_local``,
    ``g_local = dde_target / rms_local(detail)`` (limited to
    ``dde_max`` x lin_slope). ``dde_adapt`` = 0 → fixed gain (as Michael:
    detail x2.6 of the linear slope); 1 → every region gets about
    ``dde_target`` of detail contrast, weak texture (fields, crowns) is
    raised most. Warm (positive) detail is multiplied by ``warm_boost``
    (Michael: 1.5). Soft limit ``add / (1 + |add| / dde_limit)``.
    ``detail_gain`` / ``max_slope`` are not used in this mode.
    """

    name = "ToneMap"

    def __init__(self, p_low: float = 0.2, p_high: float = 99.8,
                 alpha_grow: float = 0.15, alpha_shrink: float = 0.05,
                 alpha_hist: float = 0.15, plateau: float = 3.0,
                 linear_mix: float = 0.3, foot: float = 0.05,
                 knee: float = 0.88, detail_gain: float = 1.0,
                 max_slope: float = 3.0, limit: float = 0.97,
                 gamma: float = 1.0, white: float = 1.0,
                 bin_dn: float = 2.0, max_dn: int = 16384,
                 use_plateau: bool = True, use_knee: bool = True,
                 temporal: bool = True, peak_attack: float = 1.0,
                 peak_hold: int = 12,
                 anchor_black: bool = False, lo_slew: float = 1.5,
                 fast_frac: float = 0.15, fast_gap: float = 0.25,
                 white_hold: bool = False, hi_attack: float = 0.15,
                 hi_decay: float = 0.15, min_span_dn: float = 0.0,
                 shadow_lift: float = 0.0,
                 dde: bool = False, dde_gain: float = 2.5,
                 dde_adapt: float = 0.5, dde_target: float = 0.03,
                 dde_max: float = 8.0, dde_radius: int = 7,
                 dde_coring_k: float = 2.0, dde_limit: float = 0.2,
                 warm_boost: float = 1.0) -> None:
        super().__init__()
        if not 0 <= foot < knee <= 1:
            raise ValueError("need 0 <= foot < knee <= 1")
        self.p_low, self.p_high = p_low, p_high
        self.alpha_grow, self.alpha_shrink = alpha_grow, alpha_shrink
        self.alpha_hist = alpha_hist
        self.plateau = plateau
        self.linear_mix = linear_mix
        self.foot, self.knee = foot, knee
        self.detail_gain = detail_gain
        self.max_slope = max_slope
        self.limit = limit
        self.gamma = gamma
        self.white = white
        self.bin_dn = bin_dn
        self.max_dn = max_dn
        self.use_plateau = use_plateau
        self.use_knee = use_knee
        self.temporal = temporal
        self.peak_attack = peak_attack
        self.peak_hold = peak_hold
        self.anchor_black, self.lo_slew = anchor_black, lo_slew
        self.fast_frac, self.fast_gap = fast_frac, fast_gap
        self.white_hold, self.hi_attack, self.hi_decay = white_hold, hi_attack, hi_decay
        self.min_span_dn = min_span_dn
        self.shadow_lift = shadow_lift
        self.dde, self.dde_gain, self.dde_adapt = dde, dde_gain, dde_adapt
        self.dde_target, self.dde_max, self.dde_radius = dde_target, dde_max, dde_radius
        self.dde_coring_k, self.dde_limit = dde_coring_k, dde_limit
        self.warm_boost = warm_boost
        self.reset()

    def reset(self) -> None:
        self._hist: np.ndarray | None = None
        self._lo = self._hi = self._min = self._max = None
        self._hold_max = self._hold_min = 0
        self._hi_hold = None
        self._lo_fast = False

    # ----- helpers ---------------------------------------------------------
    def _ema(self, old, new, grow_up: bool):
        if old is None or not self.temporal:
            return new
        grows = (new > old) if grow_up else (new < old)
        a = self.alpha_grow if grows else self.alpha_shrink
        return old + a * (new - old)

    def _peak(self, old, new, hold: int, up: bool):
        """Peak hold: follow a new peak fast, keep it, then release slowly."""
        if old is None or not self.temporal:
            return new, int(self.peak_hold)
        if (new > old) if up else (new < old):
            return old + self.peak_attack * (new - old), int(self.peak_hold)
        if hold > 0:
            return old, hold - 1
        return old + self.alpha_shrink * (new - old), 0

    def _update_stats(self, base: np.ndarray) -> None:
        sub = base[::2, ::2]
        nbins = int(np.ceil(self.max_dn / self.bin_dn))
        b = np.clip((sub / self.bin_dn).astype(np.int32), 0, nbins - 1)
        h = np.bincount(b.ravel(), minlength=nbins).astype(np.float64)
        h /= h.sum()
        self._hist = h if (self._hist is None or not self.temporal) else \
            self._hist + self.alpha_hist * (h - self._hist)
        lo, hi = np.percentile(sub, [self.p_low, self.p_high])
        if self.anchor_black and self.temporal and self._lo is not None:
            self._lo = self._anchored_lo(sub, float(lo))
        else:
            self._lo = self._ema(self._lo, float(lo), grow_up=False)
        if self.white_hold and self.temporal and self._hi_hold is not None:
            if hi > self._hi_hold:
                self._hi_hold += self.hi_attack * (float(hi) - self._hi_hold)
            else:
                self._hi_hold = max(float(hi), self._hi_hold - self.hi_decay)
            self._hi = self._hi_hold
        elif self.white_hold:
            self._hi = self._hi_hold = float(hi)
        else:
            self._hi = self._ema(self._hi, float(hi), grow_up=True)
        if self.min_span_dn > 0 and self._hi - self._lo < self.min_span_dn:
            self._hi = self._lo + float(self.min_span_dn)
        self._max, self._hold_max = self._peak(self._max, float(sub.max()),
                                               self._hold_max, up=True)
        self._min, self._hold_min = self._peak(self._min, float(sub.min()),
                                               self._hold_min, up=False)
        if self._hi - self._lo < 4 * self.bin_dn:  # flat scene
            c = 0.5 * (self._hi + self._lo)
            self._lo, self._hi = c - 2 * self.bin_dn, c + 2 * self.bin_dn

    def _anchored_lo(self, sub: np.ndarray, lo_t: float) -> float:
        """Black point that moves slowly, except on a real scene change."""
        old = self._lo
        span = max((self._hi or old + 1.0) - old, 1.0)
        if lo_t < old:   # something colder: how much of the frame is below lo?
            fast = float(np.mean(sub < old)) > self.fast_frac
        else:            # scene warmer: empty gap at the bottom of the range
            fast = (lo_t - old) / span > self.fast_gap
        self._lo_fast = fast
        if fast:
            a = self.alpha_grow if lo_t < old else self.alpha_shrink
            return old + a * (lo_t - old)
        return old + float(np.clip(lo_t - old, -self.lo_slew, self.lo_slew))

    def _dde_add(self, detail: np.ndarray, lin_slope: float,
                 frame: FrameData) -> tuple[np.ndarray, np.ndarray]:
        """Detail added in display units (DDE mode); returns (add, gain map)."""
        d = detail.astype(np.float32, copy=False)
        noise = frame.layers.get("noise_sigma_map")
        if noise is None:
            noise = np.float32(frame.meta.get("noise_sigma", 1.0))
        d2 = d * d
        w = d2 / (d2 + (self.dde_coring_k * noise) ** 2 + 1e-6)
        g = np.float32(self.dde_gain * lin_slope)
        if self.dde_adapt > 0:
            k = (2 * int(self.dde_radius) + 1,) * 2
            rms = np.sqrt(cv2.blur(d2, k, borderType=cv2.BORDER_REFLECT)
                          + (2.0 * noise) ** 2)
            g_loc = np.minimum(self.dde_target / rms, self.dde_max * lin_slope)
            g = (1.0 - self.dde_adapt) * g + self.dde_adapt * g_loc
        add = g * w * d
        if self.warm_boost != 1.0:
            add = np.where(add > 0, add * self.warm_boost, add)
        add = add / (1.0 + np.abs(add) / self.dde_limit)
        g = np.broadcast_to(np.asarray(g, np.float32), d.shape)
        return add.astype(np.float32), g

    def _main_curve(self) -> tuple[np.ndarray, np.ndarray]:
        """LUT (DN grid, values in 0..1) of the main curve on [lo, hi]."""
        i0 = int(np.floor(self._lo / self.bin_dn))
        i1 = int(np.ceil(self._hi / self.bin_dn))
        h = self._hist[i0:i1 + 1].copy()
        if h.sum() <= 0:
            h[:] = 1.0
        h = np.minimum(h, self.plateau * h.mean())
        cdf = np.concatenate([[0.0], np.cumsum(h)])
        cdf /= cdf[-1]
        edges = (np.arange(i0, i1 + 2) * self.bin_dn).astype(np.float64)
        lin = (edges - edges[0]) / (edges[-1] - edges[0])
        mix = self.linear_mix if self.use_plateau else 1.0
        curve = (1 - mix) * cdf + mix * lin
        if self.gamma != 1.0:
            curve = curve ** self.gamma
        if self.shadow_lift:
            # Michael's shadow lift: max +shadow_lift at 1/6 of the range,
            # ~0 above mid grey (14.9 normalises max of v(1-v)^5 to 1)
            curve = np.clip(curve + self.shadow_lift * 14.9 * curve
                            * (1.0 - curve) ** 5, 0.0, 1.0)
        return edges, curve

    def _map(self, v: np.ndarray, edges: np.ndarray, curve: np.ndarray):
        """Map DN to display values; also return the slope (per DN)."""
        f, k = (self.foot, self.knee) if self.use_knee else (0.0, 1.0)
        lo, hi = edges[0], edges[-1]
        y = np.interp(v, edges, curve) * (k - f) + f
        slope_lut = np.gradient(curve, edges) * (k - f)
        slope = np.interp(v, edges, slope_lut)

        # soft knee above hi: y = k + (1-k)(1 - exp(-(v-hi)/s_hi))
        s_hi = max((self._max - hi) / 2.0, 4 * self.bin_dn)
        above = v > hi
        if not self.use_knee:  # hard clipping (for comparison)
            below = v < lo
            y[above], slope[above] = 1.0, 0.0
            y[below], slope[below] = 0.0, 0.0
            return y, slope
        if above.any():
            t = (v[above] - hi) / s_hi
            y[above] = k + (1 - k) * (1 - np.exp(-t))
            slope[above] = (1 - k) / s_hi * np.exp(-t)
        s_lo = max((lo - self._min) / 2.0, 4 * self.bin_dn)
        below = v < lo
        if below.any():
            t = (lo - v[below]) / s_lo
            y[below] = f * np.exp(-t)
            slope[below] = f / s_lo * np.exp(-t)
        return y, slope

    def warmup(self, frame: FrameData) -> FrameData:
        """Update only the temporal state (range, histogram); no mapping.

        Used by the interactive tuner to prepare the state for a frame
        quickly. The frame is returned unchanged.
        """
        x = frame.image
        self._update_stats(frame.layers.get("base", x))
        return frame

    # ----- main ------------------------------------------------------------
    def process(self, frame: FrameData) -> FrameData:
        frame.require(Domain.LINEAR)
        x = frame.image.astype(np.float32, copy=False)
        base = frame.layers.get("base", x)
        detail = frame.layers.get("detail")
        self._update_stats(base)
        edges, curve = self._main_curve()

        y, slope = self._map(base.astype(np.float64), edges, curve)
        f, k = (self.foot, self.knee) if self.use_knee else (0.0, 1.0)
        if detail is not None and self.dde:
            lin_slope = (k - f) / (self._hi - self._lo)
            add, g = self._dde_add(detail, lin_slope, frame)
            y = y + add
            slope = g  # display units per DN of detail (for the noise gate of step 7)
        elif detail is not None and self.detail_gain > 0:
            lin_slope = (k - f) / (self._hi - self._lo)
            slope = np.minimum(slope, self.max_slope * lin_slope)
            y = y + self.detail_gain * slope * detail
        else:
            lin_slope = (k - f) / (self._hi - self._lo)
            slope = np.minimum(slope, self.max_slope * lin_slope)
        if self.use_knee:
            y = soft_bottom(soft_top(y, self.limit), 1.0 - self.limit)
        else:
            y = np.clip(y, 0.0, 1.0)
        if self.white != 1.0:
            y = y * self.white
            slope = slope * self.white

        frame.layers["linear"] = x  # linear frame before tone mapping (CompactHotBoost)
        frame.image = y.astype(np.float32)
        # Display units per DN of fine detail / noise (detail is re-added
        # with detail_gain, so noise in the output is scaled by it as well).
        gain = 1.0 if self.dde else max(self.detail_gain, 1.0)
        frame.layers["tonemap.slope"] = (slope * gain).astype(np.float32)
        frame.domain = Domain.DISPLAY
        frame.meta["tonemap.knee"] = (self.knee if self.use_knee else 1.0) * self.white
        frame.meta["tonemap.lin_slope"] = lin_slope * self.white
        frame.meta["agc.lo_fast"] = bool(self._lo_fast)
        frame.meta["agc.lo"] = self._lo
        frame.meta["agc.hi"] = self._hi
        frame.meta["agc.min"] = self._min
        frame.meta["agc.max"] = self._max
        return frame
