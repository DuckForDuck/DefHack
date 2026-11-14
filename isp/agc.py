"""Stage 5 - automatic gain control (14-bit -> 8-bit).

agc_mode "absolute" (default, v2): fixed counts-to-grey scale between a slow black point and a
peak-tracking white point, with an optional share of v1 equalisation. "plateau" is v1.

Why each piece is there:
  * percentile limits   : a few dead/hot pixels don't decide the range
  * plateau cap         : a big uniform sky can't take all the grey levels
  * empty bins get ~0   : a hot object can't flatten the rest of the scene
  * small linear share  : "how much hotter" is still visible
  * minimum range       : a blank wall stays calm grey, not boosted noise
  * LUT from the PREVIOUS frames, smoothed in time : no flicker, FPGA-friendly
"""

import numpy as np
import cv2

from .config import MAX_VALUE
from .detail import soft_coring


# ---------------------------------------------------------------- building a LUT
def histogram_of(image):
    """14-bit histogram on a 1/4 subsample (enough for statistics)."""
    values = np.clip(image[::2, ::2], 0, MAX_VALUE).astype(np.int32).ravel()
    return np.bincount(values, minlength=MAX_VALUE + 1).astype(np.float64)


def percentile_range(hist, low_share, high_share):
    cumulative = np.cumsum(hist)
    total = cumulative[-1]
    low = int(np.searchsorted(cumulative, low_share * total))
    high = int(np.searchsorted(cumulative, high_share * total))
    return low, high


def widen_to_minimum(low, high, min_width):
    """Flat scene: widen the range so noise is not stretched to full contrast."""
    if high - low < min_width:
        middle = 0.5 * (high + low)
        low = int(max(0, middle - min_width / 2))
        high = int(min(MAX_VALUE, middle + min_width / 2))
    return low, max(high, low + 2)


def level_shares(hist_part, plateau_factor, linear_share):
    """How many grey levels each count value gets (sums to 1)."""
    smooth = np.convolve(hist_part, np.ones(5) / 5, mode="same")
    used_bins = max(int((smooth > 0).sum()), 1)
    plateau = plateau_factor * smooth.sum() / used_bins
    capped = np.minimum(smooth, plateau)
    equalised = capped / max(capped.sum(), 1e-9)
    linear = np.full_like(equalised, 1.0 / len(equalised))
    return (1 - linear_share) * equalised + linear_share * linear


def mean_std_range(hist, std_factor):
    """Mid value (mean) +- std_factor * std, computed from the histogram."""
    values = np.arange(len(hist), dtype=np.float64)
    total = max(hist.sum(), 1.0)
    mean = (hist * values).sum() / total
    std = np.sqrt(max((hist * (values - mean) ** 2).sum() / total, 0.0))
    low = int(max(0, mean - std_factor * std))
    high = int(min(MAX_VALUE, mean + std_factor * std))
    return low, high


def ramp_into_lut(ramp, low, high, cfg):
    """Place a 0..1 ramp between `low` and `high`, flat outside, scaled to the output range."""
    lut = np.empty(MAX_VALUE + 1, np.float32)
    lut[:low] = 0.0
    lut[low:high + 1] = ramp
    lut[high + 1:] = 1.0
    return cfg.out_low + lut * (cfg.out_high - cfg.out_low)


def build_lut(image, cfg, sigma):
    """Look-up table: 14-bit count -> output grey level (float, 16384 entries).

    agc_mode ("absolute" is handled by AbsoluteScale, not here):
      plateau    - plateau histogram equalisation + small linear share (default, best
                   with hot objects: empty count ranges get almost no grey levels)
      percentile - straight line between the clip_low / clip_high percentiles
      mean       - straight line over mean +- mean_std_factor * std (mid-value AGC)
    """
    hist = histogram_of(image)
    if cfg.agc_mode == "mean":
        low, high = mean_std_range(hist, cfg.mean_std_factor)
    else:
        low, high = percentile_range(hist, cfg.clip_low, cfg.clip_high)
    low, high = widen_to_minimum(low, high, cfg.min_range_sigmas * sigma)

    if cfg.agc_mode == "plateau":
        shares = level_shares(hist[low:high + 1], cfg.plateau_factor, cfg.linear_share)
        ramp = np.concatenate([[0.0], np.cumsum(shares)[:-1]])
    else:
        ramp = np.linspace(0.0, 1.0, high - low + 1)
    return ramp_into_lut(ramp, low, high, cfg)


# ---------------------------------------------------------------- absolute (v2) mode
WHITE_REACH = 0.5   # pixels more than half a scene range above white are ignored for white
                    # (an exhaust plume's warm fringe stays out; animals are pulled in step by step)
def percentile_of(hist, share):
    cumulative = np.cumsum(hist)
    return int(np.searchsorted(cumulative, share * cumulative[-1]))


def equalised_ramp(hist_part, plateau_factor):
    """0..1 cumulative of a plateau-capped histogram (the v1 curve, no linear share)."""
    shares = level_shares(hist_part, plateau_factor, 0.0)
    return np.concatenate([[0.0], np.cumsum(shares)[:-1]]) if len(shares) > 1 else np.zeros(1)


class AbsoluteScale:
    """Temperature-faithful tone curve (v2), with the useful parts of v1 kept.

      black point : a low percentile, tracked SLOWLY  -> the scene does not pump
      white point : a high percentile, rises fast / falls slowly (peak tracking), but never
                    closer than white_floor_sigmas to black, so the scale stays put when
                    nothing hot is in view
      hot peak    : everything above the white point shares the top (1 - knee) of the grey
                    scale; the scene part never changes, so one engine can't darken the scene
      eq_share    : how much of v1's plateau equalisation is blended in (0 = pure linear).
                    Uses a time-averaged histogram, so it doesn't make the scene pump.
      gamma       : < 1 lifts the cool end, > 1 spreads the warm end
    Equal raw counts -> equal grey everywhere in the scene part (when eq_share = 0), so a
    roof 150 counts above the road stays a fixed grey, whatever else is in the frame.
    """

    def __init__(self):
        self.black = self.white = self.peak = None
        self.hist = None

    def update(self, hist, cfg, sigma):
        black = percentile_of(hist, cfg.black_percentile)
        peak = percentile_of(hist, cfg.peak_percentile)
        # white point from the SCENE only: pixels more than WHITE_REACH scene ranges above the
        # current white point (engine, exhaust, sun glint) are left to the knee, so a
        # small extreme object can't pull the white point up and darken everything else.
        # Warm objects closer than that (animals, people) do raise it, a bit per frame.
        if self.white is not None:
            limit = int(min(self.white + WHITE_REACH * max(self.white - self.black, 1.0), MAX_VALUE))
            white = percentile_of(hist[:limit + 1], cfg.white_percentile)
        else:
            white = percentile_of(hist, cfg.white_percentile)
        if self.black is None:
            self.black, self.white, self.peak, self.hist = float(black), float(white), float(peak), hist.copy()
            return
        # scene change: most of the frame left the tracked range -> follow much faster
        inside = hist[int(self.black):int(self.peak) + 1].sum() / max(hist.sum(), 1.0)
        boost = 5.0 if 1.0 - inside > cfg.scene_change_share else 1.0
        self.black += min(cfg.black_rate * boost, 1.0) * (black - self.black)
        for name, new in (("white", white), ("peak", peak)):
            old = getattr(self, name)
            rate = cfg.white_attack if new > old else cfg.white_release * boost
            if name == "peak" and new < old:
                rate = max(rate, 0.05)          # only shapes the hot band: may relax faster
            setattr(self, name, old + min(rate, 1.0) * (new - old))
        self.hist += cfg.lut_smoothing * (hist - self.hist)

    def lut(self, cfg, sigma):
        # white floor (v1's "minimum range", anchored at black): with nothing hotter than
        # the roofs in view, the roofs are NOT stretched to white; the scale stays put
        white = max(self.white, self.black + cfg.white_floor_sigmas * sigma)
        low, high = widen_to_minimum(int(self.black), int(white), cfg.min_range_sigmas * sigma)
        peak = int(min(max(self.peak, high), MAX_VALUE))

        linear = np.linspace(0.0, 1.0, high - low + 1)
        ramp = linear
        if cfg.eq_share > 0:
            ramp = (1 - cfg.eq_share) * linear + cfg.eq_share * equalised_ramp(
                self.hist[low:high + 1], cfg.plateau_factor)
        ramp = ramp ** cfg.gamma

        # Fixed split: the scene (black..white) ALWAYS gets the bottom `knee` share of the
        # grey levels, everything hotter shares the top. So an engine entering or leaving
        # never re-scales the scene. When nothing is far above white, the top part simply
        # continues with the same slope (plain linear, no noise boost).
        knee = cfg.knee
        same_slope_top = high + (1.0 - knee) / knee * (high - low)
        peak = int(min(max(peak, same_slope_top), MAX_VALUE))
        hot = np.linspace(knee, 1.0, peak - high + 1)[1:]
        return ramp_into_lut(np.concatenate([knee * ramp, hot]), low, peak, cfg)


# ---------------------------------------------------------------- optional: large warm areas
def suppress_large_warm(base, strength, size):
    """Darken LARGE areas warmer than the scene median (sun-warmed roofs, walls).

    A grey-level opening (erode, then dilate) with a `size` x `size` square removes every
    warm object narrower than `size` (animals, people, exhaust) and keeps large areas with
    their own sharp outline. Only that large-area excess above the median is reduced, so
    small warm objects keep full brightness and roofs darken evenly, edge to edge, with
    no halo. ~3 ms at full resolution (OpenCV rectangle morphology). OFF by default: it removes real thermal
    information (a sun-warmed roof IS warmer than the road).
    """
    smooth = cv2.blur(base, (5, 5))                       # noise would bias the erosion low
    k = max(3, size | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    large = cv2.morphologyEx(smooth, cv2.MORPH_OPEN, kernel, borderType=cv2.BORDER_REPLICATE)
    reference = float(np.median(base[::4, ::4]))
    return base - strength * np.maximum(large - reference, 0.0)


# ---------------------------------------------------------------- using a LUT
def apply_lut(lut, image):
    """LUT with linear interpolation between entries (no banding in narrow scenes)."""
    values = np.clip(image, 0, MAX_VALUE)
    index = values.astype(np.int32)
    fraction = values - index
    next_index = np.minimum(index + 1, MAX_VALUE)
    return lut[index] * (1 - fraction) + lut[next_index] * fraction, index


def lut_slope(lut, cfg, sigma):
    """Grey levels per count at each input value, smoothed and limited.

    Upper limit = slope of the minimum-range stretch (noise never boosted more).
    Lower limit keeps texture visible inside compressed regions (hot objects).
    """
    smooth = cv2.blur(lut.reshape(1, -1), (33, 1)).ravel()
    slope = np.gradient(smooth)
    max_slope = 255.0 / (cfg.min_range_sigmas * sigma)
    return np.clip(slope, 0.2 * max_slope, max_slope)


def lut_range(lut, cfg):
    """Input values that the LUT maps inside the output range."""
    low = np.searchsorted(lut, cfg.out_low + 1)
    high = np.searchsorted(lut, cfg.out_high - 1)
    return low, high


# ---------------------------------------------------------------- the AGC
class AGC:
    def __init__(self, cfg):
        self.cfg = cfg
        self.lut = None          # built from previous frames
        self.absolute = AbsoluteScale()

    def __call__(self, base, detail, sigma, detail_threshold):
        if self.cfg.warm_suppress:
            base = suppress_large_warm(base, self.cfg.warm_suppress_strength, self.cfg.warm_suppress_size)
        absolute = self.cfg.agc_mode == "absolute"
        if absolute:
            if self.lut is None:                 # fresh start / mode switch: re-seed trackers
                self.absolute = AbsoluteScale()
            self.absolute.update(histogram_of(base), self.cfg, sigma)
            new_lut = self.absolute.lut(self.cfg, sigma)
        else:
            new_lut = build_lut(base, self.cfg, sigma)
        if self.lut is None:
            self.lut = new_lut

        image, index = apply_lut(self.lut, base)
        if detail is not None:
            slope = lut_slope(self.lut, self.cfg, sigma)
            cored = soft_coring(detail, detail_threshold)
            image = image + self.cfg.detail_gain * slope[index] * cored

        if absolute:
            self.lut = new_lut                   # already smoothed by the point trackers
        else:
            self._update_lut(new_lut, base)
        return np.clip(image + 0.5, 0, 255).astype(np.uint8)

    def _update_lut(self, new_lut, base):
        """Blend the new LUT in slowly; faster only on a real scene change."""
        low, high = lut_range(self.lut, self.cfg)
        sample = np.clip(base[::4, ::4], 0, MAX_VALUE)
        outside = np.mean((sample < low) | (sample > high))
        rate = self.cfg.lut_smoothing
        if outside > self.cfg.scene_change_share:
            rate = self.cfg.lut_smoothing_fast
        self.lut = self.lut + rate * (new_lut - self.lut)
