"""The full pipeline: raw 14-bit frame in, 8-bit white-hot frame out.

    1  bad pixels          bad_pixels.py
    2  stripes             stripes.py
    3a offset stabiliser   offset_stabiliser.py
    3b temporal denoise    temporal.py
    4  base / detail       detail.py
    5  AGC + 6 detail      agc.py
"""

import numpy as np

from .config import Config, WIDTH, HEIGHT, MAX_VALUE
from .noise import NoiseTracker, estimate_noise_sigma
from .bad_pixels import BadPixelCorrector
from .stripes import StripeRemover
from .offset_stabiliser import OffsetStabiliser
from .temporal import TemporalDenoiser
from .detail import split_base_detail
from .agc import AGC


class ThermalISP:
    def __init__(self, cfg=None, width=WIDTH, height=HEIGHT):
        self.cfg = cfg or Config()
        self.noise = NoiseTracker()
        self.fix_bad_pixels = BadPixelCorrector(self.cfg, (height, width))
        self.remove_stripes = StripeRemover(self.cfg, width)
        self.stabilise_offset = OffsetStabiliser(self.cfg)
        self.denoise = TemporalDenoiser(self.cfg)
        self.agc = AGC(self.cfg)

    def process(self, raw):
        cfg = self.cfg
        frame = np.minimum(raw, MAX_VALUE).astype(np.uint16)

        if cfg.badpix:
            sigma_so_far = self.noise.sigma or estimate_noise_sigma(frame.astype(np.float32))
            frame = self.fix_bad_pixels(frame, sigma_so_far)

        frame = frame.astype(np.float32)
        sigma = self.noise.update(frame)

        if cfg.stripes:
            frame = self.remove_stripes(frame, sigma)
        if cfg.temporal:
            frame = self.stabilise_offset(frame, sigma)
            frame = self.denoise(frame, sigma)

        frame = frame.astype(np.float32, copy=False)   # guard: everything below is float32
        if cfg.dde:
            base, detail = split_base_detail(frame, cfg, sigma)
        else:
            base, detail = frame, None

        # after temporal averaging the remaining noise is about half
        detail_threshold = sigma * (0.5 if cfg.temporal else 1.0)
        return self.agc(base, detail, sigma, detail_threshold)


def naive_scale(raw):
    """What 'just scale to 8 bit' looks like: per-frame min/max."""
    frame = raw.astype(np.float32)
    low, high = frame.min(), frame.max()
    return ((frame - low) * 255.0 / max(high - low, 1)).astype(np.uint8)
