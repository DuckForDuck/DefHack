"""Stage 1 - bad pixel correction.

Goes first: one stuck 16383 pixel would distort every statistic after it.
  * dynamic: a pixel far outside the range of its 8 neighbours is a spike
  * static : a map of pixels that are always wrong, learned while video runs
Bad pixels are replaced by the 3x3 median.
"""

import numpy as np
import cv2

NEIGHBOURS_ONLY = np.array([[1, 1, 1],
                            [1, 0, 1],
                            [1, 1, 1]], np.uint8)


def find_spikes(frame, threshold):
    """True where a pixel is more than `threshold` above the brightest or below the
    darkest of its 8 neighbours. Real hot spots are spread over several pixels by
    the lens, so they pass this test."""
    brightest_neighbour = cv2.dilate(frame, NEIGHBOURS_ONLY).astype(np.int32)
    darkest_neighbour = cv2.erode(frame, NEIGHBOURS_ONLY).astype(np.int32)
    value = frame.astype(np.int32)
    too_bright = value > brightest_neighbour + threshold
    too_dark = value < darkest_neighbour - threshold
    return too_bright | too_dark


def replace_with_median(frame, bad_mask):
    median = cv2.medianBlur(frame, 3)
    return np.where(bad_mask, median, frame)


class StaticBadPixelMap:
    """Learns pixels that are permanently wrong from running per-pixel statistics."""

    LEARN_RATE = 0.05
    WARMUP_FRAMES = 20
    REBUILD_EVERY = 10

    def __init__(self, shape):
        self.mask = np.zeros(shape, bool)
        self.mean = None          # running mean of each pixel
        self.activity = None      # running mean of |frame-to-frame change|
        self.previous = None
        self.frame_count = 0

    def observe(self, frame, sigma):
        frame = frame.astype(np.float32)
        if self.mean is None:
            self.mean = frame.copy()
            self.activity = np.zeros_like(frame)
        else:
            change = np.abs(frame - self.previous)
            self.activity += self.LEARN_RATE * (change - self.activity)
            self.mean += self.LEARN_RATE * (frame - self.mean)
        self.previous = frame

        if self.frame_count >= self.WARMUP_FRAMES and self.frame_count % self.REBUILD_EVERY == 0:
            self.mask = self._rebuild(sigma)
        self.frame_count += 1

    def _rebuild(self, sigma):
        offset_outlier = np.abs(self.mean - cv2.medianBlur(self.mean, 3)) > max(10 * sigma, 60.0)
        typical_activity = float(np.median(self.activity)) + 1e-3
        stuck = self.activity < 0.1 * typical_activity       # no noise at all
        blinking = self.activity > 6 * typical_activity      # far too much noise
        return offset_outlier | stuck | blinking


class BadPixelCorrector:
    def __init__(self, cfg, shape):
        self.cfg = cfg
        self.static_map = StaticBadPixelMap(shape)

    def __call__(self, frame, sigma):
        threshold = max(self.cfg.spike_sigmas * sigma, 20.0)
        bad = find_spikes(frame, threshold) | self.static_map.mask
        fixed = replace_with_median(frame, bad)
        self.static_map.observe(frame, sigma)   # map is used from the next frame on
        return fixed
