"""Stage 2 - stripe removal (column fixed-pattern noise + row noise).

Done before denoising: column stripes are fixed, so averaging frames can't remove them.
  * columns: offsets are static -> learned slowly over many frames
  * rows   : readout noise is new every frame -> measured and removed per frame
Pixels near strong scene edges are excluded from the estimate, otherwise
building edges or people would be 'corrected' into new stripes.
"""

import numpy as np
import cv2

WINDOW = 15          # high-pass window along the line
EDGE_MARGIN = 17     # how far around an edge pixels are excluded
MIN_CLEAN_SHARE = 0.3
SMOOTH_ALONG = 7     # smoothing along the stripe before looking for scene edges


def edge_mask(frame, threshold, axis):
    """Pixels close to a strong gradient along `axis` (1 = horizontal, 0 = vertical).

    The gradient is measured on a copy smoothed ALONG the stripe direction (7 px):
    noise drops ~2.6x, while scene edges and the stripes themselves stay. On the raw
    frame plain noise crossed the threshold so often that, after widening, ~70 % of
    pixels (and half of all columns) were excluded and never corrected.
    """
    along = (1, SMOOTH_ALONG) if axis == 1 else (SMOOTH_ALONG, 1)   # cv2 ksize = (w, h)
    smooth = cv2.blur(frame, along, borderType=cv2.BORDER_REFLECT)
    gradient = np.zeros_like(frame)
    if axis == 1:
        gradient[:, 1:-1] = np.abs(smooth[:, 2:] - smooth[:, :-2])
        widen = np.ones((1, EDGE_MARGIN), np.uint8)
    else:
        gradient[1:-1, :] = np.abs(smooth[2:, :] - smooth[:-2, :])
        widen = np.ones((EDGE_MARGIN, 1), np.uint8)
    return cv2.dilate((gradient > threshold).astype(np.uint8), widen) > 0


def high_pass(frame, axis):
    """Frame minus its local mean along one direction: keeps stripes, removes scene."""
    size = (WINDOW, 1) if axis == 1 else (1, WINDOW)
    return frame - cv2.blur(frame, size, borderType=cv2.BORDER_REFLECT)


def stripe_profile(frame, threshold, axis):
    """Average stripe offset per column (axis=1) or per row (axis=0), using clean pixels only.

    Returns one value per column/row; zero where too few clean pixels exist.
    """
    detail = high_pass(frame, axis)
    clean = (np.abs(detail) < threshold) & ~edge_mask(frame, threshold, axis)
    sum_axis = 0 if axis == 1 else 1
    count = clean.sum(axis=sum_axis)
    profile = (detail * clean).sum(axis=sum_axis) / np.maximum(count, 1)
    line_length = frame.shape[sum_axis]
    profile[count < MIN_CLEAN_SHARE * line_length] = 0.0
    return profile.astype(np.float32)


class StripeRemover:
    def __init__(self, cfg, width):
        self.cfg = cfg
        self.column_offsets = np.zeros(width, np.float32)
        self.frames_seen = 0

    def __call__(self, frame, sigma):
        threshold = 3.0 * sigma
        # columns: correct with what we already know, then learn the remaining error
        frame = frame - self.column_offsets[None, :]
        remaining = stripe_profile(frame, threshold, axis=1)
        # start-up: plain running average (1, 1/2, 1/3 ...) so stripes are gone within a
        # few frames instead of ~2 s; afterwards the slow configured rate takes over
        self.frames_seen += 1
        rate = max(self.cfg.column_learn_rate, 1.0 / self.frames_seen)
        self.column_offsets += rate * remaining
        # rows: measure on this frame, remove right away
        row_offsets = stripe_profile(frame, threshold, axis=0)
        return frame - row_offsets[:, None]
