"""Stage 3a - offset stabiliser (anti-flicker for whole-frame jumps).

A shutter calibration (FFC) or gain switch shifts the whole frame by some counts
in one frame. The median difference to the previous frame measures that shift
(a moving object covers too few pixels to fake it). The jump is removed and then
released slowly, so a one-frame flash becomes a ~1 second fade.
"""

import numpy as np


def global_shift(frame, previous):
    """Median frame-to-frame difference on a 1/16 subsample."""
    return float(np.median(frame[::4, ::4] - previous[::4, ::4]))


class OffsetStabiliser:
    def __init__(self, cfg):
        self.cfg = cfg
        self.held_offset = 0.0    # part of a jump not yet released
        self.previous = None

    def __call__(self, frame, sigma):
        frame = frame - self.held_offset
        if self.previous is not None:
            shift = global_shift(frame, self.previous)
            if abs(shift) > 1.5 * sigma:          # a real jump, not just noise
                self.held_offset += shift
                frame = frame - shift
        self.previous = frame
        self.held_offset *= 1.0 - self.cfg.offset_release_rate
        return frame
