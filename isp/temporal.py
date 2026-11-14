"""Stage 3b - motion-compensated, motion-adaptive temporal denoise.

    aligned  = previous_output shifted along the motion (motion.py)
    output   = aligned + weight * (frame - aligned)

1. Motion compensation: the previous frame is first lined up with the current one,
   so moving objects can be averaged too (less noise on targets, no trail).
2. Safety net: where aligned and current still differ clearly (flow wrong, object
   appearing/uncovered), weight -> 1 and the new frame passes through: no ghosting.
   There a light 3x3 blur stops the target having a noise halo.

Still areas: weight 0.25 -> averages several frames (~2.6x less noise).
Needs one frame buffer (the previous output).
"""

import numpy as np
import cv2

from .motion import MotionAligner


def motion_amount(difference, sigma):
    """0 = frames agree, 1 = clearly different, in units of the difference's own noise.

    A 3x3 mean of the difference lowers noise 3x but keeps real changes.
    """
    local_difference = np.abs(cv2.blur(difference, (3, 3)))
    noise_of_local_difference = float(sigma) / 3.0 * 1.4142135   # plain float keeps float32
    in_sigmas = local_difference / noise_of_local_difference
    return np.clip((in_sigmas - 1.5) / 3.0, 0.0, 1.0)


class TemporalDenoiser:
    def __init__(self, cfg):
        self.cfg = cfg
        self.previous_output = None
        self.aligner = MotionAligner()

    def __call__(self, frame, sigma):
        if self.previous_output is None or self.previous_output.shape != frame.shape:
            self.previous_output = frame.copy()
            return frame

        reference = self.previous_output
        if self.cfg.motion_compensation:
            flow = self.aligner.estimate(frame, reference)
            reference = self.aligner.align(reference, flow)

        difference = frame - reference
        mismatch = motion_amount(difference, sigma)
        still = self.cfg.still_weight
        weight = still + (1.0 - still) * mismatch

        averaged = reference + weight * difference
        self.previous_output = averaged

        blurred = cv2.GaussianBlur(frame, (3, 3), 0)
        blur_share = 0.6 * mismatch
        return averaged * (1.0 - blur_share) + blurred * blur_share
