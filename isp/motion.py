"""Motion estimation and alignment for motion-compensated temporal denoising.

Instead of making an in-between frame, the previous (already denoised) frame is
shifted along the estimated motion so it lines up with the current frame. Then
the two can be averaged even where things move: moving targets get less noise
and no ghost trail.

Optical flow: OpenCV DIS (fast, dense, one vector per pixel). It runs on an
8-bit copy of both frames that uses the same grey scale for both, so brightness
matches.
"""

import numpy as np
import cv2

MIN_MOTION_PX = 0.25    # smaller vectors are treated as 'not moving' (see align)


def to_8bit_pair(current, previous):
    """Both frames to 8 bit with the SAME scale, taken from the current frame."""
    low, high = np.percentile(current[::4, ::4], (1, 99))
    scale = 255.0 / max(high - low, 1.0)
    current8 = cv2.convertScaleAbs(current, alpha=scale, beta=-low * scale)
    previous8 = cv2.convertScaleAbs(previous, alpha=scale, beta=-low * scale)
    # light blur: flow should follow the scene, not the pixel noise
    return cv2.GaussianBlur(current8, (3, 3), 0), cv2.GaussianBlur(previous8, (3, 3), 0)


class MotionAligner:
    def __init__(self):
        self.dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST)  # as good as FAST here, 5 ms cheaper
        self.grid = None

    def estimate(self, current, previous):
        """Per-pixel motion: current(x) ~ previous(x + flow(x)).

        Full resolution on purpose: a half-resolution flow was tested and gave
        worse results on small moving targets for only ~3 ms saving.
        """
        current8, previous8 = to_8bit_pair(current, previous)
        return self.dis.calc(current8, previous8, None)

    def align(self, previous, flow):
        """Shift the previous frame so it lines up with the current one.

        Tiny vectors are set to zero: on a still scene the flow is just noise, and
        re-sampling the previous frame every frame would slowly blur fine detail.
        """
        height, width = previous.shape
        if self.grid is None:                    # pixel coordinates, made once
            xs, ys = np.meshgrid(np.arange(width, dtype=np.float32),
                                 np.arange(height, dtype=np.float32))
            self.grid = np.dstack([xs, ys])
        squared_length = flow[..., 0] ** 2 + flow[..., 1] ** 2
        flow[squared_length < MIN_MOTION_PX ** 2] = 0.0
        sample_at = cv2.add(self.grid, flow)
        return cv2.remap(previous, sample_at, None, cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REPLICATE)
