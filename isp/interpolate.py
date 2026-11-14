"""Frame interpolation: create a new frame halfway between the previous and the current
output frame, doubling the frame rate (25 -> 50 fps).

  blend : plain average of the two frames. Cheap, but a moving object shows up
          twice, half transparent (= ghosting).
  flow  : motion-compensated. Optical flow (DIS, fast) estimates how every pixel
          moved; both frames are shifted half way along that motion and then
          averaged, so a moving object appears once, at its in-between position.
          Where the two shifted frames still disagree (an object uncovering
          background), the current frame is used, so no double edges.

Works on the final 8-bit frames: cheaper, and flow is more reliable on a
contrast-enhanced image than on raw 14-bit data.
Note: the in-between frame is an estimate, not a measurement. It adds one frame
of latency (the next frame must arrive before the in-between one can be made).
"""

import numpy as np
import cv2

DISAGREEMENT_LEVELS = 20      # warped frames differing more than this = occlusion


def blend(previous, current):
    return cv2.addWeighted(previous, 0.5, current, 0.5, 0)


def shift_by_flow(image, flow, fraction):
    """Sample `image` at position + fraction * flow (backward warping)."""
    height, width = image.shape
    grid_x, grid_y = np.meshgrid(np.arange(width, dtype=np.float32),
                                 np.arange(height, dtype=np.float32))
    map_x = grid_x + fraction * flow[..., 0]
    map_y = grid_y + fraction * flow[..., 1]
    return cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


class FrameInterpolator:
    def __init__(self, mode="flow"):
        self.mode = mode
        if mode == "flow":
            self.flow = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)

    def __call__(self, previous, current):
        if self.mode == "blend":
            return blend(previous, current)

        # motion from the middle towards each frame (approximated at the same pixel)
        forward = self.flow.calc(previous, current, None)    # previous -> current
        backward = self.flow.calc(current, previous, None)   # current -> previous
        from_previous = shift_by_flow(previous, backward, 0.5)
        from_current = shift_by_flow(current, forward, 0.5)

        middle = cv2.addWeighted(from_previous, 0.5, from_current, 0.5, 0)
        disagree = cv2.absdiff(from_previous, from_current) > DISAGREEMENT_LEVELS
        middle[disagree] = from_current[disagree]
        return middle
