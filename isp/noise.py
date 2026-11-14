"""Noise level estimate. Every threshold in the pipeline is expressed in noise sigmas."""

import numpy as np


def estimate_noise_sigma(frame):
    """Robust noise estimate from differences of horizontal neighbours (MAD).

    The median ignores edges, so the result is the sensor noise, not scene contrast.
    Uses every 4th row: plenty of samples and cheap.
    """
    differences = np.abs(np.diff(frame[::4, :], axis=1))
    mad_to_sigma = 1.4826
    sigma = mad_to_sigma * float(np.median(differences)) / np.sqrt(2.0)
    return float(max(sigma, 0.5))   # plain float: a np.float64 would turn float32 frames into float64


class NoiseTracker:
    """Keeps a smoothed noise sigma across frames (90 % old, 10 % new)."""

    def __init__(self):
        self.sigma = None

    def update(self, frame):
        new_sigma = estimate_noise_sigma(frame)
        if self.sigma is None:
            self.sigma = new_sigma
        else:
            self.sigma = 0.9 * self.sigma + 0.1 * new_sigma
        return self.sigma
