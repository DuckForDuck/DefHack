"""Small numeric helpers shared by several modules."""

from __future__ import annotations

import cv2
import numpy as np

# Laplacian-like kernel: pixel minus the mean of its 4 neighbours.
# For white noise with std s the result has std s * sqrt(1 + 4/16).
_LAP4 = np.array([[0, -0.25, 0], [-0.25, 1, -0.25], [0, -0.25, 0]], np.float32)
_LAP4_GAIN = np.sqrt(1.25)

# 3x3 kernel without the centre: max/min of the 8 neighbours.
NEIGHBOURS_8 = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], np.uint8)


def estimate_noise_sigma(img: np.ndarray, mask: np.ndarray | None = None,
                         step: int = 2) -> float:
    """Robust estimate of the std of white noise in one frame.

    Uses the median absolute deviation of a 4-neighbour Laplacian. Scene
    edges affect only a small share of pixels, so the median ignores them.
    ``mask`` (True = ignore) excludes for example bad pixels. ``step``
    subsamples the Laplacian for speed (2 → 4x fewer values, same result
    within ≈ 0.5 %).
    """
    lap = cv2.filter2D(img.astype(np.float32, copy=False), -1, _LAP4,
                       borderType=cv2.BORDER_REFLECT)[1:-1:step, 1:-1:step]
    if mask is not None:
        lap = lap[~mask[1:-1:step, 1:-1:step]]
    lap = lap.ravel()
    mad = np.median(np.abs(lap - np.median(lap)))
    return float(1.4826 * mad / _LAP4_GAIN)


def gaussian(img: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian blur with mirrored borders (float32)."""
    return cv2.GaussianBlur(img, (0, 0), sigma, borderType=cv2.BORDER_REFLECT)


def neighbour_max_min(img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Max and min of the 8 neighbours of each pixel (centre excluded)."""
    mx = cv2.dilate(img, NEIGHBOURS_8, borderType=cv2.BORDER_REFLECT)
    mn = cv2.erode(img, NEIGHBOURS_8, borderType=cv2.BORDER_REFLECT)
    return mx, mn
