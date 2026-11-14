"""Stage 4 + 6 - split into base and detail layers, and prepare detail for re-injection.

The guided filter is edge-preserving: strong edges stay in the base layer and are
never boosted, so no halos. Fine texture goes to the detail layer, which is boosted
after the AGC has compressed the base. Only box filters -> cheap, line-buffer friendly.
"""

import numpy as np
import cv2


def box(image, radius):
    size = 2 * radius + 1
    return cv2.boxFilter(image, -1, (size, size))


def guided_filter(image, radius, eps):
    """Self-guided filter: smooths where local variance < eps, keeps edges where > eps.

    Overflow / precision safety:
      * the image is centred on its median first. variance = E[x^2] - E[x]^2 subtracts
        two huge numbers; at 16000 counts x^2 ~ 2.6e8, where float32 steps are 16-32,
        so the variance (true value ~1) came out between -16 and +32.
      * variance is clamped to >= 0, so variance + eps can never get close to 0.
    """
    centre = np.float32(np.median(image[::8, ::8]))
    image = image - centre
    mean = box(image, radius)
    variance = np.maximum(box(image * image, radius) - mean * mean, 0.0)
    a = variance / (variance + eps)      # ~1 on edges, ~0 in flat areas, always 0..1
    b = mean - a * mean
    return box(a, radius) * image + box(b, radius) + centre


def split_base_detail(frame, cfg, sigma):
    eps = (cfg.edge_sigmas * sigma) ** 2
    base = guided_filter(frame, cfg.guided_radius, eps)
    return base, frame - base


def soft_coring(detail, threshold):
    """Fades out detail around noise level smoothly (no hard cut -> no 'plastic' look).

    Large detail passes almost unchanged, detail ~ threshold is halved, tiny detail -> 0.
    """
    squared = detail * detail
    return detail * squared / (squared + threshold * threshold)
