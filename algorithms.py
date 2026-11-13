from dataclasses import dataclass, field
from typing import Any, Protocol

import cv2
import numpy as np

WIDTH = 640
HEIGHT = 512
MAX_14BIT = 16383


@dataclass(slots=True)
class FramePacket:
    """Data passed sequentially between ISP stages."""

    image: np.ndarray  # uint16 [0..16383] in sensor domain, uint8 [0..255] at output
    base: np.ndarray | None = None  # uint16 [0..16383] smooth thermal background
    detail: np.ndarray | None = None  # int16 signed high-frequency detail
    lut: np.ndarray | None = None  # 1024-entry tone mapping curve
    frame_idx: int = 0
    meta: dict[str, Any] = field(default_factory=dict)


class ISPStage(Protocol):
    """Interface every algorithm in this file must follow."""

    name: str

    def reset(self) -> None: ...
    def process(self, pkt: FramePacket) -> FramePacket: ...


class BadPixelReplacement:
    """Replaces dead, hot, and blinking (RTS) pixels via 3x3 median."""

    def __init__(self, threshold: int = 150) -> None:
        self.name = "BadPixelReplacement"
        self.threshold = threshold

    def reset(self) -> None:
        pass

    def process(self, pkt: FramePacket) -> FramePacket:
        med = cv2.medianBlur(pkt.image, 3)
        diff = np.abs(pkt.image.astype(np.int32) - med.astype(np.int32))
        mask = diff > self.threshold
        pkt.image = np.where(mask, med, pkt.image).astype(np.uint16)
        pkt.meta["bpr_fixed"] = int(mask.sum())
        return pkt


class ColumnDeStripe:
    """Removes vertical ROIC column fixed-pattern noise without blurring edges."""

    def __init__(
        self,
        edge_threshold: int = 40,
        kernel_width: int = 15,
        ema_alpha: float = 0.2,
    ) -> None:
        if kernel_width % 2 == 0:
            raise ValueError("kernel_width must be odd")
        self.name = "ColumnDeStripe"
        self.edge_threshold = edge_threshold
        self.kernel_width = kernel_width
        self.ema_alpha = ema_alpha
        self.smoothed_offset: np.ndarray | None = None

    def reset(self) -> None:
        self.smoothed_offset = None

    def process(self, pkt: FramePacket) -> FramePacket:
        img_i32 = pkt.image.astype(np.int32)
        horiz_diff = img_i32[:, 1:] - img_i32[:, :-1]
        clamped = np.clip(horiz_diff, -self.edge_threshold, self.edge_threshold)

        col_steps = np.zeros(img_i32.shape[1], dtype=np.float32)
        col_steps[1:] = clamped.mean(axis=0)
        profile = np.cumsum(col_steps)

        pad = self.kernel_width // 2
        padded = np.pad(profile, pad, mode="reflect")
        kernel = np.ones(self.kernel_width, dtype=np.float32) / self.kernel_width
        target_offset = profile - np.convolve(padded, kernel, mode="valid")

        if self.smoothed_offset is None:
            self.smoothed_offset = target_offset
        else:
            self.smoothed_offset = (
                1.0 - self.ema_alpha
            ) * self.smoothed_offset + self.ema_alpha * target_offset

        corrected = img_i32 - np.round(self.smoothed_offset).astype(np.int32)
        pkt.image = np.clip(corrected, 0, MAX_14BIT).astype(np.uint16)
        return pkt

class GuidedDetailSplit:
    """Splits 14-bit image into edge-preserving Base (uint16) and tight Detail (int16)."""

    def __init__(self, radius: int = 2, eps: float = 3500.0) -> None:
        self.name = "GuidedDetailSplit"
        self.radius = radius  # radius=2 (5x5) captures much finer edges than 9x9
        self.eps = eps  # Lower eps keeps strong edges in Base (zero halos) while extracting texture

    def reset(self) -> None:
        pass

    def process(self, pkt: FramePacket) -> FramePacket:
        ksize = (2 * self.radius + 1, 2 * self.radius + 1)
        img = pkt.image.astype(np.float32)

        mean_i = cv2.boxFilter(img, cv2.CV_32F, ksize)
        mean_ii = cv2.boxFilter(img * img, cv2.CV_32F, ksize)
        var_i = np.maximum(0.0, mean_ii - mean_i * mean_i)

        a = var_i / (var_i + self.eps)
        b = (1.0 - a) * mean_i

        mean_a = cv2.boxFilter(a, cv2.CV_32F, ksize)
        mean_b = cv2.boxFilter(b, cv2.CV_32F, ksize)

        base = mean_a * img + mean_b
        pkt.base = np.clip(base, 0, MAX_14BIT).astype(np.uint16)
        pkt.detail = np.round(img - base).astype(np.int16)
        return pkt

class PlateauToneMapper:
    """Maps 14-bit Base to 8-bit white-hot with center-weighted DRC, highlight roll-off, and slew-limited EMA."""

    def __init__(
        self,
        plateau_ratio: float = 0.0015,
        low_clip: float = 0.005,
        high_clip: float = 0.015,
        he_weight: float = 0.35,
        gamma: float = 1.30,
        ema_alpha: float = 0.08,
        max_lut_step: float = 2.0,
        max_output: float = 190.0,
        highlight_ceiling: float = 235.0,
        center_weight: float = 4.0,
        min_span_bins: int = 48,
    ) -> None:
        self.name = "PlateauToneMapper"
        self.plateau_ratio = plateau_ratio
        self.low_clip = low_clip
        self.high_clip = high_clip
        self.he_weight = he_weight
        self.gamma = gamma
        self.ema_alpha = ema_alpha
        self.max_lut_step = max_lut_step
        self.max_output = max_output
        self.highlight_ceiling = highlight_ceiling
        self.center_weight = center_weight
        self.min_span_bins = min_span_bins

        self.smoothed_lut: np.ndarray | None = None
        self._spatial_weights: np.ndarray | None = None

    def reset(self) -> None:
        self.smoothed_lut = None

    def _get_weights(self, shape: tuple[int, int]) -> np.ndarray:
        if self._spatial_weights is None or self._spatial_weights.shape != shape:
            h, w = shape
            weights = np.ones((h, w), dtype=np.float32)
            y0, y1 = int(h * 0.2), int(h * 0.8)
            x0, x1 = int(w * 0.2), int(w * 0.8)
            weights[y0:y1, x0:x1] = self.center_weight
            self._spatial_weights = weights
        return self._spatial_weights

    def process(self, pkt: FramePacket) -> FramePacket:
        source = pkt.base if pkt.base is not None else pkt.image
        idx = np.clip(source, 0, MAX_14BIT) >> 4

        # 1. Center-weighted histogram ignores transient heat traces/sky at frame borders during turns
        weights = self._get_weights(source.shape)
        hist = np.bincount(idx.ravel(), weights=weights.ravel(), minlength=1024).astype(
            np.float32
        )
        total_weight = float(hist.sum())

        cdf_raw = np.cumsum(hist)
        low_bin = int(np.searchsorted(cdf_raw, total_weight * self.low_clip))
        high_bin = int(np.searchsorted(cdf_raw, total_weight * (1.0 - self.high_clip)))

        # 2. Enforce minimum ADU span so uniform sky/smoke during a bank never spikes gain
        if high_bin - low_bin < self.min_span_bins:
            high_bin = min(1023, low_bin + self.min_span_bins)
            low_bin = max(0, high_bin - self.min_span_bins)

        hist_clipped = hist.copy()
        hist_clipped[:low_bin] = 0.0
        hist_clipped[high_bin + 1 :] = 0.0
        hist_clipped = np.minimum(hist_clipped, total_weight * self.plateau_ratio)

        # 3. Main scene curve [low_bin .. high_bin] -> [0 .. max_output] (keeps walls/roads mid-gray)
        cdf = np.cumsum(hist_clipped)
        cdf_min, cdf_max = cdf[low_bin], cdf[high_bin]
        if cdf_max > cdf_min:
            cdf_norm = np.clip((cdf - cdf_min) / (cdf_max - cdf_min), 0.0, 1.0)
        else:
            cdf_norm = np.linspace(0.0, 1.0, 1024, dtype=np.float32)

        bins = np.arange(1024, dtype=np.float32)
        span = float(max(1, high_bin - low_bin))
        linear_norm = np.clip((bins - low_bin) / span, 0.0, 1.0)

        blended = (1.0 - self.he_weight) * linear_norm + self.he_weight * cdf_norm
        target_lut = np.power(blended, self.gamma) * self.max_output

        # 4. Soft highlight roll-off [high_bin .. +headroom] -> [max_output .. highlight_ceiling]
        # Keeps true hot spots bright without dragging house walls up with them
        if high_bin < 1023 and self.highlight_ceiling > self.max_output:
            headroom_bins = float(max(32, (high_bin - low_bin) // 2))
            over = np.clip((bins - high_bin) / headroom_bins, 0.0, 1.0)
            target_lut = np.where(
                bins > high_bin,
                self.max_output + (self.highlight_ceiling - self.max_output) * over,
                target_lut,
            )

        # 5. Slew-rate-limited EMA prevents 0.5s brightness flashes during rapid turns
        if self.smoothed_lut is None:
            self.smoothed_lut = target_lut
        else:
            step = self.ema_alpha * (target_lut - self.smoothed_lut)
            step = np.clip(step, -self.max_lut_step, self.max_lut_step)
            self.smoothed_lut = self.smoothed_lut + step

        pkt.lut = self.smoothed_lut
        pkt.image = self.smoothed_lut[idx].astype(np.uint8)
        return pkt

class DetailFusion:
    """Noise-cores signed Detail and fuses it onto the 8-bit Base."""

    def __init__(self, core_threshold: int = 5, gain: float = 0.85) -> None:
        self.name = "DetailFusion"
        self.core_threshold = core_threshold  # Low coring preserves subtle foliage/road grain
        self.gain = gain

    def reset(self) -> None:
        pass

    def process(self, pkt: FramePacket) -> FramePacket:
        if pkt.detail is None:
            return pkt

        base_f = pkt.image.astype(np.float32)
        d = pkt.detail.astype(np.float32)
        cored = np.sign(d) * np.maximum(0.0, np.abs(d) - self.core_threshold)

        # Keep strong texture on warm walls/roads while still preventing 255 white clipping
        pos_scale = np.clip((255.0 - base_f) / 110.0, 0.50, 1.0)
        effective_gain = np.where(cored > 0.0, self.gain * pos_scale, self.gain)

        fused = base_f + effective_gain * cored
        pkt.image = np.clip(fused, 0, 255).astype(np.uint8)
        return pkt

class MotionAdaptiveTemporalFilter:

    """Reduces temporal sensor noise on static sky/areas without blurring drone motion."""

    def __init__(
        self,
        motion_threshold: int = 16,
        static_alpha: float = 0.45,
    ) -> None:
        self.name = "MotionAdaptiveTemporalFilter"
        self.motion_threshold = motion_threshold
        self.static_alpha = static_alpha
        self.prev_frame: np.ndarray | None = None

    def reset(self) -> None:
        self.prev_frame = None

    def process(self, pkt: FramePacket) -> FramePacket:
        curr = pkt.image.astype(np.float32)
        if self.prev_frame is None:
            self.prev_frame = curr.copy()
            return pkt

        diff = np.abs(curr - self.prev_frame)
        motion_weight = np.clip(diff / float(self.motion_threshold), 0.0, 1.0)
        alpha = self.static_alpha + (1.0 - self.static_alpha) * motion_weight

        filtered = alpha * curr + (1.0 - alpha) * self.prev_frame
        self.prev_frame = filtered
        pkt.image = np.clip(np.round(filtered), 0, MAX_14BIT).astype(np.uint16)
        return pkt


class OpticalSharpen:

    """Optional final 3x3 binomial unsharp mask to restore 1-pixel lens crispness."""

    def __init__(self, amount: float = 0.45, noise_core: int = 2) -> None:
        self.name = "OpticalSharpen"
        self.amount = amount
        self.noise_core = noise_core

    def reset(self) -> None:
        pass

    def process(self, pkt: FramePacket) -> FramePacket:
        img = pkt.image.astype(np.float32)
        # Separable [1, 2, 1] / 4 blur (2 line buffers on FPGA, zero multipliers)
        blur = cv2.GaussianBlur(img, (3, 3), sigmaX=0.85)
        high_freq = img - blur
        cored = np.sign(high_freq) * np.maximum(0.0, np.abs(high_freq) - self.noise_core)
        pkt.image = np.clip(img + self.amount * cored, 0, 255).astype(np.uint8)
        return pkt

class CLAHEToneMapper:
    """256-bin active-range 4x4 CLAHE with temporally smoothed tile LUTs (no block artifacts)."""

    def __init__(
        self,
        grid_size: tuple[int, int] = (4, 4),
        clip_limit: float = 2.0,
        local_blend: float = 0.45,
        gamma: float = 1.25,
        max_output: float = 205.0,
        ema_alpha: float = 0.10,
        max_bound_step: float = 12.0,
        center_weight: float = 4.0,
    ) -> None:
        self.name = "CLAHEToneMapper"
        self.grid_size = grid_size  # (rows, cols)
        self.clip_limit = clip_limit
        self.local_blend = local_blend
        self.gamma = gamma
        self.max_output = max_output
        self.ema_alpha = ema_alpha
        self.max_bound_step = max_bound_step
        self.center_weight = center_weight

        self.low_bound: float | None = None
        self.high_bound: float | None = None
        self.smoothed_tile_luts: np.ndarray | None = None
        self._spatial_weights: np.ndarray | None = None
        self._interp_cache: tuple[Any, ...] | None = None

    def reset(self) -> None:
        self.low_bound = None
        self.high_bound = None
        self.smoothed_tile_luts = None

    def _get_weights(self, shape: tuple[int, int]) -> np.ndarray:
        if self._spatial_weights is None or self._spatial_weights.shape != shape:
            h, w = shape
            weights = np.ones((h, w), dtype=np.float32)
            weights[int(h * 0.2) : int(h * 0.8), int(w * 0.2) : int(w * 0.8)] = (
                self.center_weight
            )
            self._spatial_weights = weights
        return self._spatial_weights

    def _get_interp_grid(self, shape: tuple[int, int]) -> tuple[Any, ...]:
        if self._interp_cache is None or self._interp_cache[0] != shape:
            h, w = shape
            gy, gx = self.grid_size
            th, tw = h / float(gy), w / float(gx)

            y = np.arange(h, dtype=np.float32)
            fy = np.clip((y - 0.5 * th) / th, 0.0, gy - 1.0001)
            ty0 = np.floor(fy).astype(np.int32)
            ty1 = np.minimum(ty0 + 1, gy - 1)
            wy = (fy - ty0)[:, None]

            x = np.arange(w, dtype=np.float32)
            fx = np.clip((x - 0.5 * tw) / tw, 0.0, gx - 1.0001)
            tx0 = np.floor(fx).astype(np.int32)
            tx1 = np.minimum(tx0 + 1, gx - 1)
            wx = (fx - tx0)[None, :]

            self._interp_cache = (shape, ty0[:, None], ty1[:, None], wy, tx0[None, :], tx1[None, :], wx)
        return self._interp_cache

    def process(self, pkt: FramePacket) -> FramePacket:
        source = pkt.base if pkt.base is not None else pkt.image
        h, w = source.shape

        # 1. Slew-limited center-weighted active range prevents drone-turn flashes
        idx_10 = np.clip(source, 0, MAX_14BIT) >> 4
        weights = self._get_weights((h, w))
        hist_10 = np.bincount(idx_10.ravel(), weights=weights.ravel(), minlength=1024)
        cdf_10 = np.cumsum(hist_10)
        total_w = float(cdf_10[-1])

        target_low = float(np.searchsorted(cdf_10, total_w * 0.005) << 4)
        target_high = float(np.searchsorted(cdf_10, total_w * 0.985) << 4)
        if target_high - target_low < 768.0:
            target_high = min(float(MAX_14BIT), target_low + 768.0)
            target_low = max(0.0, target_high - 768.0)

        if self.low_bound is None or self.high_bound is None:
            self.low_bound, self.high_bound = target_low, target_high
        else:
            self.low_bound += float(
                np.clip(self.ema_alpha * (target_low - self.low_bound), -self.max_bound_step, self.max_bound_step)
            )
            self.high_bound += float(
                np.clip(self.ema_alpha * (target_high - self.high_bound), -self.max_bound_step, self.max_bound_step)
            )

        # Quantize active thermal range into 256 bins (every bin actively used)
        span = max(256.0, self.high_bound - self.low_bound)
        q = np.clip((source.astype(np.float32) - self.low_bound) / span * 255.0, 0, 255).astype(np.uint8)

        # 2. Compute clipped CDF per tile
        gy, gx = self.grid_size
        th, tw = h // gy, w // gx
        clip_pixels = max(1.0, self.clip_limit * (th * tw) / 256.0)
        linear_ramp = np.linspace(0.0, 1.0, 256, dtype=np.float32)
        target_luts = np.empty((gy, gx, 256), dtype=np.float32)

        for ty in range(gy):
            for tx in range(gx):
                tile = q[ty * th : (ty + 1) * th, tx * tw : (tx + 1) * tw]
                hist = np.bincount(tile.ravel(), minlength=256).astype(np.float32)
                excess = float(np.maximum(0.0, hist - clip_pixels).sum())
                hist = np.minimum(hist, clip_pixels) + (excess / 256.0)

                cdf = np.cumsum(hist)
                denom = max(1e-5, float(cdf[-1] - cdf[0]))
                cdf_norm = (cdf - cdf[0]) / denom
                target_luts[ty, tx] = (1.0 - self.local_blend) * linear_ramp + self.local_blend * cdf_norm

        if self.smoothed_tile_luts is None:
            self.smoothed_tile_luts = target_luts
        else:
            self.smoothed_tile_luts = (
                1.0 - self.ema_alpha
            ) * self.smoothed_tile_luts + self.ema_alpha * target_luts

        # 3. Smooth bilinear interpolation across tile centers (eliminates square borders)
        _, ty0, ty1, wy, tx0, tx1, wx = self._get_interp_grid((h, w))
        v00 = self.smoothed_tile_luts[ty0, tx0, q]
        v01 = self.smoothed_tile_luts[ty0, tx1, q]
        v10 = self.smoothed_tile_luts[ty1, tx0, q]
        v11 = self.smoothed_tile_luts[ty1, tx1, q]

        blended = (1.0 - wy) * ((1.0 - wx) * v00 + wx * v01) + wy * ((1.0 - wx) * v10 + wx * v11)
        mapped = np.power(np.clip(blended, 0.0, 1.0), self.gamma) * self.max_output

        pkt.image = np.clip(mapped, 0, 255).astype(np.uint8)
        return pkt