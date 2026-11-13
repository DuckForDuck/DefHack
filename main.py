import argparse
import time
from pathlib import Path

import cv2
import numpy as np

from algorithms import (
    HEIGHT,
    WIDTH,
    BadPixelReplacement,
    CLAHEToneMapper,
    ColumnDeStripe,
    DetailFusion,
    FramePacket,
    GuidedDetailSplit,
    ISPStage,
    MotionAdaptiveTemporalFilter,
    OpticalSharpen,
)


def build_stages() -> list[ISPStage]:
    """Configure active algorithms and their order here."""
    return [
        BadPixelReplacement(threshold=150),
        ColumnDeStripe(edge_threshold=40, kernel_width=15, ema_alpha=0.2),
        MotionAdaptiveTemporalFilter(motion_threshold=16, static_alpha=0.45),
        GuidedDetailSplit(radius=2, eps=3500.0),
        CLAHEToneMapper(
            grid_size=(4, 4),
            clip_limit=2.0,
            local_blend=0.45,
            gamma=1.25,
            max_output=205.0,
        ),
        DetailFusion(core_threshold=5, gain=0.75),
        OpticalSharpen(amount=0.35, noise_core=2),
    ]


class ISPPipeline:
    """Runs stages in sequence and tracks per-stage latency."""

    def __init__(self, stages: list[ISPStage]) -> None:
        self.stages = stages
        self.stage_times_ms: dict[str, float] = {s.name: 0.0 for s in stages}
        self.frame_count = 0

    def reset(self) -> None:
        for stage in self.stages:
            stage.reset()
        self.stage_times_ms = {s.name: 0.0 for s in self.stages}
        self.frame_count = 0

    def run_frame(self, raw_frame: np.ndarray) -> FramePacket:
        pkt = FramePacket(image=raw_frame, frame_idx=self.frame_count)
        for stage in self.stages:
            t0 = time.perf_counter()
            pkt = stage.process(pkt)
            self.stage_times_ms[stage.name] += (time.perf_counter() - t0) * 1000.0
        self.frame_count += 1
        return pkt

    def report_timing(self) -> str:
        n = max(1, self.frame_count)
        parts = [f"{k}: {v / n:.2f}ms" for k, v in self.stage_times_ms.items()]
        total = sum(self.stage_times_ms.values()) / n
        return f"Total: {total:.2f}ms/frame | " + " | ".join(parts)


def raw_linear_preview(frame: np.ndarray) -> np.ndarray:
    """Simple min-max 8-bit scaling for before/after visual comparison."""
    f_min, f_max = int(frame.min()), int(frame.max())
    if f_max <= f_min:
        return np.zeros_like(frame, dtype=np.uint8)
    return ((frame.astype(np.float32) - f_min) / (f_max - f_min) * 255.0).astype(
        np.uint8
    )


def process_bin_to_mp4(
    bin_path: Path,
    out_path: Path,
    fps: float = 25.0,
    side_by_side: bool = False,
) -> None:
    raw = np.fromfile(bin_path, dtype="<u2")
    frame_pixels = WIDTH * HEIGHT
    num_frames = raw.size // frame_pixels
    if num_frames == 0:
        raise ValueError(f"File too small for {WIDTH}x{HEIGHT} frame: {bin_path}")

    frames = raw[: num_frames * frame_pixels].reshape((num_frames, HEIGHT, WIDTH))
    pipeline = ISPPipeline(build_stages())
    pipeline.reset()

    out_width = WIDTH * 2 if side_by_side else WIDTH
    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (out_width, HEIGHT),
    )

    for i in range(num_frames):
        pkt = pipeline.run_frame(frames[i])
        vis = (
            np.hstack([raw_linear_preview(frames[i]), pkt.image])
            if side_by_side
            else pkt.image
        )
        writer.write(cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR))

    writer.release()
    print(f"Saved {num_frames} frames -> {out_path}")
    print(pipeline.report_timing())


def main() -> None:
    parser = argparse.ArgumentParser(description="Thermal 14-bit ISP Pipeline")
    parser.add_argument("input_bin", type=Path, help="Input 640x512 uint16 .bin file")
    parser.add_argument("-o", "--output", type=Path, default=Path("output.mp4"))
    parser.add_argument("--fps", type=float, default=25.0)
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Render raw linear scale (left) vs ISP output (right)",
    )
    args = parser.parse_args()
    process_bin_to_mp4(
        args.input_bin,
        args.output,
        fps=args.fps,
        side_by_side=args.compare,
    )


if __name__ == "__main__":
    main()