"""Reading raw .bin sequences and writing MP4 video."""

import subprocess

import numpy as np

from .config import WIDTH, HEIGHT


def open_frames(path, width=WIDTH, height=HEIGHT):
    """All frames of a raw uint16 little-endian .bin, memory-mapped (not loaded to RAM)."""
    data = np.memmap(path, dtype="<u2", mode="r")
    count = data.size // (width * height)
    return data[: count * width * height].reshape(count, height, width)


class Mp4Writer:
    """Pipes 8-bit grey frames into ffmpeg (H.264)."""

    def __init__(self, path, width, height, fps=25):
        command = ["ffmpeg", "-y", "-loglevel", "error",
                   "-f", "rawvideo", "-pix_fmt", "gray", "-s", f"{width}x{height}",
                   "-r", str(fps), "-i", "-",
                   "-c:v", "libx264", "-preset", "veryfast", "-crf", "14",
                   "-pix_fmt", "yuv420p", path]
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE)

    def write(self, image):
        self.process.stdin.write(np.ascontiguousarray(image).tobytes())

    def close(self):
        self.process.stdin.close()
        self.process.wait()
