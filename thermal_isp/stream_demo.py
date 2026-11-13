"""Streaming feasibility check: replay a raw .bin as a 25 fps camera and measure per-frame processing time,
end-to-end latency and deadline misses. No video encoding, no look-ahead, frames are fed one by one.
Usage: python stream_demo.py data.bin [--fps 25] [--calib 16] [--max-frames N]"""
import argparse
import time
import numpy as np
from isp import ISP, find_bad_pixels, H, W


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bin")
    ap.add_argument("--fps", type=float, default=25.0)
    ap.add_argument("--calib", type=int, default=16, help="frames collected at start-up for the bad-pixel map")
    ap.add_argument("--max-frames", type=int, default=0)
    a = ap.parse_args()

    raw = np.memmap(a.bin, dtype="<u2", mode="r").reshape(-1, H, W)
    n = len(raw) if not a.max_frames else min(a.max_frames, len(raw))
    period = 1.0 / a.fps

    t = time.perf_counter()
    isp = ISP(bad_mask=find_bad_pixels(np.asarray(raw[:a.calib])))
    print("start-up: %d calibration frames, bad-pixel map in %.0f ms" % (a.calib, 1e3 * (time.perf_counter() - t)))

    proc = np.empty(n)
    for i in range(n):
        f = np.asarray(raw[i])                       # "camera" delivers one frame
        t = time.perf_counter()
        isp.process(f)                               # only this is timed
        proc[i] = time.perf_counter() - t

    # virtual real-time clock: frame i arrives at i*period; one worker, FIFO queue, nothing is dropped
    arrive = np.arange(n) * period
    finish = np.empty(n)
    free = 0.0
    for i in range(n):
        start = max(arrive[i], free)
        free = finish[i] = start + proc[i]
    lat = finish - arrive                            # processing + queueing
    miss = proc > period
    ms = lambda x: 1e3 * x
    print("frames %d @ %.0f fps, budget %.1f ms" % (n, a.fps, ms(period)))
    print("processing ms: mean %.1f  p50 %.1f  p95 %.1f  p99 %.1f  max %.1f" % (
        ms(proc.mean()), ms(np.percentile(proc, 50)), ms(np.percentile(proc, 95)), ms(np.percentile(proc, 99)), ms(proc.max())))
    print("frames over budget: %d (%.1f%%)" % (miss.sum(), 100 * miss.mean()))
    print("end-to-end latency ms (incl. queueing): mean %.1f  p99 %.1f  max %.1f  | algorithmic latency +1 frame (%.0f ms) for tone stats" % (
        ms(lat.mean()), ms(np.percentile(lat, 99)), ms(lat.max()), ms(period)))
    print("max queue backlog: %d frames; sustained load %.0f%% of one core" % (
        int(np.max(np.ceil(np.maximum(lat, 0) / period))) - 1 if n else 0, 100 * proc.mean() / period))
    ok = lat.max() < 2 * period and proc.mean() < 0.9 * period
    print("VERDICT:", "real-time capable at %.0f fps (latency < 2 frames)" % a.fps if ok else "NOT sustainable at %.0f fps on this machine" % a.fps)


if __name__ == "__main__":
    main()
