"""CPU training demo for a small image-enhancement network on the dataset from make_dataset.py.
Needs PyTorch (not part of requirements.txt).  Usage:
    python train_demo.py dataset --minutes 5 [--model unet|dncnn] [--batch 16]
Prints seconds per step, loss, and a test-split comparison (real raw patches vs clean targets):
residual noise of the network output, sharpness kept, against the raw input and a plain Gaussian blur baseline."""
import argparse
import glob
import os
import time

import cv2
import numpy as np
import torch
import torch.nn as nn

from make_dataset import NoiseBank, robust_sigma, sharp_proxy

SCALE = 16.0                                            # counts -> network units (sensor noise sigma is about 6 counts)


class DnCNN(nn.Module):
    """Residual denoiser: out = x - net(x). Full resolution everywhere, so it is the slowest per parameter."""

    def __init__(self, layers=8, ch=32):
        super().__init__()
        L = [nn.Conv2d(1, ch, 3, padding=1), nn.ReLU(inplace=True)]
        for _ in range(layers - 2):
            L += [nn.Conv2d(ch, ch, 3, padding=1), nn.ReLU(inplace=True)]
        L += [nn.Conv2d(ch, 1, 3, padding=1)]
        self.net = nn.Sequential(*L)

    def forward(self, x):
        return x - self.net(x)


class UNet(nn.Module):
    """Small 2-level U-Net: most of the work happens at 1/2 and 1/4 resolution, so it is several times cheaper than DnCNN
    and sees a wider context (useful because the real noise is low-frequency dominated)."""

    def __init__(self, c=16):
        super().__init__()
        cb = lambda i, o: nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(o, o, 3, padding=1), nn.ReLU(inplace=True))
        self.e1, self.e2, self.b = cb(1, c), cb(c, 2 * c), cb(2 * c, 4 * c)
        self.u2, self.d2 = nn.ConvTranspose2d(4 * c, 2 * c, 2, 2), cb(4 * c, 2 * c)
        self.u1, self.d1 = nn.ConvTranspose2d(2 * c, c, 2, 2), cb(2 * c, c)
        self.out = nn.Conv2d(c, 1, 1)
        self.pool = nn.AvgPool2d(2)

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(self.pool(e1))
        b = self.b(self.pool(e2))
        d2 = self.d2(torch.cat([self.u2(b), e2], 1))
        d1 = self.d1(torch.cat([self.u1(d2), e1], 1))
        return x - self.out(d1)


def prep(raw):
    """(N,H,W) uint16 -> normalised float tensor and the per-patch offsets used to undo it."""
    x = raw.astype(np.float32)
    off = np.median(x.reshape(len(x), -1), axis=1)[:, None, None]
    return torch.from_numpy((x - off) / SCALE)[:, None], off


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("--minutes", type=float, default=5.0)
    ap.add_argument("--model", default="unet", choices=["unet", "dncnn"])
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--threads", type=int, default=0)
    a = ap.parse_args()
    if a.threads:
        torch.set_num_threads(a.threads)
    torch.manual_seed(0)
    rng = np.random.default_rng(0)

    f = lambda s: glob.glob(os.path.join(a.dataset, "*_%s.npz" % s))[0]
    tr, te = np.load(f("train")), np.load(f("test"))
    bank = NoiseBank.load(glob.glob(os.path.join(a.dataset, "*_noise_bank.npz"))[0])
    model = UNet() if a.model == "unet" else DnCNN()
    params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3)
    print("model %s: %d parameters, %d torch threads, %d train patches" % (a.model, params, torch.get_num_threads(), len(tr["raw"])))

    t0, steps, ema, last = time.time(), 0, None, time.time()
    N = len(tr["raw"])
    while time.time() - t0 < a.minutes * 60:
        idx = rng.integers(0, N, a.batch)
        noisy = np.empty((a.batch,) + tr["raw"].shape[1:], np.uint16)
        for k, i in enumerate(idx):
            # half of the inputs are the real raw patches, half get a fresh real noise residual on the clean patch
            noisy[k] = tr["raw"][i] if rng.random() < 0.5 else bank.add(tr["clean"][i], rng, exclude_frame=int(tr["frame"][i]))
        x, off = prep(noisy)
        y = torch.from_numpy((tr["clean"][idx].astype(np.float32) - off) / SCALE)[:, None]
        if rng.random() < 0.5:
            x, y = x.flip(3), y.flip(3)
        loss = torch.mean(torch.abs(model(x) - y))
        opt.zero_grad()
        loss.backward()
        opt.step()
        steps += 1
        ema = loss.item() if ema is None else 0.95 * ema + 0.05 * loss.item()
        if time.time() - last > 30:
            print("  step %5d  %.2f s/step  L1 %.4f (x%.0f counts = %.2f)" % (steps, (time.time() - t0) / steps, ema, SCALE, ema * SCALE), flush=True)
            last = time.time()
    el = time.time() - t0
    print("trained %d steps (batch %d) in %.0f s -> %.2f s/step, %.1f patches/s" % (steps, a.batch, el, el / steps, steps * a.batch / el))

    # test on REAL raw patches from the test split (frames the network never saw)
    model.eval()
    raw, cln = te["raw"], te["clean"].astype(np.float32)
    outs = []
    t1 = time.time()
    with torch.no_grad():
        for i in range(0, len(raw), 32):
            x, off = prep(raw[i:i + 32])
            outs.append(model(x).numpy()[:, 0] * SCALE + off)
    out = np.concatenate(outs)
    inf = (time.time() - t1) / len(raw)
    blur = np.stack([cv2.GaussianBlur(r.astype(np.float32), (0, 0), 1.0) for r in raw])
    print("inference %.1f ms per 128x128 patch (=%.0f ms per 640x512 frame, %d threads)" % (1e3 * inf, 1e3 * inf * 20, torch.get_num_threads()))
    print("test split, %d real patches; error vs clean target (counts), sharpness vs target:" % len(raw))
    for nm, v in (("raw input", raw.astype(np.float32)), ("gaussian blur 1.0", blur), ("network output", out)):
        e = v - cln
        print("  %-18s residual sigma %.2f   RMSE %.2f   sharpness %.3f" % (nm, np.median([robust_sigma(x) for x in e]), np.sqrt(np.mean(e ** 2)),
                                                                          np.mean([sharp_proxy(x) for x in v]) / np.mean([sharp_proxy(x) for x in cln])))


if __name__ == "__main__":
    main()
