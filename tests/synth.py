"""Synthetic test material, made on the fly so the repository carries no footage.

The "person" is a soft-edged ellipse that drifts right and sways, on a transparent background:
enough for layout detection, follow mode, motion, the time effects and body-part masks.
"""
from __future__ import annotations

import math
import wave
from pathlib import Path

import numpy as np
from PIL import Image

W, H = 96, 128


def make_frames(folder: Path, n: int = 12, w: int = W, h: int = H, move: float = 2.0) -> list[Path]:
    folder.mkdir(parents=True, exist_ok=True)
    yy, xx = np.mgrid[:h, :w].astype(np.float32)
    out = []
    for i in range(n):
        cx = w * 0.42 + move * i
        cy = h * 0.5 + 3.0 * math.sin(i * 0.7)
        r = ((xx - cx) / (w * 0.23)) ** 2 + ((yy - cy) / (h * 0.39)) ** 2
        a = np.clip((1.0 - r) * 12.0, 0, 1)                    # soft edge, like a real cut-out
        rgb = np.stack([150 + 80 * (yy / h), 110 + 60 * np.sin(xx / 7.0 + i), 90 + 0 * xx], -1)
        img = np.dstack([np.clip(rgb, 0, 255), a * 255]).astype(np.uint8)
        p = folder / f"f_{i + 1:04d}.png"
        Image.fromarray(img, "RGBA").save(p)
        out.append(p)
    return out


def make_plate(path: Path, w: int = W, h: int = H) -> Path:
    yy, xx = np.mgrid[:h, :w]
    rgb = np.stack([30 + xx, 60 + 0 * xx, 90 + yy // 2], -1).astype(np.uint8)
    Image.fromarray(rgb, "RGB").save(path)
    return path


def make_landmarks(path: Path, n: int = 12) -> Path:
    """A standing figure in MediaPipe's 33-point layout (full-frame pixels), head at the top."""
    lm = np.zeros((n, 33, 3), np.float32)
    lm[..., 2] = 1.0
    pts = {7: (40, 22), 8: (52, 22), 11: (34, 40), 12: (58, 40), 13: (30, 58), 14: (62, 58),
           15: (28, 74), 16: (64, 74), 17: (27, 78), 19: (27, 78), 21: (27, 78), 18: (65, 78),
           20: (65, 78), 22: (65, 78), 23: (38, 80), 24: (54, 80), 25: (37, 100), 26: (55, 100),
           27: (36, 118), 28: (56, 118), 29: (34, 122), 30: (58, 122), 31: (40, 124), 32: (52, 124)}
    for i in range(11):                                       # face points
        pts.setdefault(i, (46, 22 + i % 3))
    for i, (x, y) in pts.items():
        lm[:, i, 0], lm[:, i, 1] = x, y
    np.savez(path, frames=np.arange(n, dtype=np.float64), lm=lm)
    return path


def make_clicks_wav(path: Path, clicks=(0.5, 1.0, 1.5, 2.0), dur: float = 2.6, sr: int = 22050,
                    width: int = 2, amps=None) -> Path:
    """Silence with short decaying noise bursts (kick-like onsets) at the given times."""
    rng = np.random.default_rng(1)
    x = np.zeros(int(dur * sr), np.float64)
    k = int(0.08 * sr)
    env = np.exp(-np.arange(k) / (0.015 * sr))
    for j, t in enumerate(clicks):
        i = int(t * sr)
        x[i:i + k] += (amps[j] if amps else 0.8) * env[:len(x) - i] * rng.standard_normal(min(k, len(x) - i))
    x = np.clip(x, -1, 1)
    scale = 2 ** (8 * width - 1) - 1
    q = np.round(x * scale).astype(np.int64)
    if width == 3:                                            # 24-bit little-endian PCM
        b = q.astype("<i4").view(np.uint8).reshape(-1, 4)[:, :3].tobytes()
    else:
        b = q.astype({2: "<i2", 4: "<i4"}[width]).tobytes()      # 16 / 32-bit PCM
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(width)
        w.setframerate(sr)
        w.writeframes(b)
    return path


def load(path) -> np.ndarray:
    with Image.open(path) as im:
        return np.array(im)


def same_images(a: Path, b: Path) -> bool:
    x, y = load(a), load(b)
    return x.shape == y.shape and bool((x == y).all())
