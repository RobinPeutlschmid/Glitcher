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


# a standing person in MediaPipe's layout, in units of body size from the top of the head
_FIGURE = {0: (0.0, 0.3), 1: (-0.05, 0.26), 2: (-0.1, 0.25), 3: (-0.15, 0.26), 4: (0.05, 0.26),
           5: (0.1, 0.25), 6: (0.15, 0.26), 7: (-0.25, 0.3), 8: (0.25, 0.3), 9: (-0.05, 0.4),
           10: (0.05, 0.4), 11: (-0.5, 0.8), 12: (0.5, 0.8), 13: (-0.6, 1.4), 14: (0.6, 1.4),
           15: (-0.65, 1.9), 16: (0.65, 1.9), 17: (-0.7, 2.0), 18: (0.7, 2.0), 19: (-0.65, 2.05),
           20: (0.65, 2.05), 21: (-0.6, 2.0), 22: (0.6, 2.0), 23: (-0.3, 2.0), 24: (0.3, 2.0),
           25: (-0.3, 2.9), 26: (0.3, 2.9), 27: (-0.3, 3.8), 28: (0.3, 3.8), 29: (-0.35, 3.9),
           30: (0.35, 3.9), 31: (-0.2, 3.95), 32: (0.2, 3.95)}


def figure_landmarks(i: int, w: int, h: int) -> np.ndarray:
    """(33, 3) landmarks of the test figure at frame i: it walks right a little and swings its arms."""
    s = h / 4.4
    cx, top = w * 0.42 + i * w / 300, h * 0.04
    lm = np.ones((33, 3), np.float32)
    sw = 0.25 * math.sin(i * 0.25)
    for k, (x, y) in _FIGURE.items():
        if k in (13, 15, 17, 19, 21):
            x -= sw * (y - 0.8)
        if k in (14, 16, 18, 20, 22):
            x += sw * (y - 0.8)
        lm[k, 0], lm[k, 1] = cx + x * s, top + y * s
    return lm


def make_figure(folder: Path, n: int = 24, w: int = 240, h: int = 320) -> tuple[list[Path], Path, Path]:
    """A humanoid cut-out whose alpha is drawn from its own landmarks (so body-part masks fit it
    exactly), its landmarks as .npz and .csv. -> (frame paths, npz, csv)."""
    import glitch
    from estimate_pose import write_landmarks_csv
    folder.mkdir(parents=True, exist_ok=True)
    yy, xx = np.mgrid[:h, :w].astype(np.float32)
    out, LM = [], []
    for i in range(n):
        lm = figure_landmarks(i, w, h)
        a = glitch.part_mask(lm, glitch.PARTS, (h, w), feather=0.4)
        rgb = np.stack([120 + 100 * (yy / h), 100 + 70 * np.sin(xx / 6.0 + i * 0.2), 80 + 90 * (xx / w)], -1)
        img = np.dstack([np.clip(rgb, 0, 255), a * 255]).astype(np.uint8)
        p = folder / f"fig_{i + 1:04d}.png"
        Image.fromarray(img, "RGBA").save(p)
        out.append(p)
        LM.append(lm)
    npz = folder / "fig_landmarks.npz"
    np.savez(npz, frames=np.arange(n, dtype=np.int64), lm=np.asarray(LM, np.float32), size=np.array([w, h]))
    csv = folder / "fig_landmarks.csv"
    write_landmarks_csv(csv, np.arange(n), np.asarray(LM), (w, h))
    return out, npz, csv


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
