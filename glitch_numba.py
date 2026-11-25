"""Compiled versions of three of glitch.py's hottest loops (Numba). Optional, nothing else.

glitch.py uses them when Numba is installed (and GLITCHER_NUMBA isn't 0), else its own numpy
code. Each kernel is the numpy code rewritten as one loop over the pixels: the same float32
operations in the same order, so the bytes out are identical (tests/test_numba.py checks that
against the numpy code on random and real frames). They are faster because numpy builds a
full-size temporary array for every step of a formula, while a loop keeps each pixel in
registers from start to end.

Not here on purpose: the mosh motion search sums 64 float32 differences per block, and numpy's
summation order (pairwise, chosen by its iterator) can't be reproduced exactly in a loop. A
last-bit difference could flip a near-tie between two motion vectors, so frames could differ.
It is ~1 % of a frame; it stays numpy.

Compiled code is cached on disk (first use on a machine: ~10 s, then milliseconds). The
signatures are explicit (C-contiguous arrays) so each kernel compiles once, not once per
combination of array layouts; callers pass contiguous arrays.
"""
from __future__ import annotations

import numpy as np
from numba import njit

_OPTS = dict(cache=True, error_model="numpy", nogil=True)   # no fastmath: exact IEEE float32


@njit("boolean(uint8[:, :, ::1])", **_OPTS)
def any_alpha(a):
    """Any pixel with alpha > 0 (glitch._under returns front untouched if not)."""
    for y in range(a.shape[0]):
        for x in range(a.shape[1]):
            if a[y, x, 3] > 0:
                return True
    return False


@njit("void(uint8[:, :, ::1], uint8[:, :, ::1], uint8[:, :, ::1])", **_OPTS)
def under_u8(front, ghost, out):
    """glitch._under / _under_px for 8-bit RGBA: ghost composited under front, into out."""
    one, k255, half = np.float32(1.0), np.float32(255.0), np.float32(0.5)
    eps, zero = np.float32(1e-6), np.float32(0.0)
    for y in range(front.shape[0]):
        for x in range(front.shape[1]):
            fa8, ga8 = front[y, x, 3], ghost[y, x, 3]
            if ga8 > 0 and fa8 < 255:                         # blended: _under_px, step by step
                fa = np.float32(fa8) / k255
                ga = np.float32(ga8) / k255
                oa = fa + ga * (one - fa)
                inv = one / max(oa, eps) if oa > zero else zero
                gw = ga * (one - fa)
                for c in range(3):
                    v = (np.float32(front[y, x, c]) * fa + np.float32(ghost[y, x, c]) * gw) * inv + half
                    out[y, x, c] = np.uint8(min(max(v, zero), k255))
                out[y, x, 3] = np.uint8(min(max(oa * k255 + half, zero), k255))
            elif fa8 == 0 and ga8 == 0:                       # nothing visible: colour cleared
                for c in range(4):
                    out[y, x, c] = 0
            else:                                             # front wins (or no ghost here)
                for c in range(4):
                    out[y, x, c] = front[y, x, c]


@njit("void(uint8[:, :, ::1], float64[:, ::1], uint8[:, :, ::1])", **_OPTS)
def noise_u8(a, n, out):
    """fx_noise after its random numbers: grain added where visible (n as float32, like numpy)."""
    zero, k255 = np.float32(0.0), np.float32(255.0)
    for y in range(a.shape[0]):
        for x in range(a.shape[1]):
            if a[y, x, 3] > 0:
                v = np.float32(n[y, x])
                for c in range(3):
                    out[y, x, c] = np.uint8(min(max(np.float32(a[y, x, c]) + v, zero), k255))
                out[y, x, 3] = a[y, x, 3]
            else:
                for c in range(4):
                    out[y, x, c] = a[y, x, c]


_REPLAY = "void({t}[:, :, ::1], int32[:, ::1], int32[:, ::1], int64, int64, int64, int64, int64, " \
          "float64, int64, int64, {t}[:, :, ::1])"


@njit([_REPLAY.format(t="uint8"), _REPLAY.format(t="float32")], **_OPTS)
def replay(base, vy, vx, bs, cy0, cy1, cx0, cx1, gain, h, w, tmp):
    """One step of fx_mosh's replay: base[cy0:cy1, cx0:cx1] = base[y + dy, x + dx], with the
    block motion (vy, vx) of block size bs scaled by gain - gathered into tmp first, like
    numpy's fancy indexing reads everything before it writes."""
    for y in range(cy0, cy1):
        iy = min(max(y // bs, 0), vy.shape[0] - 1)
        for x in range(cx0, cx1):
            ix = min(max(x // bs, 0), vy.shape[1] - 1)
            sy = min(max(y + np.int64(np.round(vy[iy, ix] * gain)), 0), h - 1)
            sx = min(max(x + np.int64(np.round(vx[iy, ix] * gain)), 0), w - 1)
            for c in range(base.shape[2]):
                tmp[y - cy0, x - cx0, c] = base[sy, sx, c]
    for y in range(cy0, cy1):
        for x in range(cx0, cx1):
            for c in range(base.shape[2]):
                base[y, x, c] = tmp[y - cy0, x - cx0, c]
