#!/usr/bin/env python3
"""Would a GPU (CuPy) or compiled CPU code (Numba) make Glitcher faster on *this* machine, and
would the frames stay identical? Measures instead of guessing. Glitcher itself never uses this.

    pip install "cupy-cuda12x[ctk]" numba      # both optional: a missing one is skipped
    python tools/gpu_bench.py                  # synthetic 1920x1080 cut-out
    python tools/gpu_bench.py FRAMES_DIR --preset my_look.json    # your footage and look

Prints a summary and writes gpu_bench_result.json (send that file back). Takes a few minutes.

What it does
  1. Profiles the real render (glitch.py, one process, like one render worker) frame by frame:
     milliseconds per effect, PNG load/save, history loading - exclusive times, so an effect's
     time doesn't include the history frames it loaded.
  2. Benchmarks the hot spots - echo's compositing (_under), the JPEG DCT, the mosh motion
     search and replay, the noise - as the numpy code in glitch.py (reference), as a CuPy port
     and as a Numba port. Every port first runs on numpy and must equal glitch.py bit for bit
     (so it measures the same work); then each variant is compared with the reference: exact,
     or how many pixels differ and by how much.
  3. GPU costs: CUDA start-up, VRAM per process, upload/download of a frame, first-call
     compile times, and how throughput holds up when several render workers share the GPU.
  4. Estimates the frame time with each backend from 1-3, and the ceiling (those effects free).
Close DaVinci Resolve for the clean numbers; run it again with Resolve open to see the sharing.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from statistics import median
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import numpy as np  # noqa: E402

import glitch  # noqa: E402

perf = time.perf_counter


# --------------------------------------------------------------------------- helpers
def timed(fn, repeat: int, sync=None):
    """(stats, result): first call (includes any compile) and the median of `repeat` more."""
    sync = sync or (lambda: None)
    t0 = perf()
    r = fn()
    sync()
    first = (perf() - t0) * 1e3
    ts = []
    for _ in range(repeat):
        t0 = perf()
        r = fn()
        sync()
        ts.append((perf() - t0) * 1e3)
    return {"first_ms": round(first, 3), "ms": round(median(ts), 3), "min_ms": round(min(ts), 3)}, r


def compare(ref, got) -> dict:
    """How far `got` is from the reference: exact, or the share of pixels that differ."""
    got = np.asarray(got)
    if got.shape != ref.shape:
        return {"exact": False, "error": f"shape {got.shape} != {ref.shape}"}
    d = np.abs(ref.astype(np.float64) - got.astype(np.float64))
    per = d.reshape(-1, ref.shape[-1]).max(1) if ref.ndim >= 2 else d.ravel()
    res = {"exact": bool(d.max() == 0) and got.dtype == ref.dtype, "max_abs": float(d.max()),
           "differing": round(float((per > 0).mean()), 6)}
    if got.dtype != ref.dtype:
        res["dtype"] = f"{got.dtype} (reference {ref.dtype})"
    return res


def cli_size(s: str):
    w, h = s.lower().split("x")
    return int(w), int(h)


def nvidia_smi() -> dict | None:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    try:
        out = subprocess.run([exe, "--query-gpu=name,driver_version,memory.total,memory.used",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True,
                             timeout=20, stdin=subprocess.DEVNULL).stdout.strip().splitlines()[0]
        name, drv, tot, used = [x.strip() for x in out.split(",")]
        return {"name": name, "driver": drv, "total_mb": float(tot), "used_mb": float(used)}
    except Exception:                                   # noqa: BLE001 - info only
        return None


def load_cupy():
    """(cupy, None) if a GPU is usable, else (None, why)."""
    try:
        import cupy as cp
    except Exception as e:                              # noqa: BLE001
        return None, f"CuPy not installed ({type(e).__name__}: {e})"
    try:
        cp.cuda.runtime.getDeviceCount()
        cp.zeros(4).sum().item()
    except Exception as e:                              # noqa: BLE001
        return None, f"CuPy is installed but no usable CUDA GPU ({type(e).__name__}: {e})"
    return cp, None


def load_numba():
    try:
        import numba
        return numba, None
    except Exception as e:                              # noqa: BLE001
        return None, f"Numba not installed ({type(e).__name__}: {e})"


# --------------------------------------------------------------------------- 1. pipeline profile
class Profiler:
    """Exclusive time per label: a wrapped call's time minus the wrapped calls inside it."""

    def __init__(self):
        self.stack, self.ms, self.calls = [], {}, {}

    def wrap(self, fn, label):
        def w(*a, **k):
            lab = label(self) if callable(label) else label
            self.stack.append([lab, perf(), 0.0])
            try:
                return fn(*a, **k)
            finally:
                lab, t0, child = self.stack.pop()
                el = perf() - t0
                self.ms[lab] = self.ms.get(lab, 0.0) + (el - child) * 1e3
                self.calls[lab] = self.calls.get(lab, 0) + 1
                if self.stack:
                    self.stack[-1][2] += el
        return w


def glitch_argv(frames_dir: Path, out: Path, preset: str | None, rest: list[str], tmp: Path):
    if preset:
        import glitch_gui                                # only for the preset -> options helper
        s = glitch_gui.load_preset_file(preset)
        s["input"] = str(frames_dir)
        argv = glitch_gui.preset_argv(s, with_output=False, curve_tmp=tmp / "curves.csv")
        return [str(frames_dir), "-o", str(out), *argv[1:], *rest]
    return [str(frames_dir), "-o", str(out), "-s", "7", *rest]


def profile_pipeline(frames_dir: Path, n: int, preset, rest, tmp: Path) -> dict:
    out = tmp / "render"
    args = glitch.build_parser().parse_args(glitch_argv(frames_dir, out, preset, rest, tmp))
    work = glitch.plan_render(args)["work"][:n]
    P = Profiler()
    in_hist = lambda p: "history load" if p.stack and p.stack[-1][0] == "history load" else "load frame"
    # (owner, attribute, label). The kernel labels (in KERNEL_PARTS) split the hot effects into
    # the part a backend would replace and the rest, which stays under the effect's own name.
    patches = [(glitch, "load_frame", in_hist), (glitch, "save_frame", "save frame"),
               (glitch, "glitch_pass", "glitch pass"), (glitch, "save_pass", "save frame"),
               (glitch.SeqHistory, "__call__", "history load"), (glitch, "_under", "under (blend)"),
               (glitch, "_block_motion", "motion search"), (glitch, "_feat", "motion features")]
    patches += [(glitch, f, "jpeg dct") for f in ("_dct", "_idct", "_rgb2ycc", "_ycc2rgb")]
    orig = [(o, at, getattr(o, at)) for o, at, _ in patches]
    fx = dict(glitch._FX)
    for (o, at, lab), (_, _, f) in zip(patches, orig):
        setattr(o, at, P.wrap(f, lab))
    for k, f in fx.items():
        glitch._FX[k] = P.wrap(f, k)
    render = P.wrap(glitch._render, "other")
    per_frame, snaps = [], []
    try:
        for job in work:
            t0 = perf()
            render(job)
            per_frame.append((perf() - t0) * 1e3)
            snaps.append(dict(P.ms))
    finally:
        for o, at, f in orig:
            setattr(o, at, f)
        glitch._FX.update(fx)
    warm = 2 if len(work) > 4 else 0                     # first frames: imports, empty caches
    nf = len(work) - warm
    base = snaps[warm - 1] if warm else {}
    parts = {k: round((v - base.get(k, 0.0)) / nf, 2) for k, v in P.ms.items()}
    parts = dict(sorted(parts.items(), key=lambda kv: -kv[1]))
    first = glitch.load_frame(work[0][0], work[0][7])[0]
    shape = first.shape if isinstance(first, np.ndarray) else (first.height, first.width, 4)
    return {"frames": len(work), "warmup_frames": warm, "size": [shape[1], shape[0]],
            "argv": glitch_argv(frames_dir, out, preset, rest, tmp)[3:],
            "frame_ms": {"mean": round(sum(per_frame[warm:]) / nf, 1), "median": round(median(per_frame[warm:]), 1), "min": round(min(per_frame[warm:]), 1),
                         "max": round(max(per_frame[warm:]), 1)},
            "ms_per_frame": parts,
            "history_loads_per_frame": round(P.calls.get("history load", 0) / len(work), 2),
            "render_code": glitch.FAST_STATUS}


# --------------------------------------------------------------------------- 2. ports (xp = numpy or cupy)
# Each mirrors the numpy code in glitch.py line for line (same operations, same order, same
# dtypes), so with xp=numpy it is bit-identical to glitch.py - checked before anything is timed.
def under_px_xp(xp, f, g):                               # glitch._under_px
    fa = f[..., 3].astype(xp.float32) / 255.0
    ga = g[..., 3].astype(xp.float32) / 255.0
    oa = fa + ga * (1 - fa)
    inv = xp.where(oa > 0, 1.0 / xp.maximum(oa, 1e-6), 0.0)
    rgb = (f[..., :3] * fa[..., None] + g[..., :3] * (ga * (1 - fa))[..., None]) * inv[..., None]
    out = xp.empty_like(f)
    out[..., :3] = xp.clip(rgb + 0.5, 0, 255).astype(xp.uint8)
    out[..., 3] = xp.clip(oa * 255 + 0.5, 0, 255).astype(xp.uint8)
    return out


def under_xp(xp, front, ghost):
    """glitch._under for uint8 as whole-array operations (no data-dependent boxes or boolean
    indexing - those force a GPU->CPU round trip each)."""
    gm = ghost[..., 3] > 0
    fa = front[..., 3]
    m = gm & (fa < 255)
    clear = ((fa == 0) & ~gm)[..., None] & (xp.arange(4) < 3)
    out = xp.where(clear, xp.uint8(0), front)
    return xp.where(m[..., None], under_px_xp(xp, front, ghost), out)


UNDER_CU = r"""
extern "C" __global__ void under_u8(const unsigned char* f, const unsigned char* g,
                                    unsigned char* o, const long long n) {
    long long i = (long long)blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n) return;
    const unsigned char* F = f + 4 * i;
    const unsigned char* G = g + 4 * i;
    unsigned char* O = o + 4 * i;
    const unsigned char fa8 = F[3], ga8 = G[3];
    if (ga8 > 0 && fa8 < 255) {                 /* same float32 operations, same order */
        const float fa = (float)fa8 / 255.0f, ga = (float)ga8 / 255.0f;
        const float oa = fa + ga * (1.0f - fa);
        const float inv = oa > 0.0f ? 1.0f / fmaxf(oa, 1e-6f) : 0.0f;
        const float gw = ga * (1.0f - fa);
        for (int c = 0; c < 3; ++c) {
            float v = ((float)F[c] * fa + (float)G[c] * gw) * inv + 0.5f;
            O[c] = (unsigned char)fminf(fmaxf(v, 0.0f), 255.0f);
        }
        O[3] = (unsigned char)fminf(fmaxf(oa * 255.0f + 0.5f, 0.0f), 255.0f);
    } else if (fa8 == 0 && ga8 == 0) {
        O[0] = 0; O[1] = 0; O[2] = 0; O[3] = 0;
    } else {
        O[0] = F[0]; O[1] = F[1]; O[2] = F[2]; O[3] = F[3];
    }
}
"""


def ycc_xp(xp, x):                                       # glitch._rgb2ycc
    r, g, b = x[..., 0], x[..., 1], x[..., 2]
    return xp.stack([0.299 * r + 0.587 * g + 0.114 * b - 128,
                     -0.168736 * r - 0.331264 * g + 0.5 * b,
                     0.5 * r - 0.418688 * g - 0.081312 * b], -1)


def rgb_xp(xp, x):                                       # glitch._ycc2rgb
    y, cb, cr = x[..., 0] + 128, x[..., 1], x[..., 2]
    return xp.stack([y + 1.402 * cr, y - 0.344136 * cb - 0.714136 * cr, y + 1.772 * cb], -1)


def jpeg_core_xp(xp, src, C, Q, add):
    """fx_jpeg's per-run work for 'AC garbage' (the heaviest kind): colour transform, DCT, noise,
    quantise, inverse, back to 8 bit. src (n,8,8,3) float32 blocks."""
    co = C @ xp.moveaxis(ycc_xp(xp, src), -1, 1) @ C.T
    co += add
    co = xp.round(co / Q) * Q
    rgb = rgb_xp(xp, xp.moveaxis(C.T @ co @ C, 1, -1))
    return xp.rint(xp.clip(rgb, 0, 255)).astype(xp.uint8)


def jpeg_core_ref(src, Q, add):
    co = glitch._dct(np.moveaxis(glitch._rgb2ycc(src), -1, 1))
    co += add
    co = np.round(co / Q) * Q
    rgb = glitch._ycc2rgb(np.moveaxis(glitch._idct(co), 1, -1))
    return np.rint(np.clip(rgb, 0, 255)).astype(np.uint8)


def search_xp(xp, fp, fn):
    """glitch._block_motion after the feature images: exhaustive block search. -> (vy, vx)
    in low-res pixels."""
    MB, R = glitch._MB, glitch._MR
    by, bx = fn.shape[0] // MB, fn.shape[1] // MB
    fn = fn[:by * MB, :bx * MB]
    vy = xp.zeros((by, bx), xp.int32)
    vx = xp.zeros((by, bx), xp.int32)
    ys, xs = xp.nonzero((fp[:by * MB, :bx * MB] > 0) | (fn > 0))
    if ys.size == 0:
        return vy, vx
    b0, b1 = max(0, (int(ys.min()) - R) // MB), min(by, (int(ys.max()) + R) // MB + 1)
    c0, c1 = max(0, (int(xs.min()) - R) // MB), min(bx, (int(xs.max()) + R) // MB + 1)
    nb, nc = b1 - b0, c1 - c0
    pp = xp.pad(fp, R, mode="edge")
    fw = fn[b0 * MB:b1 * MB, c0 * MB:c1 * MB]
    best = xp.full((nb, nc), xp.inf, xp.float32)
    wy = xp.zeros((nb, nc), xp.int32)
    wx = xp.zeros((nb, nc), xp.int32)
    for dy in range(-R, R + 1):
        for dx in range(-R, R + 1):
            sh = pp[R + dy + b0 * MB:R + dy + b1 * MB, R + dx + c0 * MB:R + dx + c1 * MB]
            sad = xp.abs(fw - sh).reshape(nb, MB, nc, MB).sum(axis=(1, 3)) + 0.5 * MB * MB * (abs(dy) + abs(dx))
            m = sad < best
            best = xp.where(m, sad, best)
            wy = xp.where(m, xp.int32(dy), wy)
            wx = xp.where(m, xp.int32(dx), wx)
    vy[b0:b1, c0:c1], vx[b0:b1, c0:c1] = wy, wx
    return vy, vx


def replay_xp(xp, base, fields, box, gain, h, w):
    """fx_mosh's replay: drag the held frame along each later frame's block motion."""
    cy0, cy1, cx0, cx1 = box
    Y, X = xp.arange(cy0, cy1), xp.arange(cx0, cx1)
    for vy, vx, bs in fields:
        iy = xp.clip(Y // bs, 0, vy.shape[0] - 1)
        ix = xp.clip(X // bs, 0, vy.shape[1] - 1)
        dy = xp.round(vy[iy][:, ix] * gain).astype(xp.int64)
        dx = xp.round(vx[iy][:, ix] * gain).astype(xp.int64)
        sy = xp.clip(Y[:, None] + dy, 0, h - 1)
        sx = xp.clip(X[None, :] + dx, 0, w - 1)
        base[cy0:cy1, cx0:cx1] = base[sy, sx]
    return base


def noise_add_xp(xp, a, n32):
    """fx_noise after the random numbers: add the grain where visible."""
    vis = (a[..., 3] > 0)[..., None] & (xp.arange(4) < 3)
    res = xp.clip(a.astype(xp.float32) + n32[..., None], 0, 255).astype(xp.uint8)
    return xp.where(vis, res, a)


# --------------------------------------------------------------------------- Numba ports
def _under_nb(front, ghost, out):
    h, w = front.shape[0], front.shape[1]
    one, k255, half = np.float32(1.0), np.float32(255.0), np.float32(0.5)
    eps, zero = np.float32(1e-6), np.float32(0.0)
    for y in _prange(h):
        for x in range(w):
            fa8, ga8 = front[y, x, 3], ghost[y, x, 3]
            if ga8 > 0 and fa8 < 255:
                fa = np.float32(fa8) / k255
                ga = np.float32(ga8) / k255
                oa = fa + ga * (one - fa)
                inv = one / max(oa, eps) if oa > zero else zero
                gw = ga * (one - fa)
                for c in range(3):
                    v = (np.float32(front[y, x, c]) * fa + np.float32(ghost[y, x, c]) * gw) * inv + half
                    out[y, x, c] = np.uint8(min(max(v, zero), k255))
                out[y, x, 3] = np.uint8(min(max(oa * k255 + half, zero), k255))
            elif fa8 == 0 and ga8 == 0:
                for c in range(4):
                    out[y, x, c] = 0
            else:
                for c in range(4):
                    out[y, x, c] = front[y, x, c]
    return out


def _search_nb(pp, fw, nb, nc, oy, ox, MB, R, wy, wx):
    for i in _prange(nb):
        for j in range(nc):
            best = np.float32(np.inf)
            by_, bx_ = 0, 0
            for dy in range(-R, R + 1):
                for dx in range(-R, R + 1):
                    s = np.float32(0.0)
                    for u in range(MB):
                        for v in range(MB):
                            s += abs(fw[i * MB + u, j * MB + v] - pp[oy + dy + i * MB + u, ox + dx + j * MB + v])
                    s += np.float32(0.5 * MB * MB * (abs(dy) + abs(dx)))
                    if s < best:
                        best, by_, bx_ = s, dy, dx
            wy[i, j], wx[i, j] = by_, bx_


def _noise_nb(a, n32, out):
    h, w = a.shape[0], a.shape[1]
    zero, k255 = np.float32(0.0), np.float32(255.0)
    for y in _prange(h):
        for x in range(w):
            if a[y, x, 3] > 0:
                v = n32[y, x]
                for c in range(3):
                    out[y, x, c] = np.uint8(min(max(np.float32(a[y, x, c]) + v, zero), k255))
                out[y, x, 3] = a[y, x, 3]
            else:
                for c in range(4):
                    out[y, x, c] = a[y, x, c]
    return out


def _replay_nb(base, vy, vx, bs, cy0, cy1, cx0, cx1, gain, h, w, tmp):
    for y in _prange(cy0, cy1):
        iy = min(max(y // bs, 0), vy.shape[0] - 1)
        for x in range(cx0, cx1):
            ix = min(max(x // bs, 0), vy.shape[1] - 1)
            sy = min(max(y + np.int64(np.round(vy[iy, ix] * gain)), 0), h - 1)
            sx = min(max(x + np.int64(np.round(vx[iy, ix] * gain)), 0), w - 1)
            for c in range(base.shape[2]):
                tmp[y - cy0, x - cx0, c] = base[sy, sx, c]
    base[cy0:cy1, cx0:cx1] = tmp                         # gather first, then write (like numpy)


def _prange(*a):                                         # replaced by numba.prange when compiled
    return range(*a)


def numba_ports(numba, parallel: bool):
    """Compile the Numba ports. parallel=False is the fair one for full renders (the render
    already runs one worker process per core); parallel=True is what a single preview gets."""
    g = dict(_under_nb.__globals__, _prange=numba.prange)
    mk = lambda f: numba.njit(parallel=parallel, error_model="numpy", cache=False)(
        type(f)(f.__code__, g, f.__name__))
    under, search, noise, replay = mk(_under_nb), mk(_search_nb), mk(_noise_nb), mk(_replay_nb)

    def replay_nb(base, fields, box, gain, h, w):
        cy0, cy1, cx0, cx1 = box
        tmp = np.empty((cy1 - cy0, cx1 - cx0, base.shape[2]), base.dtype)
        for vy, vx, bs in fields:
            replay(base, vy, vx, bs, cy0, cy1, cx0, cx1, float(gain), h, w, tmp)
        return base

    def search_nb(fp, fn):
        MB, R = glitch._MB, glitch._MR
        by, bx = fn.shape[0] // MB, fn.shape[1] // MB
        fn = fn[:by * MB, :bx * MB]
        vy = np.zeros((by, bx), np.int32)
        vx = np.zeros((by, bx), np.int32)
        ys, xs = np.nonzero((fp[:by * MB, :bx * MB] > 0) | (fn > 0))
        if ys.size == 0:
            return vy, vx
        b0, b1 = max(0, (ys.min() - R) // MB), min(by, (ys.max() + R) // MB + 1)
        c0, c1 = max(0, (xs.min() - R) // MB), min(bx, (xs.max() + R) // MB + 1)
        nb, nc = b1 - b0, c1 - c0
        pp = np.pad(fp, R, mode="edge")
        fw = np.ascontiguousarray(fn[b0 * MB:b1 * MB, c0 * MB:c1 * MB])
        wy = np.zeros((nb, nc), np.int32)
        wx = np.zeros((nb, nc), np.int32)
        search(pp, fw, nb, nc, R + b0 * MB, R + c0 * MB, MB, R, wy, wx)
        vy[b0:b1, c0:c1], vx[b0:b1, c0:c1] = wy, wx
        return vy, vx

    return SimpleNamespace(under=lambda f, gh: under(f, gh, np.empty_like(f)), search=search_nb,
                           noise=lambda a, n32: noise(a, n32, np.empty_like(a)), replay=replay_nb)


# --------------------------------------------------------------------------- 2. material
def bench_material(paths, frames_needed: int = 10):
    """Realistic inputs for the kernels, cut from real frames the way the effects cut them."""
    imgs = []
    for p in paths[:frames_needed]:
        im, _ = glitch.load_frame(p)
        a = np.asarray(im.convert("RGBA")) if not isinstance(im, np.ndarray) else \
            np.asarray(glitch.to_display8(im))
        imgs.append(np.ascontiguousarray(a))
    cur, hist = imgs[-1], imgs[-2::-1]                   # hist[0] = 1 back
    h, w = cur.shape[:2]
    y0, y1, x0, x1 = glitch._content_box(cur)
    bw, bh = x1 - x0, y1 - y0
    # echo: a band over the whole subject box widened by half a box width each side
    ya, yb = max(0, y0 - bh // 12), min(h, y1 + bh // 12)
    xa, xb = max(0, x0 - bw // 2), min(w, x1 + bw // 2)
    front = np.ascontiguousarray(cur[ya:yb, xa:xb])
    ghost = glitch._scaled_alpha(glitch._offset_window(hist[1], ya, yb, xa, xb, 3, int(0.04 * bw)), 0.6)
    # jpeg: one run of block rows across the subject (fx_jpeg's typical run ~0.15 box heights)
    gh_, gw_ = h // 8, w // 8
    vis = cur[..., 3] > 0
    base8 = cur[:gh_ * 8, :gw_ * 8, :3].astype(np.float32)
    np.copyto(base8, cur[..., :3][vis].mean(0).astype(np.float32), where=~vis[:gh_ * 8, :gw_ * 8, None])
    blocks = base8.reshape(gh_, 8, gw_, 8, 3).transpose(0, 2, 1, 3, 4)
    r0 = (y0 + bh // 3) // 8
    rows = max(1, int(0.15 * bh / 8))
    idx = np.arange(r0 * gw_, min(gh_, r0 + rows) * gw_)
    by_, bx_ = idx // gw_, idx % gw_
    src = np.ascontiguousarray(blocks[by_, bx_])
    tile = glitch._ac_tile((7, glitch.FX_ID["jpeg"], 3, 1))
    add = (tile[by_ % 8, bx_ % 8] * glitch._FALL * (160 * 0.7)).astype(np.float32)
    Q = glitch._qtabs(45)
    # mosh: motion between consecutive history frames, replayed over a band
    s = max(1, int(round(max(h, w) / 384)))
    lag = min(8, len(hist) - 1)
    return SimpleNamespace(cur=cur, hist=hist, front=front, ghost=ghost, src=src, add=add, Q=Q,
                           s=s, lag=lag, size=(w, h), n_blocks=int(len(idx)),
                           band=(ya, yb, xa, xb), mosh_box=(max(0, ya), min(h, yb), max(0, xa), min(w, xb)))


# --------------------------------------------------------------------------- 2. kernel benchmarks
def bench_kernels(M, repeat: int, cp, numba) -> dict:
    R = {}
    note = lambda k, **v: R.setdefault(k, {}).update(v)
    sync = (lambda: cp.cuda.Device().synchronize()) if cp is not None else None

    # ---- echo: _under -------------------------------------------------------------
    st, ref = timed(lambda: glitch._under(M.front, M.ghost), repeat)
    note("under", px=int(M.front.shape[0] * M.front.shape[1]), cpu_numpy=st)
    port = under_xp(np, M.front, M.ghost)
    note("under", port_check=compare(ref, port))
    if numba is not None:
        for par in (False, True):
            nb = numba_ports(numba, par)
            st, got = timed(lambda: nb.under(M.front, M.ghost), repeat)
            note("under", **{f"numba_{'parallel' if par else 'serial'}": {**st, **compare(ref, got)}})
    if cp is not None:
        f, g = cp.asarray(M.front), cp.asarray(M.ghost)
        st, got = timed(lambda: under_xp(cp, f, g), repeat, sync)
        note("under", cupy_port={**st, **compare(ref, cp.asnumpy(got))})
        for name, opts in (("cuda_strict", ("--fmad=false",)), ("cuda_fast", ("--use_fast_math",))):
            t0 = perf()
            k = cp.RawKernel(UNDER_CU, "under_u8", options=opts)
            o = cp.empty_like(f)
            n = f.shape[0] * f.shape[1]
            run = lambda: k(((n + 255) // 256,), (256,), (f, g, o, np.int64(n)))
            run()
            sync()
            comp = (perf() - t0) * 1e3
            st, _ = timed(run, repeat, sync)
            note("under", **{name: {**st, "compile_ms": round(comp, 1), **compare(ref, cp.asnumpy(o))}})
        k = cp.RawKernel(UNDER_CU, "under_u8", options=("--fmad=false",))

        def strict_io():
            fg, gg = cp.asarray(M.front), cp.asarray(M.ghost)
            og = cp.empty_like(fg)
            n = fg.shape[0] * fg.shape[1]
            k(((n + 255) // 256,), (256,), (fg, gg, og, np.int64(n)))
            return cp.asnumpy(og)
        st, _ = timed(strict_io, repeat, sync)
        note("under", cuda_strict_with_transfer=st)

    # ---- jpeg core ----------------------------------------------------------------
    st, ref = timed(lambda: jpeg_core_ref(M.src, M.Q, M.add), repeat)
    note("jpeg", blocks=M.n_blocks, cpu_numpy=st)
    note("jpeg", port_check=compare(ref, jpeg_core_xp(np, M.src, glitch._C, M.Q, M.add)))
    if cp is not None:
        s_, C_, Q_, a_ = cp.asarray(M.src), cp.asarray(glitch._C), cp.asarray(M.Q), cp.asarray(M.add)
        st, got = timed(lambda: jpeg_core_xp(cp, s_, C_, Q_, a_), repeat, sync)
        note("jpeg", cupy_port={**st, **compare(ref, cp.asnumpy(got))})
        st, _ = timed(lambda: cp.asnumpy(jpeg_core_xp(cp, cp.asarray(M.src), C_, Q_, cp.asarray(M.add))),
                      repeat, sync)
        note("jpeg", cupy_port_with_transfer=st)

    # ---- mosh: motion search + replay -----------------------------------------------
    prev, nxt = M.hist[1], M.hist[0]
    st_feat, (fp, fn) = timed(lambda: (glitch._feat(prev, M.s), glitch._feat(nxt, M.s)), repeat)
    st, (rvy, rvx, bs) = timed(lambda: glitch._block_motion(prev, nxt, M.s), repeat)
    ref_v = np.stack([rvy, rvx], -1)
    note("mosh_search", scale=M.s, feat_px=list(fp.shape), cpu_numpy=st, cpu_features_only=st_feat)
    pv = search_xp(np, fp, fn)
    note("mosh_search", port_check=compare(ref_v, np.stack([pv[0] * M.s, pv[1] * M.s], -1)))
    if numba is not None:
        for par in (False, True):
            nb = numba_ports(numba, par)
            st2, got = timed(lambda: nb.search(fp, fn), repeat)
            note("mosh_search", **{f"numba_{'parallel' if par else 'serial'}":
                                   {**st2, **compare(ref_v, np.stack([got[0] * M.s, got[1] * M.s], -1))}})
    if cp is not None:
        fpg, fng = cp.asarray(fp), cp.asarray(fn)
        st2, got = timed(lambda: search_xp(cp, fpg, fng), repeat, sync)
        note("mosh_search", cupy_port={**st2, **compare(ref_v, np.stack([cp.asnumpy(got[0]) * M.s,
                                                                         cp.asnumpy(got[1]) * M.s], -1))})
    h, w = M.size[1], M.size[0]
    fields = []
    for k in range(M.lag - 1, -1, -1):                    # older -> newer, like fx_mosh
        vy, vx, bs = glitch._block_motion(M.hist[k + 1], M.hist[k], M.s)
        fields.append((vy, vx, bs))
    held = M.hist[M.lag]
    G = 1.375                                            # a 'bloom' gain: exercises the rounding
    st, ref = timed(lambda: replay_xp(np, held.copy(), fields, M.mosh_box, G, h, w), repeat)
    note("mosh_replay", lag=M.lag, cpu_numpy=st)
    if numba is not None:
        for par in (False, True):
            nb = numba_ports(numba, par)
            st, got = timed(lambda: nb.replay(held.copy(), fields, M.mosh_box, G, h, w), repeat)
            note("mosh_replay", **{f"numba_{'parallel' if par else 'serial'}": {**st, **compare(ref, got)}})
    if cp is not None:
        fg = [(cp.asarray(a), cp.asarray(b), c) for a, b, c in fields]
        hg = cp.asarray(held)
        st, got = timed(lambda: replay_xp(cp, hg.copy(), fg, M.mosh_box, G, h, w), repeat, sync)
        note("mosh_replay", cupy_port={**st, **compare(ref, cp.asnumpy(got))})
        st, _ = timed(lambda: cp.asnumpy(replay_xp(cp, cp.asarray(held), fg, M.mosh_box, G, h, w)), repeat, sync)
        note("mosh_replay", cupy_port_with_transfer=st)

    # ---- noise ----------------------------------------------------------------------
    c = SimpleNamespace(gain=lambda fx: 1.0, p=SimpleNamespace(seed=7, intensity=0.5), fi=11)
    st, ref = timed(lambda: glitch.fx_noise(M.cur, c), repeat)
    sigma = 3 + 10 * 0.5
    rng = lambda: np.random.default_rng((7, glitch.FX_ID["noise"], 0, 11))
    st_rng, n64 = timed(lambda: rng().normal(0, sigma, size=M.cur.shape[:2]), repeat)
    st_rng32, _ = timed(lambda: rng().standard_normal(M.cur.shape[:2], dtype=np.float32) * np.float32(sigma), repeat)
    n32 = n64.astype(np.float32)
    note("noise", cpu_numpy=st, cpu_random_numbers_only=st_rng, cpu_float32_random_numbers=st_rng32,
         port_check=compare(ref, noise_add_xp(np, M.cur, n32)))
    if numba is not None:
        for par in (False, True):
            nb = numba_ports(numba, par)
            st, got = timed(lambda: nb.noise(M.cur, n32), repeat)
            note("noise", **{f"numba_{'parallel' if par else 'serial'}_add": {**st, **compare(ref, got)}})
    if cp is not None:
        ag = cp.asarray(M.cur)
        st, got = timed(lambda: noise_add_xp(cp, ag, cp.asarray(n64.astype(np.float32))), repeat, sync)
        note("noise", cupy_cpu_numbers={**st, **compare(ref, cp.asnumpy(got))})
        st, _ = timed(lambda: cp.asnumpy(noise_add_xp(cp, cp.asarray(M.cur), cp.asarray(n64.astype(np.float32)))),
                      repeat, sync)
        note("noise", cupy_cpu_numbers_with_transfer=st)
        gen = cp.random.RandomState(7)
        st, _ = timed(lambda: noise_add_xp(cp, ag, gen.normal(0, sigma, M.cur.shape[:2], dtype=cp.float32)),
                      repeat, sync)
        note("noise", cupy_gpu_numbers={**st, "note": "different grain values: never matches the CPU"})
    return R


def bench_transfers(M, repeat, cp) -> dict:
    sync = lambda: cp.cuda.Device().synchronize()
    out = {}
    for name, arr in (("uint8_frame", M.cur), ("float32_frame", M.cur.astype(np.float32))):
        d = cp.asarray(arr)
        st_up, _ = timed(lambda: cp.asarray(arr), repeat, sync)
        st_dn, _ = timed(lambda: cp.asnumpy(d), repeat, sync)
        import cupyx
        pin = cupyx.empty_pinned(arr.shape, arr.dtype)
        pin[...] = arr
        st_pin, _ = timed(lambda: d.set(pin), repeat, sync)
        out[name] = {"mb": round(arr.nbytes / 1e6, 1), "upload": st_up, "download": st_dn,
                     "upload_pinned": st_pin,
                     "upload_gb_s": round(arr.nbytes / 1e6 / max(st_up["ms"], 1e-6), 2)}
    return out


# --------------------------------------------------------------------------- 3. several workers
def _worker(kind, npz, iters, barrier, q):
    """One fake render worker: the hot kernels in a loop, on the CPU or the GPU."""
    try:
        d = dict(np.load(npz))
        if kind == "gpu":
            import cupy as cp
            k = cp.RawKernel(UNDER_CU, "under_u8", options=("--fmad=false",))
            f, g = cp.asarray(d["front"]), cp.asarray(d["ghost"])
            o = cp.empty_like(f)
            n = f.shape[0] * f.shape[1]
            s_, C_, Q_, a_ = (cp.asarray(d[x]) for x in ("src", "C", "Q", "add"))
            cur, n32 = cp.asarray(d["cur"]), cp.asarray(d["n32"])

            def step():
                k(((n + 255) // 256,), (256,), (f, g, o, np.int64(n)))
                jpeg_core_xp(cp, s_, C_, Q_, a_)
                noise_add_xp(cp, cur, n32)
                cp.cuda.Device().synchronize()
            step()
            free, total = cp.cuda.runtime.memGetInfo()
        else:
            def step():
                glitch._under(d["front"], d["ghost"])
                jpeg_core_ref(d["src"], d["Q"], d["add"])
                noise_add_xp(np, d["cur"], d["n32"])
            step()
            free = total = None
        barrier.wait(timeout=600)
        t0 = perf()
        for _ in range(iters):
            step()
        q.put({"ok": True, "s": perf() - t0, "free": free, "total": total})
    except Exception as e:                               # noqa: BLE001
        try:
            barrier.abort()
        except Exception:                                # noqa: BLE001
            pass
        q.put({"ok": False, "error": f"{type(e).__name__}: {e}"})


def bench_scaling(M, procs: int, iters_cpu: int, iters_gpu: int, kinds, tmp: Path) -> dict:
    npz = tmp / "bench_data.npz"
    np.savez(npz, front=M.front, ghost=M.ghost, src=M.src, C=glitch._C, Q=M.Q, add=M.add, cur=M.cur,
             n32=np.random.default_rng(1).normal(0, 8, M.cur.shape[:2]).astype(np.float32))
    ctx = mp.get_context("spawn")
    out = {}
    for kind in kinds:
        iters = iters_gpu if kind == "gpu" else iters_cpu
        res = {}
        for n in sorted({1, procs}):
            barrier, q = ctx.Barrier(n), ctx.Queue()
            ps = [ctx.Process(target=_worker, args=(kind, str(npz), iters, barrier, q)) for _ in range(n)]
            for p in ps:
                p.start()
            rs = [q.get(timeout=900) for _ in ps]
            for p in ps:
                p.join(timeout=60)
            bad = [r["error"] for r in rs if not r["ok"]]
            if bad:
                res[str(n)] = {"error": bad[0]}
                continue
            wall = max(r["s"] for r in rs)
            res[str(n)] = {"steps_per_s": round(n * iters / wall, 2), "ms_per_step_each": round(wall / iters * 1e3, 2)}
            if kind == "gpu":
                res[str(n)]["vram_free_mb_while_running"] = round(min(r["free"] for r in rs) / 2**20)
        if all("steps_per_s" in res.get(str(k), {}) for k in (1, procs)) and procs > 1:
            res["speedup"] = round(res[str(procs)]["steps_per_s"] / res["1"]["steps_per_s"], 2)
        out[kind] = res
    return out


# --------------------------------------------------------------------------- 4. estimate
KERNEL_PARTS = ("under (blend)", "jpeg dct", "motion search", "motion features")
HOT = ("echo", "jpeg", "mosh", "noise") + KERNEL_PARTS


def estimate(prof, K) -> dict:
    """Frame time if each measured kernel ran at its benchmarked speed. Only the kernel's own
    share of the profile is scaled (the profile splits echo into its _under calls and the rest,
    etc.), so the rest of each effect stays at CPU speed. Still an estimate: the kernels were
    timed on one representative input, the render calls them on many different ones."""
    if not prof:
        return {}
    parts = prof["ms_per_frame"]
    total = prof["frame_ms"]["mean"]
    hot = {k: parts.get(k, 0.0) for k in HOT}

    def ms(k, v):
        d = K.get(k, {}).get(v)
        return d["ms"] if isinstance(d, dict) and "ms" in d else None

    def r(k, v):
        a, b = ms(k, v), ms(k, "cpu_numpy")
        return None if a is None or b is None else a / max(b, 1e-6)

    rng = ms("noise", "cpu_random_numbers_only") or 0.0
    noise = lambda v: (rng + ms("noise", v)) / ms("noise", "cpu_numpy") if ms("noise", v) is not None else None
    # which part of the profile each kernel replaces (motion search: the profile's
    # _block_motion time excludes its features, so compare search-only times)
    search = lambda v: (None if ms("mosh_search", v) is None else
                        ms("mosh_search", v) / max(ms("mosh_search", "cpu_numpy") - ms("mosh_search", "cpu_features_only"), 1e-6))
    plans = {
        "gpu_resident": {"under (blend)": r("under", "cuda_strict") or r("under", "cupy_port"),
                         "jpeg dct": r("jpeg", "cupy_port"), "motion search": search("cupy_port"),
                         "mosh": r("mosh_replay", "cupy_port"), "noise": noise("cupy_cpu_numbers")},
        "gpu_with_transfers": {"under (blend)": r("under", "cuda_strict_with_transfer"),
                               "jpeg dct": r("jpeg", "cupy_port_with_transfer"), "motion search": search("cupy_port"),
                               "mosh": r("mosh_replay", "cupy_port_with_transfer"),
                               "noise": noise("cupy_cpu_numbers_with_transfer")},
        "numba_serial": {"under (blend)": r("under", "numba_serial"), "motion search": search("numba_serial"),
                         "mosh": r("mosh_replay", "numba_serial"), "noise": noise("numba_serial_add")},
        "numba_parallel": {"under (blend)": r("under", "numba_parallel"), "motion search": search("numba_parallel"),
                           "mosh": r("mosh_replay", "numba_parallel"), "noise": noise("numba_parallel_add")},
    }
    res = {"measured_frame_ms": total, "hot_ms": {k: v for k, v in hot.items() if v > 0},
           "did_not_run": [e for e in ("echo", "jpeg", "mosh", "noise") if hot[e] < 0.05]}
    res["ceiling_frame_ms"] = round(total - sum(hot.values()), 1)
    res["ceiling_speedup"] = round(total / max(res["ceiling_frame_ms"], 1e-6), 2)
    exact = {"gpu_resident": [("under", "cuda_strict"), ("jpeg", "cupy_port"), ("mosh_search", "cupy_port"),
                              ("mosh_replay", "cupy_port"), ("noise", "cupy_cpu_numbers")],
             "numba_serial": [("under", "numba_serial"), ("mosh_search", "numba_serial"),
                              ("mosh_replay", "numba_serial"), ("noise", "numba_serial_add")]}
    for name, rat in plans.items():
        if any(v is None for v in rat.values()):
            continue
        saved = sum(hot.get(k, 0.0) * (1 - v) for k, v in rat.items())
        est = total - saved
        res[name] = {"frame_ms": round(est, 1), "speedup": round(total / max(est, 1e-6), 2),
                     "ratios": {k: round(v, 3) for k, v in rat.items()}}
        chk = exact.get(name) or exact.get(name.replace("_with_transfers", "_resident").replace("parallel", "serial"))
        if chk:
            res[name]["identical_frames"] = all(K.get(k, {}).get(v, {}).get("exact", False) for k, v in chk)
    return res


# --------------------------------------------------------------------------- report
def fmt_variant(name, v):
    if not isinstance(v, dict) or "ms" not in v:
        return None
    acc = ""
    if "exact" in v:
        acc = "identical" if v["exact"] else f"DIFFERS: {v['differing'] * 100:.3f}% px, max {v['max_abs']:g}"
        if "dtype" in v:
            acc += f", dtype {v['dtype']}"
    elif "note" in v:
        acc = v["note"]
    first = f"  first call {v['first_ms']:.0f} ms" if v["first_ms"] > 3 * v["ms"] + 5 else ""
    return f"    {name:<26}{v['ms']:>9.2f} ms   {acc}{first}"


def report(R):
    p = print
    s = R["system"]
    p(f"\n== Glitcher GPU bench {R['version']} ==")
    p(f"CPU: {s['cpu']} ({s['cores']} logical)   Python {s['python']}   numpy {s['numpy']}")
    p(f"GPU: {s.get('gpu') or '-'}   CuPy: {s.get('cupy') or R['notes'].get('cupy', '-')}")
    p(f"Numba: {s.get('numba') or R['notes'].get('numba', '-')}")
    pr = R.get("profile")
    if pr:
        p(f"\n-- 1. Real render, one worker, {pr['size'][0]}x{pr['size'][1]}, {pr['frames']} frames --")
        p(f"    render code: {pr.get('render_code', '?')}   (GLITCHER_NUMBA=0 for the numpy code)")
        p(f"    frame: mean {pr['frame_ms']['mean']} ms, median {pr['frame_ms']['median']} ms (min {pr['frame_ms']['min']}, max {pr['frame_ms']['max']})")
        tot = sum(pr["ms_per_frame"].values())
        for k, v in pr["ms_per_frame"].items():
            if v >= 0.5:
                p(f"    {k:<20}{v:>9.1f} ms  {v / tot * 100:5.1f}%")
    K = R.get("kernels", {})
    if K:
        p("\n-- 2. Hot spots: time per call, and whether the result matches glitch.py --")
        for k, d in K.items():
            extra = {"under": f"{d.get('px', 0) / 1e6:.2f} Mpx band", "jpeg": f"{d.get('blocks')} blocks",
                     "mosh_search": f"features {d.get('feat_px')}", "mosh_replay": f"{d.get('lag')} frames"}.get(k, "")
            pc = d.get("port_check", {})
            chk = "" if not pc else ("   port = glitch.py: identical" if pc.get("exact")
                                     else "   !! PORT CHECK FAILED - numbers not comparable")
            p(f"  {k} ({extra}){chk}")
            for name, v in d.items():
                line = fmt_variant(name, v)
                if line:
                    p(line)
    T = R.get("transfers")
    if T:
        p("\n-- 3. GPU costs --")
        p(f"    CUDA start-up {R['gpu_init_ms']:.0f} ms;  VRAM: {R.get('vram')}")
        for k, v in T.items():
            p(f"    {k:<14} {v['mb']:>6} MB  up {v['upload']['ms']:.2f} ms ({v['upload_gb_s']} GB/s), "
              f"pinned {v['upload_pinned']['ms']:.2f} ms, down {v['download']['ms']:.2f} ms")
    S = R.get("scaling")
    if S:
        p("\n-- several workers at once (under + jpeg + noise per step) --")
        for kind, d in S.items():
            p(f"    {kind}: " + ", ".join(f"{n} proc: {v.get('steps_per_s', v.get('error'))} steps/s"
                                       for n, v in d.items() if n != "speedup")
              + (f"  -> x{d['speedup']}" if "speedup" in d else ""))
    E = R.get("estimate")
    if E:
        p("\n-- 4. Estimated frame time (each kernel's share of the profile scaled by its measured speed-up) --")
        p(f"    now {E['measured_frame_ms']} ms; echo+jpeg+mosh+noise = {sum(E['hot_ms'].values()):.1f} ms")
        if E.get("did_not_run"):
            p(f"    (no {', '.join(E['did_not_run'])} events in these frames: counted as 0 ms - try a longer -n)")
        p(f"    ceiling (those four free): {E['ceiling_frame_ms']} ms = x{E['ceiling_speedup']}")
        for k in ("gpu_resident", "gpu_with_transfers", "numba_serial", "numba_parallel"):
            if k in E:
                same = {True: "identical frames", False: "frames DIFFER"}.get(E[k].get("identical_frames"), "")
                p(f"    {k:<19}{E[k]['frame_ms']:>7} ms = x{E[k]['speedup']:<5} {same}")
        p("    (gpu_resident: frames stay on the GPU between effects - needs every effect ported;")
        p("     gpu_with_transfers: only these kernels on the GPU, data up and down each call;")
        p("     numba_serial: what each render worker gets; numba_parallel: a single preview)")
    p(f"\nResult written to {R['json']} - send that file back.")


# --------------------------------------------------------------------------- main
def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    cut = argv.index("--") if "--" in argv else len(argv)
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 usage="gpu_bench.py [FRAMES] [options] [-- glitch.py options]")
    ap.add_argument("frames", nargs="?", help="folder of frames (default: a synthetic cut-out)")
    ap.add_argument("--preset", help="an editor preset (.json): profile your look instead of the defaults")
    ap.add_argument("--size", default="1920x1080", help="synthetic frame size (default 1920x1080)")
    ap.add_argument("-n", "--render-frames", type=int, default=60,
                    help="frames to profile (default 60; the effects fire as events, so not every frame)")
    ap.add_argument("--repeat", type=int, default=7, help="timed runs per kernel (default 7)")
    ap.add_argument("--procs", type=int, default=min(4, os.cpu_count() or 1),
                    help="workers for the sharing test (default min(4, cores); 1 = skip)")
    ap.add_argument("--no-profile", action="store_true", help="skip the real-render profile")
    ap.add_argument("--no-gpu", action="store_true")
    ap.add_argument("--no-numba", action="store_true")
    ap.add_argument("--json", default="gpu_bench_result.json")
    a = ap.parse_args(argv[:cut])
    rest = argv[cut + 1:]

    R = {"version": glitch.__version__, "notes": {}, "json": str(Path(a.json).resolve())}
    smi_before = None if a.no_gpu else nvidia_smi()
    R["system"] = {"os": platform.platform(), "cpu": platform.processor() or platform.machine(),
                   "cores": os.cpu_count(), "python": platform.python_version(), "numpy": np.__version__}
    tmp = Path(tempfile.mkdtemp(prefix="glitcher_bench_"))
    try:
        if a.frames:
            frames_dir = Path(a.frames)
            paths = glitch._gather(str(frames_dir))[1]
        else:
            sys.path.insert(0, str(ROOT / "tests"))
            import synth
            w, h = cli_size(a.size)
            frames_dir = tmp / "frames"
            print(f"making {max(a.render_frames, 12)} synthetic {w}x{h} frames ...", flush=True)
            paths = synth.make_frames(frames_dir, max(a.render_frames, 12), w, h, move=w / 300)
        if len(paths) < 6:
            sys.exit("need at least 6 frames (the time effects look back)")

        if not a.no_profile:
            print("1/4 profiling the real render ...", flush=True)
            R["profile"] = profile_pipeline(frames_dir, a.render_frames, a.preset, rest, tmp)

        cp = numba = None
        if not a.no_gpu:
            t0 = perf()
            cp, why = load_cupy()
            R["gpu_init_ms"] = round((perf() - t0) * 1e3, 1)
            if cp is None:
                R["notes"]["cupy"] = why
            else:
                props = cp.cuda.runtime.getDeviceProperties(0)
                name = props["name"]
                R["system"].update(gpu=name.decode() if isinstance(name, bytes) else str(name), cupy=cp.__version__,
                                   cuda_runtime=cp.cuda.runtime.runtimeGetVersion(),
                                   cuda_driver=cp.cuda.runtime.driverGetVersion(),
                                   compute_capability=f"{props['major']}.{props['minor']}")
        else:
            R["notes"]["cupy"] = "skipped (--no-gpu)"
        if not a.no_numba:
            numba, why = load_numba()
            if numba is None:
                R["notes"]["numba"] = why
            else:
                R["system"]["numba"] = numba.__version__
        else:
            R["notes"]["numba"] = "skipped (--no-numba)"

        print("2/4 benchmarking the hot spots ...", flush=True)
        M = bench_material(paths)
        R["kernels"] = bench_kernels(M, a.repeat, cp, numba)
        if cp is not None:
            R["transfers"] = bench_transfers(M, a.repeat, cp)
            free, total = cp.cuda.runtime.memGetInfo()
            pool = cp.get_default_memory_pool()
            smi = nvidia_smi()
            R["vram"] = {"total_mb": round(total / 2**20), "free_mb": round(free / 2**20),
                         "cupy_pool_mb": round(pool.total_bytes() / 2**20)}
            if smi and smi_before:
                R["vram"]["this_process_mb"] = round(smi["used_mb"] - smi_before["used_mb"])
                R["vram"]["context_mb_approx"] = round(smi["used_mb"] - smi_before["used_mb"]
                                                       - pool.total_bytes() / 2**20)
                R["vram"]["used_by_others_mb"] = round(smi_before["used_mb"])
                R["system"]["driver"] = smi["driver"]
        if a.procs > 1:
            print(f"3/4 {a.procs} workers at once ...", flush=True)
            ms_cpu = sum(R["kernels"][k]["cpu_numpy"]["ms"] for k in ("under", "jpeg", "noise"))
            iters_cpu = max(5, int(3000 / max(ms_cpu, 1)))
            R["scaling"] = bench_scaling(M, a.procs, iters_cpu, iters_cpu * 10,
                                         ["cpu"] + (["gpu"] if cp is not None else []), tmp)
        print("4/4 estimating ...", flush=True)
        R["estimate"] = estimate(R.get("profile"), R["kernels"])
        Path(a.json).write_text(json.dumps(R, indent=1, default=str), encoding="utf-8")
        report(R)
        return R
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
