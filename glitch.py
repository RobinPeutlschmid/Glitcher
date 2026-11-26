#!/usr/bin/env python3
"""
Glitcher 1.2 - glitch.py: temporally coherent glitch for transparent PNG cutouts and image sequences,
with an optional fake-3D mode in which the glitch reacts to the subject turning.

Dependencies: numpy, Pillow      (pip install numpy pillow)
EXR frames additionally need:     pip install OpenEXR

Usage
-----
    # a folder of RGBA PNG frames -> folder of glitched frames (all cores)
    python glitch.py frames/ -o frames_glitch/ --fps 25 --seed 7 -i 0.6

    # 3D mode: give the subject's yaw per frame (see "3D mode" below)
    python glitch.py frames/ -o out/ --pose yaw.csv --seed 7 -i 0.6

    # a glob works too; a single still can be animated over time
    python glitch.py "shot/person_*.png" -o out/
    python glitch.py person.png --frames 48 -o preview/

    # glitch layout follows the subject instead of staying fixed on screen
    python glitch.py frames/ --follow 0.4

    # re-render only frames 100-140 so they match a full render exactly: point at the whole
    # folder and pick the range (layout and the time effects see the full sequence)
    python glitch.py frames/ -o out/ --seed 7 --use-numbers --range 100-140
    # if only those files are at hand: save the layout boxes on the full render, reuse them
    # (time effects then lack the earlier frames, so they will differ near the start)
    python glitch.py frames/ -o out/ --seed 7 --use-numbers --boxes shot_boxes.json
    python glitch.py frames_100_140/ -o out/ --seed 7 --use-numbers --boxes shot_boxes.json

    # time effects only (echo trails, frame holds, datamosh), stronger datamosh
    python glitch.py frames/ -o out/ --skip jpeg,blocks,slices,wave,sort,smear,split --gain mosh=2

    # glitch the keyed actor together with the clean background plate: trails, smears and
    # datamosh spill into the world around them (output = finished comp)
    python glitch.py keyed/ -o comp/ --plate bg_plate/ --spill 0.6

    # glitch to music: 12.5 s into the track lines up with the first frame
    python glitch.py frames/ -o out/ --audio track.mp3 --audio-start 12.5 --audio-sens 0.6

    # glitch reacts to movement: calm while the actor holds still, spikes on fast moves
    python glitch.py frames/ -o out/ --motion 0.8

    # direct the timing: keyframed intensity / per-effect strength, plus bursts on story beats
    python glitch.py frames/ -o out/ --curve timing.csv --hits 48,112:0.6,200:1:0.8

    # the hallucination materialises at frame 12 (1.5 s) and dissolves at frame 180
    python glitch.py frames/ -o out/ --appear 12:1.5 --vanish 180

    # EXR in -> 16-bit half EXR out (linear, premultiplied), or force a format
    python glitch.py shot_exr/ -o out_exr/ --seed 7
    python glitch.py frames_png/ -o out_exr/ --format exr --exr-compression piz

    # alpha video, e.g. for an editor/compositor
    ffmpeg -framerate 24 -i out/%04d.png -c:v prores_ks -profile:v 4444 -pix_fmt yuva444p10le out.mov

As a module
-----------
    from glitch import glitch
    out = glitch(Image.open("f_0012.png"), t=12/24, seed=7, intensity=0.6, bbox=(120, 80, 900, 1300))
    out = glitch(img, t=12/24, seed=7, yaw=35.0)          # 3D mode, subject turned 35 deg
    out = glitch(img, t=12/24, seed=7, yaw=35, pitch=20, roll=-8)   # high angle, slight tilt

How it stays coherent in time
-----------------------------
Every glitch is an *event* with a start time, a lifetime and fixed parameters. The schedule is
a pure function of (seed, effect, time): time is cut into 1 s buckets, each seeded by
(seed, effect, bucket). No state is carried between frames, so any frame can be rendered alone,
in parallel or in chunks and matches a full render bit for bit. Events fade in/out
(envelope), displacements hold ~3 frames then jump, and a slow "activity" curve (`--bursts`)
gates events by their own threshold to give calm stretches and bursts.

3D mode (--pose / yaw=)
-----------------------
Events stop being 2-D rectangles and become *planes in the subject's own 3-D space*:

  * Space. Origin at the centre of the layout box, x right, y down, z toward the camera when the
    subject faces it. Units are pixels; the volume spans the box width and height and
    +-depth*width in z (`--depth`, default 0.35).
  * Each event gets a 3-D centre, a facing angle alpha around the vertical axis (its plane normal
    n = (sin a, 0, cos a)), an in-plane horizontal axis a = (cos a, 0, -sin a) and a half-width.
  * Camera. The subject's yaw (times `--lock`) gives a rotation. Its rows are the camera's
    axes expressed in the subject's space: `right` and `cam` (the direction toward the camera).
    That `cam` vector is the "camera position inside the glitch volume".
  * Visibility. A plane is drawn only if it faces the camera: vis = smoothstep(n . cam). As the
    subject turns, planes that were seen edge-on fade in, frontal ones fade out, and planes on the
    subject's back appear when they turn their back to camera. One-sided, like back-face culling.
  * Projection. Centre: X = p . right, Z = p . cam, with weak perspective s = D/(D-Z)
    (D = `--cam-dist` box heights). Widths shrink with a . right (foreshortening), and every
    displacement is a 3-D vector (in-plane + a bit along the normal) projected the same way, so a
    tear on a plane that turns away shrinks toward zero instead of sliding.
  * Depth layering. Planes in front of the body (Z > 0) are applied to the subject as usual.
    Planes behind it are rendered as a *ghost layer* composited underneath, so they only show
    where they stick out past the silhouette. Events crossing the body plane cross-fade.
  * Screen-space effects (split, tint, scanlines, noise) stay 2-D, except that the RGB split
    gains a parallax term proportional to sin(yaw): R sits in front, B behind.
  * Pitch and roll. The full rotation is R = Rroll . Rpitch . Ryaw. Pitch (camera above or
    below the subject) tips the volume toward/away from the camera: vertical extents shrink by
    the plane's vertical foreshortening, depth order changes, and tilted/horizontal planes
    (`--tilt`, default 25% of planes) become visible - from above you see the tops of slabs,
    from below their undersides. Roll is a pure 2-D rotation, so the 3-D effects are computed in
    an upright frame and only the changed pixels are rotated back; screen-space effects
    (scanlines, grain) stay aligned with the frame. The RGB split's depth parallax follows yaw
    horizontally and pitch vertically. `lock` applies to yaw only.
  * Scale and distance. If the pose file has a `scale` column (estimate_pose.py writes one),
    the follow box grows and shrinks with the person, so the glitch stays locked to the body
    when they walk toward the camera or the camera dollies. A `cam_dist` column (written when
    the estimator knows the focal length) sets the perspective strength per frame: close to the
    lens the volume's depth exaggerates, far away it flattens. Without it `--cam-dist` (default 3).
  * With --pose (or --pitch/--roll), `--follow` defaults to 0.4 so the volume travels with the
    subject.

Pose file: text/CSV with "frame, yaw[, pitch[, roll]]" rows in degrees (# comments allowed, any of
, ; tab space as separator; a header line may name the columns, then their order is free and
the optional `scale` and `cam_dist` columns are read too). Keys may be sparse and are taken literally like animation curves
(0 -> 360 is a full turn); `--pose-unwrap` instead treats jumps > 180 as wrap-around, for
tracker output. Linearly interpolated, optionally smoothed with `--pose-smooth SEC`. 0 = facing camera,
positive = the subject's front turns toward screen right. `frame` uses the same index as the
glitch's time axis (position in the sequence + --offset, or the filename number with
--use-numbers).

Time effects (echo, stutter, mosh)
----------------------------------
These read earlier *source* frames of the sequence (up to 24 back; never earlier glitched output,
so every frame still renders on its own and in any order). They run first, on the clean frame,
and the other effects glitch on top. A single still has no earlier frames, so they do nothing.
  echo     ghost trails: 2-4 fading copies of the last frames behind the body (sometimes one
           faintly over it), drifting slightly, natural / cyan / magenta
  stutter  frame hold: a block, a band or the whole subject freezes on an older frame, either
           holding or jumping forward every few frames (dropped packets), sometimes shifted
  mosh     datamosh: inside a band the image stops refreshing - the frame from when the event
           began is carried along by each later frame's block motion (estimated here, 8x8
           blocks on a downscaled image), so colour smears with the movement, then cuts back
They cost roughly 2x render time while active (frames are loaded and motion estimated per
worker; consecutive frames share both). Strength and count: --gain echo=..., skip: --skip.

Timing (--curve, --hits)
------------------------
By default glitches come and go on a slow random calm/burst curve (--bursts). To direct them:
  --curve FILE  keyframes, one row per key frame, header required, cells may be left empty:
                    frame,intensity,jpeg,slices,hit
                    0,0.2,,,
                    40,1.0,2,,
                    48,,,,1
                    90,0.3,0.5,0,
                intensity and effect columns are multipliers on -i and --gain (1 = as set),
                linear between a column's keys and held beyond them.
  --hits        bursts on story beats (also a 'hit' column in the curve file): activity jumps
                to full, intensity rises to at least the hit strength, every effect gets up to
                twice the events, then it decays over --hit-len seconds.
Because events are nested sets, raising a curve adds events and lowering it removes them -
nothing reshuffles - and every frame still depends only on its own number.

Materialise / dematerialise (--appear, --vanish, or 'appear' / 'vanish' curve columns)
  The subject builds up out of glitch structure over the transition: a fixed per-pixel arrival
  map (a sweep - top-down, bottom-up, centre-out or sideways - mixed with blocks at two scales
  and scanline bands) decides when each part arrives. Freshly arrived pixels sit shoved
  sideways in slices that land as they settle, as flat mosaic blocks, briefly lit up; the glitch
  peaks mid-transition. A vanish is the reverse: parts slip, block up, flare and go. Before the
  first appear and after a vanish the frame is empty. The map is laid over the layout box, so it
  follows the subject (--follow) and is the same on every frame of the transition.

EXR / high bit depth
--------------------
EXR frames are processed in 32-bit float end to end - no 8-bit step anywhere - and written as
16-bit half float (or 32-bit with --exr-depth float). By EXR convention the input is taken as
scene-linear with premultiplied alpha (--exr-space display / --exr-alpha straight otherwise):
it is un-premultiplied and sRGB-encoded for the glitch maths (the look is designed on display-
encoded values), then decoded and premultiplied again. HDR values above 1.0 are kept, including
inside glitch fragments, and pixels the glitch did not touch are copied from the source file
bit for bit. The output format follows the input unless --format is given (PNG in -> EXR out
converts sRGB to linear; EXR in -> PNG out writes the display-encoded 8-bit result).

Effects (applied in this order; names usable in --skip)
-------------------------------------------------------
    jpeg       JPEG-style block corruption: gray fills, colour drift, ringing, repeated blocks
    blocks     rectangles copied elsewhere, channel-swapped / inverted / posterized
    slices     horizontal strips displaced sideways (tearing)
    wave       scrolling sinusoidal warp of a band
    sort       pixel sorting by luminance along columns (silhouette preserved)
    smear      pixel rows stretched into vertical streaks / drips
    split      RGB channel separation
    tint       luminance -> dark purple..cyan blend (off unless --tint > 0)
    scanlines  CRT scanlines with faint flicker
    noise      film grain, fresh every frame
"""
from __future__ import annotations

import argparse
import copy

__version__ = "2.0.0"
APP_NAME = "Glitcher"
PROJECT_URL = "https://github.com/RobinPeutlschmid/Glitcher"   # downloads: PROJECT_URL + "/releases/latest"
import glob as _glob
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from collections import OrderedDict  # noqa: F401  (type hints of the per-process caches)
from dataclasses import dataclass, replace as _dc_replace
from functools import lru_cache
from pathlib import Path
from typing import NamedTuple

import numpy as np
from PIL import Image

EFFECT_ORDER = ["jpeg", "blocks", "slices", "wave", "sort", "smear",
                "split", "tint", "scanlines", "noise", "echo", "stutter", "mosh"]
# time effects read earlier source frames (sequences only); they run first, on the clean frame
TIME_FX = ("echo", "stutter", "mosh")
MAX_BACK = 24                      # deepest look-back in frames
FX_ID = {n: i + 1 for i, n in enumerate(EFFECT_ORDER)}
IMG_EXT = {".png", ".webp", ".tif", ".tiff", ".exr"}
TAU = 2 * math.pi

# events per second (at full activity), min life [s], max life [s]
_SCHED = {
    "jpeg":   (2.5, 0.25, 1.1),
    "blocks": (16.0, 0.12, 0.6),
    "slices": (30.0, 0.08, 0.5),
    "wave":   (3.5, 0.5, 1.8),
    "sort":   (8.0, 0.3, 1.2),
    "smear":  (40.0, 0.25, 1.3),
    "echo":   (1.2, 0.4, 1.4),
    "stutter": (3.0, 0.15, 0.7),
    "mosh":   (0.8, 0.35, 1.0),
}
# In 3D mode roughly half the planes face away and half of the rest sit behind the body,
# so more are spawned to keep a similar amount of visible glitch.
_PLANE_BOOST = 3.0
_LAYERED = {"jpeg", "blocks", "slices", "wave", "sort", "smear"}


# --------------------------------------------------------------------------- compiled kernels
# glitch_numba.py holds Numba versions of _under, fx_noise's grain and fx_mosh's replay: the
# same arithmetic as the numpy code below, so the frames are identical either way, ~1.3-1.7x
# faster per frame. Used when Numba is installed and GLITCHER_NUMBA isn't 0; loaded on first
# use in each process (the editor's own process never needs it).
_FAST = None                       # the glitch_numba module, False = numpy; None = not tried yet
FAST_STATUS = "not loaded yet"


def _user_cache_dir() -> Path:
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local")
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Caches"
    else:
        base = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache")
    return base / APP_NAME


def _fast():
    """The compiled kernels, or False to use numpy. Loads them the first time it's asked."""
    global _FAST, FAST_STATUS
    if _FAST is not None:
        return _FAST
    _FAST = False
    if os.environ.get("GLITCHER_NUMBA", "1").strip().lower() in ("0", "off", "no", "false"):
        FAST_STATUS = "numpy (GLITCHER_NUMBA=0)"
        return _FAST
    t0 = time.time()
    try:
        # Compiled code is cached on disk, keyed to the source file. In the packaged app the
        # module is loaded from its bundled .py file (PyInstaller keeps modules in an archive,
        # where Numba can't key a cache), and the cache goes to the user's cache folder.
        os.environ.setdefault("NUMBA_CACHE_DIR", str(_user_cache_dir() / "numba"))
        src = Path(getattr(sys, "_MEIPASS", "")) / "glitch_numba.py"
        if getattr(sys, "frozen", False) and src.is_file():
            import importlib.util
            spec = importlib.util.spec_from_file_location("glitch_numba", src)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            sys.modules["glitch_numba"] = mod
        else:
            import glitch_numba as mod
        import numba
    except ImportError as e:
        FAST_STATUS = ("numpy (Numba is not installed)" if getattr(e, "name", "") in ("numba", "llvmlite")
                       else f"numpy (could not load the compiled kernels: {e})")
        return _FAST
    except Exception as e:                                  # noqa: BLE001 - numpy is always there
        FAST_STATUS = f"numpy (the compiled kernels failed: {type(e).__name__}: {e})"
        return _FAST
    _FAST = mod
    FAST_STATUS = f"numba {numba.__version__} ({time.time() - t0:.1f} s to load)"
    return _FAST


def fast_status() -> dict:
    """Which code renders, and whether the compiled kernels came from the disk cache."""
    F = _fast()
    d = {"backend": "numba" if F else "numpy", "status": FAST_STATUS}
    if F:
        import numba
        kern = [F.any_alpha, F.under_u8, F.noise_u8, F.replay]
        d.update(numba=numba.__version__, cache_dir=numba.config.CACHE_DIR,
                 cache_hits=sum(sum(k.stats.cache_hits.values()) for k in kern),
                 cache_misses=sum(sum(k.stats.cache_misses.values()) for k in kern),
                 signatures=sum(len(k.signatures) for k in kern))
        d["from_cache"] = d["cache_hits"] == d["signatures"]
    return d


def warm_fast() -> str:
    """Load (the first time on a machine: compile) the kernels in this process; -> status."""
    _fast()
    return FAST_STATUS


def numpy_worker_init():
    """Initializer for worker processes that should stay on numpy (the editor's playback)."""
    global _FAST, FAST_STATUS
    _FAST, FAST_STATUS = False, "numpy (playback worker)"
    set_engine_threads(1)                           # a playback worker per core: one thread each
    global NATIVE_HISTORY_BYTES
    NATIVE_HISTORY_BYTES = 150_000_000              # ... and many of them: a smaller frame cache


# --------------------------------------------------------------------------- renderer
# Since 2.0 the frames are rendered by the native engine (engine/, glitch_core) when its
# library is found: the same planner, files and options, about 10x faster. GLITCHER_ENGINE:
# auto (default: native if it loads, else this file), python (always this file), native
# (native or an error). The two renderers make the same kind of look, not the same pixels.
_NATIVE = None        # the glitch_native module, or False


def native():
    """The glitch_native module if frames are rendered natively here, else None."""
    global _NATIVE
    if _NATIVE is None:
        want = os.environ.get("GLITCHER_ENGINE", "auto").strip().lower()
        _NATIVE = False
        if want not in ("python", "py", "numpy", "off", "0"):
            why = ""
            try:
                import glitch_native as gn
                if gn.available():
                    _NATIVE = gn
                else:
                    why = gn.unavailable_reason()
            except ImportError as ex:
                why = str(ex)
            if want == "native" and not _NATIVE:
                raise RuntimeError(f"GLITCHER_ENGINE=native, but the native engine can't be loaded: {why}")
    return _NATIVE or None


def renderer() -> str:
    """Which renderer this process uses, for logs and the About box."""
    gn = native()
    return f"native (glitch_core {gn.engine().version})" if gn else "python"


# A mosh replays every frame since its event began, so the native engine asks for up to 24
# earlier frames; loading them is then what takes the time. Its processes keep more decoded
# frames (it runs few of them): ~24 at 1080p... 18 at 4K.
NATIVE_HISTORY_BYTES = 600_000_000


def render(img, t: float = 0.0, **kw):
    """glitch() on whichever renderer this process uses (see native())."""
    gn = native()
    if gn:
        SeqHistory.BUDGET = max(SeqHistory.BUDGET, NATIVE_HISTORY_BYTES)
        return gn.glitch(img, t, **kw)
    return glitch(img, t, **kw)


def warm_renderer() -> str:
    """Load this process's renderer (the native library, else the compiled kernels); -> which."""
    if native():
        return renderer()
    return warm_fast()


def set_engine_threads(n: int):
    """Threads for the native engine in this process; only works before its first frame."""
    os.environ["RAYON_NUM_THREADS"] = str(max(1, int(n)))


def _c(a):
    """a as a C-contiguous, writable array (what the kernels' signatures take); copies if not."""
    return a if a.flags.c_contiguous and a.flags.writeable else np.array(a, order="C")


# --------------------------------------------------------------------------- time / randomness
@dataclass(frozen=True)
class Params:
    seed: int = 0
    intensity: float = 0.5     # 0..1 overall strength
    fps: float = 25.0
    tempo: float = 1.0         # >1 = shorter, snappier events
    density: float = 1.0       # multiplier on event counts
    bursts: float = 0.5        # 0 = constant activity, 1 = strong calm/burst cycles
    pad: float = 0.0
    tint: float = 0.0
    skip: tuple = ()
    bleed: int | None = None
    quality: int | None = None
    gains: tuple = ()          # ((effect, strength), ...); 1 = default, 0 = off, up to ~3
    boost: float = 0.0         # 0..1 floor on activity at this moment (hits)


def _ss(x: float, a: float, b: float) -> float:
    """Scalar smoothstep from a (->0) to b (->1); a > b gives a falling edge."""
    if a == b:
        return float(x >= a)
    t = min(1.0, max(0.0, (x - a) / (b - a)))
    return t * t * (3 - 2 * t)


def _env(u: float) -> float:
    """Event envelope over life u in [0,1]: quick attack, long decay."""
    return _ss(u, 0.0, 0.1) * _ss(u, 1.0, 0.6)


@lru_cache(maxsize=1 << 17)
def _u(*key: int) -> float:
    """Deterministic uniform [0,1) from a tuple of non-negative ints (stateless hash noise)."""
    return float(np.random.default_rng(key).random())


def _vnoise(seed: int, ch: int, x: float) -> float:
    """Smooth 1-D value noise in [0,1]."""
    i = math.floor(x)
    f = x - i
    s = f * f * (3 - 2 * f)
    a, b = _u(seed, 900 + ch, i), _u(seed, 900 + ch, i + 1)
    return a + (b - a) * s


def _activity(p: Params, t: float) -> float:
    n = 0.6 * _vnoise(p.seed, 0, t * 0.9) + 0.4 * _vnoise(p.seed, 1, t * 2.3)
    n = min(1.0, max(0.0, (n - 0.5) * 1.6 + 0.5))
    return min(1.0, max(0.0, p.intensity + p.bursts * (n - 0.5) * 1.2, p.boost))


@lru_cache(maxsize=16384)
def _bucket(seed: int, fx: int, k: int, rate: float, lo: float, hi: float):
    rng = np.random.default_rng((seed, fx, k))
    out = []
    for j in range(int(rng.poisson(rate))):
        start = k + float(rng.random())
        life = float(rng.uniform(lo, hi))
        thr = float(rng.random())
        P = tuple(float(v) for v in rng.random(8))
        out.append((j, start, life, thr, P))
    return tuple(out)


# --------------------------------------------------------------------------- 3D space
class Space3D:
    """The camera as seen from inside the subject's glitch volume.

    Subject -> camera rotation R = Rroll . Rpitch . Ryaw (angles in radians):
      yaw    subject turned about its vertical axis (already multiplied by `lock`);
             + = the subject's front turns toward screen right
      pitch  camera elevation relative to the subject; + = camera above (high angle)
      roll   rotation about the view axis; + = subject appears rotated clockwise on screen
    Roll is a pure 2-D rotation of the picture, so glitches are built in an upright working frame
    (R' = Rpitch . Ryaw) and that frame is rotated back by `roll` afterwards. Rows of R' are the
    camera's axes in subject space:
      right  screen-right direction      up   screen-down (image y) direction
      cam    direction from the subject toward the camera
    Projecting a subject-space vector v onto the screen is (v.right, v.up); its depth is v.cam.
    `tilt` = share of glitch planes that are tilted away from vertical (up to horizontal), which
    only show from above or below.
    """

    def __init__(self, yaw: float, cam_dist: float = 3.0, depth: float = 0.35,
                 pitch: float = 0.0, roll: float = 0.0, tilt: float = 0.0):
        c, s = math.cos(yaw), math.sin(yaw)
        cp, sp = math.cos(pitch), math.sin(pitch)
        Ry = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
        Rx = np.array([[1.0, 0.0, 0.0], [0.0, cp, sp], [0.0, -sp, cp]])
        R = Rx @ Ry
        self.yaw, self.pitch, self.roll = yaw, pitch, roll
        self.cam_dist = max(0.5, float(cam_dist))
        self.depth = max(0.0, float(depth))
        self.tilt = float(np.clip(tilt, 0.0, 1.0))
        self.R = R
        self.right, self.up, self.cam = R[0].copy(), R[1].copy(), R[2].copy()


class Pl(NamedTuple):
    """Screen placement of one 3-D plane event."""
    sx: float     # projected centre x (padded canvas px)
    sy: float     # projected centre y
    s: float      # perspective scale at the centre
    ax: float     # in-plane horizontal axis projected on screen x (foreshortening, signed)
    nx: float     # plane normal projected on screen x
    vis: float    # facing weight 0..1 (back-face culled, fades near edge-on)
    wf: float     # 1 = fully in front of the body, 0 = fully behind
    xr: float     # projected half-width on screen
    rho: float    # out-of-plane share of displacements (-0.5..0.5)
    Z: float      # depth of the centre toward the camera (px)
    vy: float = 1.0   # vertical foreshortening of the plane (1 = seen square-on vertically)


class Ev(NamedTuple):
    key: tuple      # (seed, fx, bucket, j) - unique id, usable as hash key
    u: float        # progress through life, 0..1
    g: float        # activity gate 0..1 (in 3D: x visibility x layer weight)
    P: tuple        # 8 fixed uniforms describing this event
    pl: object = None   # Pl in 3D mode, None in 2D mode
    t0: float = 0.0     # start time [s] (time effects count frames since the event began)


class Ctx:
    def __init__(self, p: Params, t: float, fi: int, bbox, bleed: int,
                 space: Space3D | None = None, layer: str = "front", hist=None):
        self.p, self.t, self.fi, self.bbox, self.bleed = p, t, fi, bbox, bleed
        self.space, self.layer, self.hist = space, layer, hist
        self.act = _activity(p, t)
        self.gains = dict(p.gains)
        self._raw = {}      # fx -> events before 3D layering; shared by .layered() copies

    def gain(self, fx: str) -> float:
        return self.gains.get(fx, 1.0)

    def layered(self, layer: str) -> "Ctx":
        c = copy.copy(self)
        c.layer = layer
        return c

    def events(self, fx: str) -> list[Ev]:
        out = self._raw.get(fx)
        if out is None:
            out = self._raw[fx] = self._events_raw(fx)
        if self.space is None:
            return out
        res = []
        for ev in out:
            pl = self._place(ev)
            w = pl.wf if self.layer == "front" else 1.0 - pl.wf
            g = ev.g * pl.vis * w
            if g > 0.01:
                res.append(ev._replace(g=g, pl=pl))
        return res

    def events2d(self, fx: str) -> list[Ev]:
        """Screen-space events (time effects): no 3-D placement."""
        out = self._raw.get(fx)
        if out is None:
            out = self._raw[fx] = self._events_raw(fx)
        return out

    def start_frame(self, ev: Ev) -> int:
        return int(math.ceil(ev.t0 * self.p.fps - 1e-6))

    def _events_raw(self, fx: str) -> list[Ev]:
        p = self.p
        rate, lo, hi = _SCHED[fx]
        lo, hi, rate = lo / p.tempo, hi / p.tempo, rate * p.density
        if self.space is not None and fx not in TIME_FX:
            rate *= _PLANE_BOOST
        # strength (gain): more AND bigger. Count changes through nested sets, so dragging a
        # slider adds/removes individual events instead of reshuffling them: below 1 each event
        # survives with probability = gain; above 1 a second stream (fixed rate 2x) adds events
        # with probability (gain-1)/2. Amplitude scales with sqrt(gain). Gain 1 = unchanged.
        gf = self.gain(fx)
        amp_g = math.sqrt(gf) if gf != 1.0 else 1.0
        streams = [(FX_ID[fx], rate, gf if gf < 1.0 else None)]
        if gf > 1.0:
            streams.append((FX_ID[fx] + 100, rate * 2.0, min(1.0, (gf - 1.0) / 2.0)))
        t, out = self.t, []
        for fid, rt, keep in streams:
            for k in range(max(0, int(t - hi)), int(t) + 1):
                for j, start, life, thr, P in _bucket(p.seed, fid, k, rt, lo, hi):
                    if start <= t < start + life:
                        if keep is not None and _u(p.seed, fid, k, j, 60) >= keep:
                            continue
                        g = _ss(self.act, thr - 0.12, thr + 0.12) * min(1.0, p.intensity * 5) * amp_g
                        if g > 0.01:
                            out.append(Ev((p.seed, fid, k, j), (t - start) / life, g, P, None, start))
        return out

    def _place(self, ev: Ev) -> Pl:
        """Put the event in subject space and project it through the camera."""
        sp = self.space
        x0, y0, x1, y1 = self.bbox
        bw, bh = x1 - x0, y1 - y0
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        q = [_u(*ev.key, 50, i) for i in range(6)]        # separate stream from the 2D genome
        p = np.array([(q[0] - .5) * bw, (q[1] - .5) * bh, (2 * q[2] - 1) * sp.depth * bw])
        al = q[3] * TAU
        # tilted planes: elevation of the normal, up to horizontal (floor/ceiling-like slabs)
        el = (2 * _u(*ev.key, 50, 7) - 1) * (math.pi / 2) if _u(*ev.key, 50, 6) < sp.tilt else 0.0
        ce, se = math.cos(el), math.sin(el)
        a = np.array([math.cos(al), 0.0, -math.sin(al)])            # in-plane horizontal axis
        n = np.array([math.sin(al) * ce, -se, math.cos(al) * ce])   # plane normal (y is down)
        b = np.array([math.sin(al) * se, ce, math.cos(al) * se])    # in-plane 'vertical' axis
        X, Y, Z = float(p @ sp.right), float(p @ sp.up), float(p @ sp.cam)
        D = sp.cam_dist * bh
        s = D / max(D - Z, 0.25 * D)
        ax = float(a @ sp.right)
        eps = 0.15 * bw
        return Pl(sx=cx + X * s, sy=cy + Y * s, s=s, ax=ax, nx=float(n @ sp.right),
                  vis=_ss(float(n @ sp.cam), 0.05, 0.35), wf=_ss(Z, -eps, eps),
                  xr=abs(ax) * (0.2 + 0.3 * q[5]) * bw * s, rho=(2 * q[4] - 1) * 0.5, Z=Z,
                  vy=abs(float(b @ sp.up)))

    def held(self, ev: Ev, ch: int = 0, frames: int = 3) -> float:
        """Uniform value that stays constant for `frames` frames, then jumps (stepped jitter)."""
        return _u(*ev.key, ch, self.fi // frames)


# --------------------------------------------------------------------------- pixel helpers
def _lum(rgb):
    return rgb[..., 0] * 0.299 + rgb[..., 1] * 0.587 + rgb[..., 2] * 0.114


def _shift(arr, dx: int, dy: int):
    """Translate without wrap-around; vacated area is zero (transparent)."""
    h, w = arr.shape[:2]
    out = np.zeros_like(arr)
    if abs(dx) >= w or abs(dy) >= h:
        return out
    ys, yd = (slice(0, h - dy), slice(dy, h)) if dy >= 0 else (slice(-dy, h), slice(0, h + dy))
    xs, xd = (slice(0, w - dx), slice(dx, w)) if dx >= 0 else (slice(-dx, w), slice(0, w + dx))
    out[yd, xd] = arr[ys, xs]
    return out


def _dilate(mask, r: int):
    """Square dilation by r px in O(HW) (cumulative sums), plain numpy. Works on the bounding
    box of the mask grown by r; everything outside stays False."""
    r = int(r)
    if r <= 0:
        return mask.copy()
    rows = np.flatnonzero(mask.any(1))
    if rows.size == 0:
        return mask.copy()
    cols = np.flatnonzero(mask.any(0))
    h, w = mask.shape
    y0, y1 = max(0, rows[0] - r), min(h, rows[-1] + 1 + r)
    x0, x1 = max(0, cols[0] - r), min(w, cols[-1] + 1 + r)
    out = np.zeros_like(mask)
    out[y0:y1, x0:x1] = _dilate_full(mask[y0:y1, x0:x1], r)
    return out


def _dilate_full(mask, r: int):
    m = mask
    for ax in (0, 1):
        m = np.moveaxis(m, ax, 0)
        n = m.shape[0]
        s0 = np.concatenate([np.zeros((1,) + m.shape[1:], np.int32),
                             np.cumsum(m, axis=0, dtype=np.int32)])
        i = np.arange(n)
        m = (s0[np.minimum(i + r, n - 1) + 1] - s0[np.maximum(i - r, 0)]) > 0
        m = np.moveaxis(m, 0, ax)
    return m


def _slice_window(out, a, y: int, sh: int, dx: int, xa: int, xb: int):
    """Cut the pixels of rows y..y+sh, columns xa..xb out and paste them dx further."""
    w = a.shape[1]
    win = a[y:y + sh, xa:xb]
    row = out[y:y + sh]
    row[:, xa:xb] = 0
    s0 = max(0, -(xa + dx))
    s1 = (xb - xa) - max(0, xb + dx - w)
    if s1 <= s0:
        return
    src = win[:, s0:s1]
    dst = row[:, xa + dx + s0:xa + dx + s1]
    m = src[..., 3] > 0
    dst[m] = src[m]


def _px(a):
    """RGBA uint8 (h,w,4) viewed as one uint32 per pixel: whole-pixel compares in one pass."""
    return np.ascontiguousarray(a).view(np.uint32)[..., 0]


def _nz(a):
    """Per-pixel 'any channel non-zero' for uint8 or float images."""
    return _px(a) != 0 if a.dtype == np.uint8 else np.any(a != 0, axis=2)


def _neq(a, b):
    """Per-pixel 'differs' for uint8 or float images."""
    return _px(a) != _px(b) if a.dtype == np.uint8 else np.any(a != b, axis=2)


def _content_box(a, margin: int = 0):
    """(y0, y1, x0, x1) around every pixel with any non-zero channel, grown by `margin` and
    clipped to the image; None if the image is all zero. Outside it the image is all zero, so
    per-pixel work there can be skipped."""
    nz = _nz(a)
    rows = np.flatnonzero(nz.any(1))
    if rows.size == 0:
        return None
    cols = np.flatnonzero(nz.any(0))
    h, w = a.shape[:2]
    return (max(0, rows[0] - margin), min(h, rows[-1] + 1 + margin),
            max(0, cols[0] - margin), min(w, cols[-1] + 1 + margin))


def _ghost(back, a):
    """Pixels a behind-the-body pass changed (and that are visible) as their own layer."""
    changed = _neq(back, a) & (back[..., 3] > 0)
    g = np.zeros_like(back)
    g[changed] = back[changed]
    return g


def _under(front, ghost):
    """Alpha-composite `ghost` underneath `front`. Only pixels the ghost covers are blended; the
    rest equal `front` (with colour cleared where nothing is visible). Touches only the pixels
    that change (this sits under every echo layer, so it was a hot spot)."""
    if (front.dtype == np.uint8 and ghost.dtype == np.uint8 and front.shape == ghost.shape
            and front.ndim == 3 and front.shape[2] == 4):
        F = _FAST if _FAST is not None else _fast()
        if F:
            g = _c(ghost)
            if not F.any_alpha(g):
                return front
            out = np.empty(front.shape, np.uint8)
            F.under_u8(_c(front), g, out)
            return out
    gm = ghost[..., 3] > 0
    if not gm.any():
        return front
    out = front.copy()
    np.copyto(out[..., :3], 0, where=((front[..., 3] == 0) & ~gm)[..., None])
    rows, cols = np.flatnonzero(gm.any(axis=1)), np.flatnonzero(gm.any(axis=0))
    y0, y1, x0, x1 = rows[0], rows[-1] + 1, cols[0], cols[-1] + 1
    fb, m = front[y0:y1, x0:x1], gm[y0:y1, x0:x1]
    if front.dtype == np.uint8:                         # fully covered by front: front wins exactly
        m &= fb[..., 3] < 255
    else:                                               # (float: unless negatives need clamping)
        m &= (fb[..., 3] < 255) | (fb[..., :3].min(axis=2) < 0)
    if m.any():
        gb = ghost[y0:y1, x0:x1]
        if m.mean() > 0.25:          # mostly covered: blend the whole box at once, keep the masked
            np.copyto(out[y0:y1, x0:x1], _under_px(fb, gb), where=m[..., None])
        else:                        # sparse: only the masked pixels
            out[y0:y1, x0:x1][m] = _under_px(fb[m], gb[m])
    return out


def _under_px(f, g):
    """The blend of front pixels over ghost pixels, (..., 4) each: the same per-pixel arithmetic
    for a list of pixels or a whole box."""
    fa = f[..., 3].astype(np.float32) / 255.0
    ga = g[..., 3].astype(np.float32) / 255.0
    oa = fa + ga * (1 - fa)
    inv = np.where(oa > 0, 1.0 / np.maximum(oa, 1e-6), 0.0)
    rgb = (f[..., :3] * fa[..., None] + g[..., :3] * (ga * (1 - fa))[..., None]) * inv[..., None]
    blended = np.empty_like(f)
    if f.dtype == np.uint8:
        blended[..., :3] = np.clip(rgb + 0.5, 0, 255).astype(np.uint8)
        blended[..., 3] = np.clip(oa * 255 + 0.5, 0, 255).astype(np.uint8)
    else:
        blended[..., :3] = np.maximum(rgb, 0)
        blended[..., 3] = np.clip(oa * 255, 0, 255)
    return blended


def _rotate_rgba(arr, deg: float, center):
    """Rotate an RGBA array counter-clockwise by `deg` about `center`, premultiplied so edges
    don't pick up dark fringes. Only the disc around `center` that holds content is processed:
    content within radius r of the centre stays within r, everything else is empty."""
    h, w = arr.shape[:2]
    box = _content_box(arr)
    if box is None:
        return arr.copy()
    cx, cy = center
    y0, y1, x0, x1 = box
    r = math.ceil(max(math.hypot(x - cx, y - cy) for x in (x0, x1) for y in (y0, y1))) + 3
    X0, X1 = max(0, int(math.floor(cx - r))), min(w, int(math.ceil(cx + r)))
    Y0, Y1 = max(0, int(math.floor(cy - r))), min(h, int(math.ceil(cy + r)))
    if (X1 - X0) * (Y1 - Y0) < 0.9 * w * h:
        out = np.zeros_like(arr)
        out[Y0:Y1, X0:X1] = _rotate_rgba_full(arr[Y0:Y1, X0:X1], deg, (cx - X0, cy - Y0))
        return out
    return _rotate_rgba_full(arr, deg, center)


def _rotate_rgba_full(arr, deg: float, center):
    if arr.dtype == np.uint8:                     # one premultiplied 8-bit rotation (fast)
        im = Image.fromarray(np.ascontiguousarray(arr), "RGBA").convert("RGBa")
        return np.array(im.rotate(deg, resample=Image.BICUBIC, center=center).convert("RGBA"))
    # float images: rotate each premultiplied channel in 32-bit float, keeps HDR values
    al = np.clip(arr[..., 3:4], 0, 255) / 255.0
    pm = np.concatenate([arr[..., :3] * al, arr[..., 3:4]], -1).astype(np.float32)
    rot = np.stack([np.asarray(Image.fromarray(np.ascontiguousarray(pm[..., i]), "F")
                               .rotate(deg, resample=Image.BICUBIC, center=center)) for i in range(4)], -1)
    a = np.clip(rot[..., 3], 0, 255)
    rgb = np.where(a[..., None] > 0.5, rot[..., :3] * 255.0 / np.maximum(a[..., None], 1e-3), 0)
    # bicubic overshoot divided by a tiny edge alpha can blow up; nothing may exceed the input
    return np.dstack([np.clip(rgb, 0, float(arr[..., :3].max())), a]).astype(np.float32)


# --------------------------------------------------------------------------- DCT toolkit (jpeg sim)
_k, _n = np.arange(8)[:, None], np.arange(8)[None, :]
_C = np.sqrt(2 / 8) * np.cos((2 * _n + 1) * _k * np.pi / 16)
_C[0] /= np.sqrt(2)
_C = _C.astype(np.float32)
_FALL = (1.0 / (1 + np.add.outer(np.arange(8), np.arange(8)))).astype(np.float32)

_QY = np.array([16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55,
                14, 13, 16, 24, 40, 57, 69, 56, 14, 17, 22, 29, 51, 87, 80, 62,
                18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113, 92,
                49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99],
               np.float32).reshape(8, 8)
_QC = np.full((8, 8), 99, np.float32)
_QC[:4, :4] = np.array([[17, 18, 24, 47], [18, 21, 26, 66], [24, 26, 56, 99], [47, 66, 99, 99]])


@lru_cache(maxsize=512)
def _ac_tile(key):
    """Per-event AC noise for the JPEG sim (same values every frame, so cached).
    Note: numpy seeds keys that differ only by trailing zeros identically, so (seed,fx,k,0)
    shares this stream with that bucket's schedule; harmless here, but new keys should add a
    distinct non-zero tag rather than a trailing 0."""
    t = np.random.default_rng(key).normal(size=(8, 8, 3, 8, 8)).astype(np.float32)
    t.setflags(write=False)
    return t


def _qtabs(q: int):
    q = int(np.clip(q, 1, 100))
    s = 5000 / q if q < 50 else 200 - 2 * q
    f = lambda T: np.clip(np.floor((T * s + 50) / 100), 1, 255).astype(np.float32)
    return np.stack([f(_QY), f(_QC), f(_QC)])


def _dct(b):
    return _C @ b @ _C.T


def _idct(b):
    return _C.T @ b @ _C


def _rgb2ycc(x):
    r, g, b = x[..., 0], x[..., 1], x[..., 2]
    return np.stack([0.299 * r + 0.587 * g + 0.114 * b - 128,
                     -0.168736 * r - 0.331264 * g + 0.5 * b,
                     0.5 * r - 0.418688 * g - 0.081312 * b], -1)


def _ycc2rgb(x):
    y, cb, cr = x[..., 0] + 128, x[..., 1], x[..., 2]
    return np.stack([y + 1.402 * cr, y - 0.344136 * cb - 0.714136 * cr, y + 1.772 * cb], -1)


# --------------------------------------------------------------------------- effects
# Every effect: f(a: HxWx4 uint8, c: Ctx) -> HxWx4 uint8. Each event is placed either from its
# 2-D genome (ev.pl is None) or from its projected 3-D plane (ev.pl); the pixel work is shared.

def fx_jpeg(a, c: Ctx):
    evs = c.events("jpeg")
    if not evs:
        return a
    h, w = a.shape[:2]
    gh, gw = h // 8, w // 8
    vis = a[..., 3] > 0
    if gh < 2 or gw < 2 or not vis.any():
        return a
    x0, y0, x1, y1 = c.bbox
    bw, bh = x1 - x0, y1 - y0
    # JPEG has no alpha: fill transparent area with the mean visible colour (less ringing)
    base8 = a[:gh * 8, :gw * 8, :3].astype(np.float32)   # same values as a float64 where(), cast once
    np.copyto(base8, a[..., :3][vis].mean(0).astype(np.float32), where=~vis[:gh * 8, :gw * 8, None])
    blocks = base8.reshape(gh, 8, gw, 8, 3).transpose(0, 2, 1, 3, 4)       # (gh,gw,8,8,3) view
    Q = _qtabs(c.p.quality if c.p.quality else int(75 - 55 * c.p.intensity))
    dil = None                                           # the spill area, only if a run needs it
    out = a.copy()
    for ev in evs:
        e = _env(ev.u) * ev.g
        P, pl = ev.P, ev.pl
        if pl is None:
            i0 = (int(np.clip((y0 + P[1] * bh) // 8, 0, gh - 1)) * gw
                  + int(np.clip((x0 + P[0] * bw) // 8, 0, gw - 1)))
            rows = 1 + P[2] * 0.15 * bh / 8
            xwin = None
        else:                                           # run limited to the plane's footprint
            xwin = (int(max(0.0, pl.sx - pl.xr) // 8), int((pl.sx + pl.xr) // 8))
            i0 = int(np.clip(pl.sy // 8, 0, gh - 1)) * gw + int(np.clip(xwin[0], 0, gw - 1))
            rows = 1 + P[2] * 0.15 * bh / 8 * pl.s * pl.vy
        i1 = min(i0 + int(rows * gw * e), gh * gw)       # run grows/shrinks with e
        if i1 <= i0:
            continue
        idx = np.arange(i0, i1)
        by, bx = idx // gw, idx % gw
        if xwin is not None:
            keep = (bx >= xwin[0]) & (bx <= xwin[1])
            if not keep.any():
                continue
            idx, by, bx = idx[keep], by[keep], bx[keep]
        kind = int(P[3] * 4)
        if kind == 3:                                   # repeat a short run of blocks
            sidx = i0 + (idx - i0) % (1 + int(P[4] * 5))
            src = blocks[sidx // gw, sidx % gw]
        else:
            src = blocks[by, bx]
        co = _dct(np.moveaxis(_rgb2ycc(src), -1, 1))     # (n,3,8,8)
        if kind == 0:                                    # decoder ran dry: neutral gray
            co[:] = 0
        elif kind == 1:                                  # DC drift: whole run changes colour
            sh = np.array([(P[4] - .5) * 140, (P[5] - .5) * 320, (P[6] - .5) * 320]) * min(1.0, e * 1.5)
            co[:, :, 0, 0] += 8 * sh
        elif kind == 2:                                  # AC garbage, fixed per block position
            tile = _ac_tile(ev.key)
            co += tile[by % 8, bx % 8] * _FALL * (160 * e)
        if kind != 0:
            co = np.round(co / Q) * Q
        rgb = _ycc2rgb(np.moveaxis(_idct(co), 1, -1))
        rgb = np.clip(rgb, 0, 255) if a.dtype == np.uint8 else np.maximum(rgb, 0)  # float keeps HDR

        r0, r1 = int(by[0]), int(by[-1])
        nr = r1 - r0 + 1
        cb = blocks[r0:r1 + 1].copy()
        cb[by - r0, bx] = rgb
        mb = np.zeros((nr, gw), bool)
        mb[by - r0, bx] = True
        reg = cb.transpose(0, 2, 1, 3, 4).reshape(nr * 8, gw * 8, 3)
        mp = np.repeat(np.repeat(mb, 8, 0), 8, 1)
        ys, xs = slice(r0 * 8, (r1 + 1) * 8), slice(0, gw * 8)
        sub = out[ys, xs]                                # view
        inside = mp & vis[ys, xs]
        sub[..., :3][inside] = np.rint(reg[inside]) if a.dtype == np.uint8 else reg[inside]
        if c.bleed > 0:                                  # junk may spill past the silhouette
            if dil is None:
                dil = _dilate(vis, c.bleed)
            spill = mp & dil[ys, xs] & ~vis[ys, xs]
            if kind != 0:
                spill &= np.abs(reg - base8[ys, xs]).sum(2) > 90
            sub[..., :3][spill] = np.rint(reg[spill]) if a.dtype == np.uint8 else reg[spill]
            sub[..., 3][spill] = 255
    return out


def fx_blocks(a, c: Ctx):
    evs = c.events("blocks")
    if not evs:
        return a
    h, w = a.shape[:2]
    x0, y0, x1, y1 = c.bbox
    bw, bh = x1 - x0, y1 - y0
    amp = 0.4 + 0.6 * c.p.intensity
    out = a.copy()
    for ev in evs:
        if ev.g < 0.3:
            continue
        P, pl = ev.P, ev.pl
        j = 0.6 + 0.4 * c.held(ev, 0)
        if pl is None:
            bwid = min(w, max(2, int(bw * (0.04 + 0.20 * P[2]))))
            bhei = min(h, max(2, int(bh * (0.01 + 0.07 * P[3]))))
            sx = int(np.clip(x0 + P[0] * max(1, bw - bwid), 0, w - bwid))
            sy = int(np.clip(y0 + P[1] * max(1, bh - bhei), 0, h - bhei))
            dx = int(np.clip(sx + (P[4] - .5) * 5 * bwid * amp * j, 0, w - bwid))
            dy = int(np.clip(sy + (P[5] - .5) * 5 * bhei * amp * j, 0, h - bhei))
        else:
            wl, hl = bw * (0.04 + 0.20 * P[2]), bh * (0.01 + 0.07 * P[3])   # size on the plane
            bwid = min(w, max(2, int(wl * abs(pl.ax) * pl.s)))
            bhei = min(h, max(2, int(hl * pl.s * pl.vy)))
            sx = int(np.clip(pl.sx - bwid / 2, 0, w - bwid))
            sy = int(np.clip(pl.sy - bhei / 2, 0, h - bhei))
            u = (P[4] - .5) * 5 * wl * amp * j                       # offset along the plane
            v = (P[5] - .5) * 5 * hl * amp * j
            dx = int(np.clip(sx + pl.s * u * (pl.ax + pl.rho * pl.nx), 0, w - bwid))
            dy = int(np.clip(sy + pl.s * v * pl.vy, 0, h - bhei))
        blk = a[sy:sy + bhei, sx:sx + bwid].copy()
        mode = int(P[6] * 4)
        if mode == 1:
            blk[..., :3] = blk[..., [1, 2, 0]]
        elif mode == 2:
            blk[..., :3] = 255 - blk[..., :3] if blk.dtype == np.uint8 else np.maximum(255 - blk[..., :3], 0)
        elif mode == 3:                                       # posterize to 4 levels per channel
            blk[..., :3] = (blk[..., :3] >> 6) << 6 if blk.dtype == np.uint8 else np.floor(blk[..., :3] / 64) * 64
        dst = out[dy:dy + bhei, dx:dx + bwid]
        if mode == 0:
            dst[...] = blk
        else:
            m = blk[..., 3] > 0
            dst[m] = blk[m]
    return out


def fx_slices(a, c: Ctx):
    evs = c.events("slices")
    if not evs:
        return a
    h, w = a.shape[:2]
    x0, y0, x1, y1 = c.bbox
    bw, bh = x1 - x0, y1 - y0
    amp = 0.4 + 0.6 * c.p.intensity
    out = a.copy()
    for ev in evs:
        P, pl = ev.P, ev.pl
        sign = -1 if P[3] < 0.5 else 1
        u = (sign * bw * (0.02 + 0.14 * P[2]) * amp * _env(ev.u) * ev.g
             * (0.35 + 0.65 * c.held(ev, 0)))
        if pl is None:
            sh = max(1, int(bh * (0.003 + 0.04 * P[1] ** 2)))
            y = int(np.clip(y0 + P[0] * max(1, bh - sh), 0, h - 1))
            sh = min(sh, h - y)
            dx = int(u)
            if dx:
                out[y:y + sh] = _shift(a[y:y + sh], dx, 0)
        else:
            sh = max(1, int(bh * (0.003 + 0.04 * P[1] ** 2) * pl.s * pl.vy))
            y = int(np.clip(pl.sy - sh / 2, 0, h - 1))
            sh = min(sh, h - y)
            dx = int(pl.s * u * (pl.ax + pl.rho * pl.nx))          # projected 3-D displacement
            xa = int(np.clip(pl.sx - pl.xr, 0, w))
            xb = int(np.clip(pl.sx + pl.xr, 0, w))
            if dx and xb - xa >= 2:
                _slice_window(out, a, y, sh, dx, xa, xb)
    return out


def fx_wave(a, c: Ctx):
    evs = c.events("wave")
    if not evs:
        return a
    h, w = a.shape[:2]
    x0, y0, x1, y1 = c.bbox
    bw, bh = x1 - x0, y1 - y0
    amp = 0.4 + 0.6 * c.p.intensity
    out = a.copy()
    cols_all = np.arange(w)[None, :]
    for ev in evs:
        P, pl = ev.P, ev.pl
        if pl is None:
            hb = max(8, int(bh * (0.08 + 0.25 * P[1])))
            y = int(np.clip(y0 + P[0] * max(1, bh - hb), 0, h - 1))
        else:
            if pl.xr < 2:
                continue
            hb = max(8, int(bh * (0.08 + 0.25 * P[1]) * pl.s * pl.vy))
            y = int(np.clip(pl.sy - hb / 2, 0, h - 1))
        hb = min(hb, h - y)
        if hb < 4:
            continue
        ys = np.arange(hb)
        A = bw * (0.01 + 0.045 * P[2]) * _env(ev.u) * ev.g * amp
        period = hb * (0.2 + 0.6 * P[3])
        speed = (0.4 + 2.0 * P[5]) * (1 if P[6] < 0.5 else -1)          # cycles per second
        phase = TAU * (P[4] + speed * c.t)                               # scrolls smoothly in time
        win = np.sin(np.pi * ys / max(1, hb - 1))                        # fades at band edges
        if pl is None:
            dx = np.round(A * np.sin(TAU * ys / period + phase) * win).astype(int)[:, None]
        else:
            A = A * pl.s * (pl.ax + pl.rho * pl.nx)
            wx = np.clip(1 - ((cols_all[0] - pl.sx) / pl.xr) ** 2, 0, 1)   # fades at plane sides
            dx = np.round((A * np.sin(TAU * ys / period + phase) * win)[:, None] * wx[None, :]).astype(int)
        cols = np.clip(cols_all - dx, 0, w - 1)
        out[y:y + hb] = out[y:y + hb][ys[:, None], cols]
    return out


def fx_sort(a, c: Ctx):
    evs = c.events("sort")
    if not evs:
        return a
    h, w = a.shape[:2]
    x0, y0, x1, y1 = c.bbox
    bw, bh = x1 - x0, y1 - y0
    out = a.copy()
    for ev in evs:
        P, pl = ev.P, ev.pl
        if pl is None:
            bwid = max(1, int(bw * (0.01 + 0.07 * P[1])))
            xa = int(np.clip(x0 + P[0] * max(1, bw - bwid), 0, w - bwid))
            ya = int(np.clip(y0 + P[2] * 0.8 * bh, 0, h - 1))
            L = int(bh * (0.1 + 0.4 * P[3]) * _env(ev.u) * ev.g)       # sorted run grows/retracts
        else:
            bwid = max(1, int(bw * (0.01 + 0.07 * P[1]) * abs(pl.ax) * pl.s))
            xa = int(np.clip(pl.sx - bwid / 2, 0, w - bwid))
            L = int(bh * (0.1 + 0.4 * P[3]) * _env(ev.u) * ev.g * pl.s * pl.vy)
            ya = int(np.clip(pl.sy - L / 2, 0, h - 1))
        yb = min(h, ya + L)
        if yb - ya < 2:
            continue
        seg = out[ya:yb, xa:xa + bwid]                               # view
        v = seg[..., 3] > 0
        lum = _lum(seg[..., :3])
        key = np.where(v, -lum if P[4] < 0.5 else lum, np.inf)
        order = np.argsort(key, axis=0, kind="stable")               # visible pixels first, by lum
        srt = np.take_along_axis(seg, order[..., None], axis=0)
        k = np.clip(np.cumsum(v, axis=0) - 1, 0, seg.shape[0] - 1)   # k-th visible slot gets k-th sorted px
        placed = np.take_along_axis(srt, k[..., None], axis=0)
        seg[...] = np.where(v[..., None], placed, seg)               # silhouette unchanged
    return out


def fx_smear(a, c: Ctx):
    evs = c.events("smear")
    if not evs:
        return a
    h, w = a.shape[:2]
    x0, y0, x1, y1 = c.bbox
    bw, bh = x1 - x0, y1 - y0
    amp = 0.4 + 0.6 * c.p.intensity
    out = a.copy()
    for ev in evs:
        P, pl = ev.P, ev.pl
        if pl is None:
            width = max(1, int(bw * (0.002 + 0.012 * P[1] ** 2)))
            x = int(np.clip(x0 + P[0] * bw, 0, w - width))
            L = int(bh * (0.08 + 0.5 * P[5]) * amp * _env(ev.u) * ev.g)
        else:
            width = max(1, int(bw * (0.002 + 0.012 * P[1] ** 2) * pl.s))
            x = int(np.clip(pl.sx - width / 2, 0, w - width))
            L = int(bh * (0.08 + 0.5 * P[5]) * amp * _env(ev.u) * ev.g * pl.s * pl.vy)
        if L < 2:
            continue
        up = P[2] < 0.25
        rows = np.flatnonzero(a[:, x:x + width, 3].max(axis=1) > 128)
        if rows.size == 0:
            continue
        if P[3] < 0.7:                                    # start at the silhouette edge (tracks the subject)
            ysrc = int(rows.min() if up else rows.max())
        else:                                             # start at a fixed screen row
            ysrc = int(rows[min(rows.size - 1, int(P[4] * rows.size))])
        row = a[ysrc, x:x + width].astype(np.float32)     # (width,4)
        if up:
            ya, yb = max(0, ysrc - L), ysrc
        else:
            ya, yb = ysrc + 1, min(h, ysrc + 1 + L)
        n = yb - ya
        if n < 1:
            continue
        fade = np.linspace(1.0, 0.15, L)[:n] if not up else np.linspace(0.15, 1.0, L)[-n:]
        streak = np.broadcast_to(row, (n, width, 4)).copy()
        streak[..., 3] *= fade[:, None]
        dst = out[ya:yb, x:x + width]                      # view
        da = dst[..., 3].astype(np.float32)
        under = da > 0                                      # streak crosses visible pixels:
        new = streak.astype(dst.dtype)                      # draw over them instead of cutting a
        if under.any():                                     # see-through hole into the body
            s, d = streak[under], dst[under].astype(np.float32)
            sa = s[:, 3] / 255.0
            oa = s[:, 3] + d[:, 3] * (1 - sa)
            rgb = (s[:, :3] * s[:, 3:4] + d[:, :3] * (d[:, 3:4] * (1 - sa[:, None]))) / np.maximum(oa, 1e-6)[:, None]
            if dst.dtype == np.uint8:
                new[under] = np.concatenate([np.clip(rgb + 0.5, 0, 255), np.clip(oa + 0.5, 0, 255)[:, None]], 1).astype(np.uint8)
            else:
                new[under] = np.concatenate([np.maximum(rgb, 0), np.clip(oa, 0, 255)[:, None]], 1)
        dst[...] = new
    return out


def fx_split(a, c: Ctx):
    h, w = a.shape[:2]
    ph = _u(c.p.seed, FX_ID["split"], 0, 0) * TAU
    gate = min(1.0, c.p.intensity * 5) * c.gain("split")
    amp = w * (0.002 + 0.02 * c.p.intensity) * (0.35 + 0.65 * c.act) * gate   # 0 at intensity 0
    dxf = amp * (0.7 + 0.3 * math.sin(TAU * 0.7 * c.t + ph))
    pyf = 0.0
    if c.space is not None:            # channels as depth layers: R in front, B behind -> parallax
        sp = c.space
        K = (c.bbox[2] - c.bbox[0]) * (0.015 + 0.05 * c.p.intensity)
        pxf, pyf = K * sp.right[2] * gate, K * sp.up[2] * gate        # screen image of a depth step
        if sp.roll:
            cr, sr = math.cos(sp.roll), math.sin(sp.roll)
            pxf, pyf = pxf * cr - pyf * sr, pxf * sr + pyf * cr
        dxf += pxf
    dx = int(round(dxf))
    dy = int(round(int(round(amp * (0.7 + 0.3 * math.sin(TAU * 0.7 * c.t + ph))))
                   * 0.25 * math.sin(TAU * 0.31 * c.t + ph * 1.7) + pyf))
    if dx == 0 and dy == 0:
        return a
    offs = {0: (dx, dy), 1: (0, 0), 2: (-dx, -dy)}
    out = np.zeros_like(a)
    box = _content_box(a, max(abs(dx), abs(dy)) + 1)
    if box is None:
        return out
    y0, y1, x0, x1 = box                                  # the canvas edge still clips as before
    A, O = a[y0:y1, x0:x1], out[y0:y1, x0:x1]
    ex = lambda arr, ox, oy: _shift_in(arr, ox, oy, a, (y0, y1, x0, x1))
    alpha = A[..., 3]
    new_alpha = alpha.astype(np.float32)
    for ch, (ox, oy) in offs.items():
        O[..., ch] = ex(a[..., ch], ox, oy)
        if (ox, oy) != (0, 0):                            # fringe: only in the shifted channel
            new_alpha = np.maximum(new_alpha, ex(a[..., 3], ox, oy).astype(np.float32) * 0.75)
    O[..., 3] = new_alpha.astype(a.dtype)
    return out


def _shift_in(chan, dx: int, dy: int, full, box):
    """_shift(chan, dx, dy) evaluated only inside `box` (y0, y1, x0, x1) of the full canvas."""
    y0, y1, x0, x1 = box
    h, w = chan.shape
    out = np.zeros((y1 - y0, x1 - x0), chan.dtype)
    sy0, sy1 = max(y0 - dy, 0), min(y1 - dy, h)            # source rows that land inside the box
    sx0, sx1 = max(x0 - dx, 0), min(x1 - dx, w)
    if sy1 > sy0 and sx1 > sx0:
        out[sy0 + dy - y0:sy1 + dy - y0, sx0 + dx - x0:sx1 + dx - x0] = chan[sy0:sy1, sx0:sx1]
    return out


def fx_tint(a, c: Ctx):
    amount = min(1.0, c.p.tint * c.gain("tint"))
    if amount <= 0:
        return a
    dark = np.array([40, 10, 90], np.float32)
    light = np.array([120, 255, 240], np.float32)
    out = a.copy()
    vis = a[..., 3] > 0                          # invisible pixels keep colour 0 (was tinted, unseen)
    rgb = a[..., :3][vis].astype(np.float32)
    t = (_lum(rgb) / 255.0)[..., None]
    res = rgb * (1 - amount) + (dark * (1 - t) + light * t) * amount
    out[..., :3][vis] = (np.clip(res, 0, 255).astype(np.uint8) if a.dtype == np.uint8 else np.maximum(res, 0))
    return out


def fx_scanlines(a, c: Ctx):
    h = a.shape[0]
    period = max(2, round(h / 300))
    factor = np.ones(h, np.float32)
    gs = c.gain("scanlines")
    if gs <= 0:
        return a
    factor[::period] = max(0.0, 1 - (0.15 + 0.35 * c.p.intensity) * gs)
    factor *= max(0.0, 1 - 0.06 * c.p.intensity * gs * _vnoise(c.p.seed, 2, c.t * 18))   # faint flicker
    out = a.copy()
    box = _content_box(a)
    if box is not None:                          # zeros stay zero; skip the empty canvas
        y0, y1, x0, x1 = box
        out[y0:y1, x0:x1, :3] = (a[y0:y1, x0:x1, :3].astype(np.float32)
                                 * factor[y0:y1, None, None]).astype(a.dtype)
    return out


def fx_noise(a, c: Ctx):
    gn = c.gain("noise")
    if gn <= 0:
        return a
    vis = a[..., 3] > 0
    rng = np.random.default_rng((c.p.seed, FX_ID["noise"], 0, c.fi))     # fresh grain per frame
    n = rng.normal(0, (3 + 10 * c.p.intensity) * gn, size=a.shape[:2])   # full size: same grain
    F = _fast() if a.dtype == np.uint8 and a.shape[2] == 4 else False
    if F:
        out = np.empty(a.shape, np.uint8)
        F.noise_u8(_c(a), _c(n), out)
        return out
    out = a.copy()
    res = a[..., :3][vis].astype(np.float32) + n[vis].astype(np.float32)[:, None]
    out[..., :3][vis] = (np.clip(res, 0, 255).astype(np.uint8) if a.dtype == np.uint8 else np.maximum(res, 0))
    return out


# --------------------------------------------------------------------------- time effects
_PREP: "OrderedDict" = None        # prepared (converted + padded) history frames, per process
_PREP_BYTES = [0]
_PREP_BUDGET = 160_000_000       # measured: a bigger cache didn't render any faster, it only
                                 # multiplied memory by the number of workers


class _History:
    """Earlier *source* frames (never earlier glitched output, so frames stay independent),
    loaded on demand and converted + padded exactly like the current frame. get(0) is the current
    frame. `src(k)` returns the image k frames back or None; src.key(k), if present, names it for
    the motion cache."""

    def __init__(self, src, prep, current, cur_key=None, frame=None):
        self.src, self.prep, self.cur_key, self.frame = src, prep, cur_key, frame
        self.shape, self.dtype = current.shape, current.dtype.str
        self._c = {0: current}

    def get(self, k: int):
        if k not in self._c:
            arr = None
            if 1 <= k <= MAX_BACK:
                try:
                    raw = self.src(k) if callable(self.src) else (self.src[k - 1] if k - 1 < len(self.src) else None)
                except (OSError, ValueError, IndexError):
                    raw = None
                arr = self._prepared(k, raw) if raw is not None else None
            self._c[k] = arr
        return self._c[k]

    def _prepared(self, k, raw):
        global _PREP
        if _PREP is None:
            from collections import OrderedDict
            _PREP = OrderedDict()
        sk = self.key(k)
        ck = (sk, self.shape, self.dtype) if sk is not None else None
        if ck is not None and ck in _PREP:
            _PREP.move_to_end(ck)
            return _PREP[ck]
        arr = self.prep(raw)
        if arr.shape != self.shape:                         # a frame of another size: not usable
            return None
        arr.flags.writeable = False                         # shared between frames: never edit
        if ck is not None:
            _PREP[ck] = arr
            _PREP_BYTES[0] += arr.nbytes
            while _PREP_BYTES[0] > _PREP_BUDGET and len(_PREP) > 1:
                _k, old = _PREP.popitem(last=False)
                _PREP_BYTES[0] -= old.nbytes
        return arr

    def back(self, frames: int) -> int:
        """Sequence steps back that cover `frames` frame numbers. Effects time their look-back
        in frames; with gaps in the numbering (--use-numbers on every 2nd frame) the history
        holds fewer steps than frames. Without frame numbers: one step per frame."""
        nums = getattr(self.src, "frames", None)
        if not nums or self.frame is None or frames <= 0:
            return max(0, int(frames))
        return int(sum(1 for f in nums if f >= self.frame - frames))

    def depth(self, want: int) -> int:
        """Largest k <= want with a frame available (0 if none)."""
        for k in range(min(want, MAX_BACK), 0, -1):
            if self.get(k) is not None:
                return k
        return 0

    def key(self, k: int):
        if k == 0:
            # The frame being rendered is never cached: it may already be changed (a transition
            # dissolves it before the time effects run), while the same file seen from a later
            # frame is the clean source. One cache name for both made the result depend on which
            # frames a worker had rendered before (1 worker != 4 workers).
            return None
        f = getattr(self.src, "key", None)
        return f(k) if f else None


def _band(c: Ctx, ev, wide: float, full_p: float, lo: float, hi: float, a):
    """(ya, yb, xa, xb) region for a time effect: a horizontal band across the subject box (widened
    by `wide` box-widths each side), or with probability full_p the whole box."""
    h, w = a.shape[:2]
    x0, y0, x1, y1 = c.bbox
    bw, bh = x1 - x0, y1 - y0
    P = ev.P
    if P[2] < full_p:
        ya, yb = y0 - 0.08 * bh, y1 + 0.08 * bh
    else:
        hb = bh * (lo + (hi - lo) * P[1])
        ya = y0 + P[0] * max(1.0, bh - hb)
        yb = ya + hb
    xa, xb = x0 - wide * bw, x1 + wide * bw
    ya, yb = int(np.clip(ya, 0, h)), int(np.clip(yb, 0, h))
    xa, xb = int(np.clip(xa, 0, w)), int(np.clip(xb, 0, w))
    return ya, yb, xa, xb


def _scaled_alpha(e, f: float):
    out = e.copy()
    out[..., 3] = (e[..., 3].astype(np.float32) * f).astype(e.dtype)
    return out


def _offset_window(img, ya, yb, xa, xb, dy: int, dx: int):
    """img[ya:yb, xa:xb] as if img were shifted by (dy, dx); outside the canvas is transparent."""
    out = np.zeros((yb - ya, xb - xa, img.shape[2]), img.dtype)
    h, w = img.shape[:2]
    sy0, sy1 = max(ya - dy, 0), min(yb - dy, h)
    sx0, sx1 = max(xa - dx, 0), min(xb - dx, w)
    if sy1 > sy0 and sx1 > sx0:
        out[sy0 + dy - ya:sy1 + dy - ya, sx0 + dx - xa:sx1 + dx - xa] = img[sy0:sy1, sx0:sx1]
    return out


def fx_echo(a, c: Ctx):
    """Ghost trails: fading copies of earlier frames behind (sometimes over) the body, drifting a
    little so they read even when the subject stands still."""
    if c.hist is None:
        return a
    evs = c.events2d("echo")
    if not evs:
        return a
    x0, y0, x1, y1 = c.bbox
    bw, bh = x1 - x0, y1 - y0
    amp = 0.4 + 0.6 * c.p.intensity
    out = a
    for ev in evs:
        P = ev.P
        st = _env(ev.u) * ev.g
        if st < 0.05:
            continue
        n = 2 + int(P[3] * 3)                               # 2-4 echoes
        step = 1 + int(P[4] * 3)                            # 1-3 frames apart
        lags = [k for k in (c.hist.back(step * (i + 1)) for i in range(n)) if k > 0 and c.hist.depth(k) == k]
        lags = list(dict.fromkeys(lags))                    # gaps can map two lags to one frame
        if not lags:
            continue
        ya, yb, xa, xb = _band(c, ev, 0.5, 0.3, 0.2, 0.8, a)
        if yb - ya < 2 or xb - xa < 2:
            continue
        ddx = (P[5] - 0.5) * 0.08 * bw * amp
        ddy = (P[6] - 0.5) * 0.03 * bh * amp
        mode = int(P[7] * 4)                                # 0/1 natural, 2 cyan, 3 magenta
        ghost = np.zeros((yb - ya, xb - xa, 4), a.dtype)
        layers = []
        for i in reversed(range(len(lags))):                # far echoes first, near ones on top
            e = _offset_window(c.hist.get(lags[i]), ya, yb, xa, xb,
                               int(round(ddy * (i + 1))), int(round(ddx * (i + 1))))
            if mode == 2:
                e[..., 0] = (e[..., 0] * 0.15).astype(e.dtype)
            elif mode == 3:
                e[..., 1] = (e[..., 1] * 0.15).astype(e.dtype)
            e = _scaled_alpha(e, min(1.0, st * 0.85 * (1 - i / (len(lags) + 1))))
            ghost = _under(e, ghost)
            layers.append(e)
        if out is a:
            out = a.copy()
        sl = _under(out[ya:yb, xa:xb], ghost)               # trails behind the current body
        if P[0] > 0.7 and layers:                           # and the nearest one over it, faintly
            sl = _under(_scaled_alpha(layers[-1], 0.4), sl)
        out[ya:yb, xa:xb] = sl
    return out


def fx_stutter(a, c: Ctx):
    """Frame hold: a block / band / the whole box freezes on an older frame, optionally jumping
    forward every few frames (dropped packets), sometimes displaced sideways."""
    if c.hist is None:
        return a
    evs = c.events2d("stutter")
    if not evs:
        return a
    h, w = a.shape[:2]
    x0, y0, x1, y1 = c.bbox
    bw, bh = x1 - x0, y1 - y0
    amp = 0.4 + 0.6 * c.p.intensity
    out = a
    for ev in evs:
        if ev.g < 0.3:
            continue
        P = ev.P
        el = c.fi - c.start_frame(ev)
        period = 2 + int(P[4] * 7)
        lag = el % period if P[7] > 0.35 else el            # re-latch every `period` frames or hold
        lag = c.hist.depth(c.hist.back(lag))
        if lag < 1:
            continue
        old = c.hist.get(lag)
        if P[5] < 0.1:                                       # whole subject freezes
            ya, yb, xa, xb = _band(c, ev, 0.15, 1.1, 0, 0, a)
        elif P[5] < 0.4:                                     # a band across the subject
            ya, yb, xa, xb = _band(c, ev, 0.3, 0.0, 0.04, 0.3, a)
        else:                                                # a block
            rw, rh = bw * (0.12 + 0.6 * P[1]), bh * (0.03 + 0.25 * P[3])
            xa = int(np.clip(x0 + P[0] * max(1.0, bw - rw), 0, w))
            ya = int(np.clip(y0 + P[6] * max(1.0, bh - rh), 0, h))
            xb, yb = int(np.clip(xa + rw, 0, w)), int(np.clip(ya + rh, 0, h))
        if yb - ya < 1 or xb - xa < 1:
            continue
        dx = int(round((c.held(ev, 1, period) - 0.5) * 0.25 * bw * amp)) if P[6] > 0.75 else 0
        if out is a:
            out = a.copy()
        out[ya:yb, xa:xb] = _offset_window(old, ya, yb, xa, xb, 0, dx)
    return out


_MB, _MR = 8, 3            # motion search: block size and radius, in low-res pixels
_MOTION: "OrderedDict" = None


_FEATS: "OrderedDict" = None


def _feat(a, s: int, key=None):
    """Low-res matching image: luminance where visible plus some alpha, box-averaged by s."""
    global _FEATS
    if _FEATS is None:
        from collections import OrderedDict
        _FEATS = OrderedDict()
    ck = (key, a.shape, a.dtype.str, s) if key is not None else None
    if ck is not None and ck in _FEATS:
        _FEATS.move_to_end(ck)
        return _FEATS[ck]
    h, w = (a.shape[0] // s) * s, (a.shape[1] // s) * s
    if a.dtype == np.uint8:                                 # Pillow's box reduce: fast C
        sm = np.asarray(Image.fromarray(np.ascontiguousarray(a[:h, :w])).reduce(s), np.float32)
    else:
        sm = a[:h, :w].reshape(h // s, s, w // s, s, 4).mean(axis=(1, 3), dtype=np.float32)
    al = sm[..., 3]
    f = (0.8 * np.minimum(_lum(sm[..., :3]), 255) * (al / 255) + 0.2 * al).astype(np.float32)
    if ck is not None:
        _FEATS[ck] = f
        while len(_FEATS) > 64:
            _FEATS.popitem(last=False)
    return f


def _block_motion(prev, nxt, s: int, keys=(None, None)):
    """Backward block motion between two frames: for every block of `nxt` the offset (dy, dx), in
    full-res px, where its content sat in `prev` (nxt(x) ~ prev(x + d)). Exhaustive search on a
    downscaled luminance+alpha image, with a small bias toward no motion."""
    fp, fn = _feat(prev, s, keys[0]), _feat(nxt, s, keys[1])
    by, bx = fn.shape[0] // _MB, fn.shape[1] // _MB
    fn = fn[:by * _MB, :bx * _MB]
    R = _MR
    vy = np.zeros((by, bx), np.int32)
    vx = np.zeros((by, bx), np.int32)
    # blocks whose search window is empty in both frames always pick (0, 0): skip them
    ys, xs = np.nonzero((fp[:by * _MB, :bx * _MB] > 0) | (fn > 0))
    if ys.size == 0:
        return vy * s, vx * s, _MB * s
    b0, b1 = max(0, (ys.min() - R) // _MB), min(by, (ys.max() + R) // _MB + 1)
    c0, c1 = max(0, (xs.min() - R) // _MB), min(bx, (xs.max() + R) // _MB + 1)
    nb, nc = b1 - b0, c1 - c0
    pp = np.pad(fp, R, mode="edge")
    fw = fn[b0 * _MB:b1 * _MB, c0 * _MB:c1 * _MB]
    best = np.full((nb, nc), np.inf, np.float32)
    wy = np.zeros((nb, nc), np.int32)
    wx = np.zeros((nb, nc), np.int32)
    for dy in range(-R, R + 1):
        for dx in range(-R, R + 1):
            sh = pp[R + dy + b0 * _MB:R + dy + b1 * _MB, R + dx + c0 * _MB:R + dx + c1 * _MB]
            sad = np.abs(fw - sh).reshape(nb, _MB, nc, _MB).sum(axis=(1, 3)) + 0.5 * _MB * _MB * (abs(dy) + abs(dx))
            m = sad < best
            best[m], wy[m], wx[m] = sad[m], dy, dx
    vy[b0:b1, c0:c1], vx[b0:b1, c0:c1] = wy, wx
    return vy * s, vx * s, _MB * s


def _motion(hist: _History, k: int, s: int):
    """Motion from frame k+1 (older) to frame k, cached per pair of source files."""
    global _MOTION
    if _MOTION is None:
        from collections import OrderedDict
        _MOTION = OrderedDict()
    ka, kb = hist.key(k + 1), hist.key(k)
    key = (ka, kb, hist.shape, s) if ka is not None and kb is not None else None
    if key is not None and key in _MOTION:
        _MOTION.move_to_end(key)
        return _MOTION[key]
    res = _block_motion(hist.get(k + 1), hist.get(k), s, (ka, kb))
    if key is not None:
        _MOTION[key] = res
        while len(_MOTION) > 96:
            _MOTION.popitem(last=False)
    return res


def fx_mosh(a, c: Ctx):
    """Datamosh: inside a band the picture stops refreshing - the frame from when the event began
    is dragged along by every later frame's block motion (P-frames without their I-frame), so
    colour and silhouette smear with the movement; it sits over the live frame. Ends with a hard
    cut, like a new keyframe."""
    if c.hist is None:
        return a
    evs = c.events2d("mosh")
    if not evs:
        return a
    h, w = a.shape[:2]
    s = max(1, int(round(max(h, w) / 384)))
    out = a
    for ev in evs:
        if ev.g < 0.3:
            continue
        P = ev.P
        lag = c.hist.depth(c.hist.back(min(c.fi - c.start_frame(ev), 16)))
        if lag < 1:
            continue
        ya, yb, xa, xb = _band(c, ev, 0.35, 0.25, 0.15, 0.7, a)
        if yb - ya < 2 or xb - xa < 2:
            continue
        gain = 1.0 + 1.5 * c.p.intensity * (P[6] - 0.6) / 0.4 if P[6] > 0.6 else 1.0   # 'bloom'
        m = int(min(lag, 8) * _MR * s * gain) + 2
        cy0, cy1, cx0, cx1 = max(0, ya - m), min(h, yb + m), max(0, xa - m), min(w, xb + m)
        base = np.array(c.hist.get(lag), order="C")          # a writable copy
        Y = np.arange(cy0, cy1)
        X = np.arange(cx0, cx1)
        F = _fast() if base.dtype in (np.uint8, np.float32) and base.shape[2] == 4 else False
        tmp = np.empty((cy1 - cy0, cx1 - cx0, 4), base.dtype) if F else None
        for k in range(lag - 1, -1, -1):                     # replay motion up to the current frame
            vy, vx, bs = _motion(c.hist, k, s)
            if F and vy.dtype == np.int32 and vx.dtype == np.int32:
                F.replay(base, _c(vy), _c(vx), int(bs), int(cy0), int(cy1), int(cx0), int(cx1),
                         float(gain), int(h), int(w), tmp)
                continue
            iy = np.clip(Y // bs, 0, vy.shape[0] - 1)
            ix = np.clip(X // bs, 0, vy.shape[1] - 1)
            dy = np.round(vy[iy][:, ix] * gain).astype(np.int64)
            dx = np.round(vx[iy][:, ix] * gain).astype(np.int64)
            sy = np.clip(Y[:, None] + dy, 0, h - 1)
            sx = np.clip(X[None, :] + dx, 0, w - 1)
            base[cy0:cy1, cx0:cx1] = base[sy, sx]
        if out is a:
            out = a.copy()
        # over the live frame: smeared colour on top, gaps (motion-search misses on the
        # silhouette) show the current body instead of punching see-through holes
        out[ya:yb, xa:xb] = _under(base[ya:yb, xa:xb], out[ya:yb, xa:xb])
    return out


_FX = {"jpeg": fx_jpeg, "blocks": fx_blocks, "slices": fx_slices, "wave": fx_wave,
       "sort": fx_sort, "smear": fx_smear, "split": fx_split, "tint": fx_tint,
       "scanlines": fx_scanlines, "noise": fx_noise, "echo": fx_echo, "stutter": fx_stutter,
       "mosh": fx_mosh}


# --------------------------------------------------------------------------- pose (3D mode)
class Pose:
    """Subject yaw / pitch / roll per frame from sparse keys: interpolated, optionally smoothed.

    Values are degrees and taken literally like animation curves (0 -> 360 is a full turn);
    unwrap=True treats yaw jumps > 180 between keys as wrap-around (tracker output)."""

    def __init__(self, frames, yaws_deg, fps: float = 25.0, smooth: float = 0.0,
                 unwrap: bool = False, pitch_deg=None, roll_deg=None, scale=None, cam_dist=None):
        f = np.asarray(frames, float)
        order = np.argsort(f, kind="stable")
        f = f[order]
        self.f0 = int(math.floor(f[0]))
        self.grid = np.arange(self.f0, int(math.ceil(f[-1])) + 1, dtype=float)
        sig = smooth * fps

        def dense(vals, unwrap_=False):
            if vals is None:
                return np.zeros_like(self.grid)
            y = np.radians(np.asarray(vals, float)[order])
            ok = np.isfinite(y)                           # missing cells: interpolate across them
            if not ok.any():
                return np.zeros_like(self.grid)
            y, fk = y[ok], f[ok]
            if unwrap_:
                y = np.unwrap(y)
            d = np.interp(self.grid, fk, y)
            if sig > 0.3 and d.size > 1:
                r = int(3 * sig) + 1
                k = np.exp(-0.5 * (np.arange(-r, r + 1) / sig) ** 2)
                d = np.convolve(np.pad(d, r, mode="edge"), k / k.sum(), mode="valid")
            return d
        self.dense = dense(yaws_deg, unwrap)
        self.dense_pitch = dense(pitch_deg)
        self.dense_roll = dense(roll_deg)

        def dense_lin(vals, log=False):        # not angles: no radians; scale interpolates in log
            v = np.asarray(vals, float)[order]
            ok = np.isfinite(v)
            v, fk = v[ok], f[ok]
            v = np.log(np.maximum(v, 1e-6)) if log else v
            d = np.interp(self.grid, fk, v)
            if sig > 0.3 and d.size > 1:
                r = int(3 * sig) + 1
                k = np.exp(-0.5 * (np.arange(-r, r + 1) / sig) ** 2)
                d = np.convolve(np.pad(d, r, mode="edge"), k / k.sum(), mode="valid")
            return np.exp(d) if log else d
        self.has_scale, self.has_cam = scale is not None, cam_dist is not None
        self.dense_scale = dense_lin(scale, log=True) if scale is not None else None
        self.dense_cam = dense_lin(cam_dist) if cam_dist is not None else None

    def _at(self, arr, frame):
        return math.degrees(float(np.interp(frame, self.grid, arr)))

    def yaw_deg(self, frame: float) -> float:
        return self._at(self.dense, frame)

    def pitch_deg(self, frame: float) -> float:
        return self._at(self.dense_pitch, frame)

    def roll_deg(self, frame: float) -> float:
        return self._at(self.dense_roll, frame)

    def angles(self, frame: float):
        return self.yaw_deg(frame), self.pitch_deg(frame), self.roll_deg(frame)

    def scale(self, frame: float) -> float:
        return float(np.interp(frame, self.grid, self.dense_scale)) if self.has_scale else 1.0

    def cam_dist(self, frame: float) -> float | None:
        return float(np.interp(frame, self.grid, self.dense_cam)) if self.has_cam else None

    @classmethod
    def from_file(cls, path, fps: float = 25.0, smooth: float = 0.0,
                  unwrap: bool = False) -> "Pose":
        """Rows of numbers. With a header line, columns are matched by name (frame, yaw, pitch,
        roll, scale, cam_dist; others such as confidence are ignored). Without one:
        frame, yaw[, pitch[, roll]]."""
        cols = None
        data = {"frame": [], "yaw": [], "pitch": [], "roll": [], "scale": [], "cam_dist": []}
        for no, ln in enumerate(Path(path).read_text(encoding="utf-8-sig").splitlines(), 1):
            ln = ln.split("#")[0].strip()
            if not ln:
                continue
            # one delimiter per line, empty cells kept as missing (pandas writes NaN as "")
            if "," in ln or ";" in ln:
                parts = [x.strip() for x in ln.split("," if "," in ln else ";")]
            else:
                parts = ln.split()
            try:
                nums = [float(x) if x else math.nan for x in parts]
                if all(math.isnan(v) for v in nums):
                    raise ValueError
            except ValueError:
                if data["frame"] or cols is not None:
                    raise ValueError(f"{path}:{no}: expected numbers but got {ln!r}")
                names = [x.lower() for x in parts]
                if "frame" not in names or "yaw" not in names:
                    raise ValueError(f"{path}:{no}: header needs 'frame' and 'yaw' columns")
                cols = {k: names.index(k) for k in data if k in names}
                continue
            idx = cols or {"frame": 0, "yaw": 1, "pitch": 2, "roll": 3}
            if len(nums) <= max(idx["frame"], idx["yaw"]):
                raise ValueError(f"{path}:{no}: expected at least frame and yaw, got {ln!r}")
            if not (math.isfinite(nums[idx["frame"]]) and math.isfinite(nums[idx["yaw"]])):
                continue                                   # a key without frame or yaw is no key
            for k, i in idx.items():
                data[k].append(nums[i] if i < len(nums) else math.nan)
        if not data["frame"]:
            raise ValueError(f"no 'frame, yaw' rows found in {path}")
        n = len(data["frame"])
        # optional columns: missing cells are NaN; a column with no values at all is absent
        full = lambda k: (data[k] if len(data[k]) == n and any(math.isfinite(v) for v in data[k])
                          else None)
        return cls(data["frame"], data["yaw"], fps, smooth, unwrap, full("pitch"), full("roll"),
                   full("scale"), full("cam_dist"))


# --------------------------------------------------------------------------- timing (keyframes, hits)
def _read_table(path):
    """CSV/TSV/space-separated table with a header row -> (names, rows). Empty cells are NaN,
    '#' starts a comment, a UTF-8 BOM is fine."""
    names, rows = None, []
    for no, ln in enumerate(Path(path).read_text(encoding="utf-8-sig").splitlines(), 1):
        ln = ln.split("#")[0].strip()
        if not ln:
            continue
        parts = ([x.strip() for x in ln.split("," if "," in ln else ";")]
                 if ("," in ln or ";" in ln) else ln.split())
        if names is None:
            names = [x.lower() for x in parts]
            continue
        try:
            rows.append([float(x) if x else math.nan for x in parts] + [math.nan] * (len(names) - len(parts)))
        except ValueError:
            raise ValueError(f"{path}:{no}: expected numbers, got {ln!r}") from None
    if names is None:
        raise ValueError(f"{path}: empty file")
    return names, rows


class Timing:
    """Directing the glitch over time.

    keys   {name: [(frame, value), ...]} keyframed multipliers, linear between keys, held before
           the first and after the last. 'intensity' scales -i; an effect name (e.g. 'jpeg')
           scales that effect's strength.
    hits   [(frame, strength, seconds)] forced bursts: activity jumps to full, intensity rises to
           at least `strength` and every effect gets up to twice as many events, then it all
           decays over `seconds`.
    trans  [(frame, 'in' | 'out', seconds)] materialise / dematerialise: the subject builds up
           out of glitch structure (or breaks apart and goes) starting at that frame. Before the
           first 'in' and after an 'out' it is invisible.
    Everything is a pure function of the frame number, so renders stay order-independent."""

    def __init__(self, keys=None, hits=None, fps: float = 25.0, trans=None):
        self.fps = float(fps)
        self.trans = sorted((float(f), str(k), float(sec)) for f, k, sec in (trans or []))
        if any(k not in ("in", "out") or sec <= 0 for _f, k, sec in self.trans):
            raise ValueError("transitions are (frame, 'in' | 'out', seconds > 0)")
        self.keys = {k: sorted((float(f), float(v)) for f, v in kv) for k, kv in (keys or {}).items() if kv}
        self.hits = sorted((float(f), float(st), float(sec)) for f, st, sec in (hits or []))
        bad = set(self.keys) - {"intensity", *EFFECT_ORDER}
        if bad:
            raise ValueError(f"unknown curve column(s): {sorted(bad)}; use intensity or "
                             f"{', '.join(EFFECT_ORDER)}")

    @property
    def empty(self) -> bool:
        return not self.keys and not self.hits and not self.trans

    def reveal(self, frame: float):
        """None (plainly visible), ('hidden',), or (kind, progress 0..1, index) mid-transition.
        The latest transition that has started wins."""
        if not self.trans:
            return None
        visible = self.trans[0][1] == "out"                  # the first key says what came before
        for i, (tf, kind, sec) in enumerate(self.trans):
            if tf > frame:
                break
            n = max(1.0, sec * self.fps)                     # frames; the last one lands at q = 1
            later = i + 1 < len(self.trans) and self.trans[i + 1][0] <= frame
            if frame - tf < n and not later:
                return (kind, float(min(1.0, (frame - tf + 1) / n)), i)
            visible = kind == "in"
        return None if visible else ("hidden",)

    def value(self, name: str, frame: float) -> float:
        kv = self.keys.get(name)
        if not kv:
            return 1.0
        f, v = zip(*kv)
        return float(np.interp(frame, f, v))

    def hit(self, frame: float) -> float:
        """Hit envelope at `frame`: quick rise (2 frames' worth), then a smooth fall over its length."""
        e = 0.0
        for hf, st, sec in self.hits:
            dt = (frame - hf) / self.fps
            if dt < -0.08 or dt > sec:
                continue
            env = _ss(dt, -0.08, 0.0) if dt < 0 else _ss(dt, max(sec, 1e-3), 0.0)
            e = max(e, env * st)
        return e

    def overrides(self, frame: float, intensity: float, gains: dict) -> dict:
        """Per-frame glitch() arguments for this frame (empty when there is nothing to change)."""
        if self.empty:
            return {}
        e = min(1.0, self.hit(frame))
        rv = self.reveal(frame)
        if rv is not None and rv[0] != "hidden":             # the glitch peaks mid-transition
            et = 0.9 * math.sin(math.pi * rv[1])
            e = max(e, et)
        inten = float(np.clip(intensity * self.value("intensity", frame), 0, 1))
        inten = max(inten, e)
        g = {fx: max(0.0, gains.get(fx, 1.0) * self.value(fx, frame) * (1.0 + e))
             for fx in EFFECT_ORDER}
        out = {"intensity": inten, "gains": g, "boost": e}
        if rv is not None:
            out["reveal"] = rv
        return out

    @staticmethod
    def parse_trans(text: str, kind: str, seconds: float = 1.0):
        """'24,200:1.5' -> [(24, kind, seconds), (200, kind, 1.5)]"""
        out = []
        flag = "--appear" if kind == "in" else "--vanish"
        for part in filter(None, (t.strip() for t in (text or "").split(","))):
            bits = part.split(":")
            try:
                fr = float(bits[0])
                sec = float(bits[1]) if len(bits) > 1 and bits[1] else seconds
            except ValueError:
                raise ValueError(f"{flag}: expected FRAME[:SECONDS], got {part!r}") from None
            if sec <= 0:
                raise ValueError(f"{flag} {part}: seconds must be > 0")
            out.append((fr, kind, sec))
        return out

    @staticmethod
    def parse_hits(text: str, seconds: float = 0.5):
        """'48,112:0.6,200:1:0.8' -> [(48, 1, sec), (112, 0.6, sec), (200, 1, 0.8)]"""
        out = []
        for part in filter(None, (t.strip() for t in (text or "").split(","))):
            bits = part.split(":")
            try:
                fr = float(bits[0])
                st = float(bits[1]) if len(bits) > 1 and bits[1] else 1.0
                sec = float(bits[2]) if len(bits) > 2 and bits[2] else seconds
            except ValueError:
                raise ValueError(f"--hits: expected FRAME[:STRENGTH[:SECONDS]], got {part!r}") from None
            if st < 0 or sec <= 0:
                raise ValueError(f"--hits {part}: strength must be >= 0 and seconds > 0")
            out.append((fr, st, sec))
        return out

    @classmethod
    def from_file(cls, path, fps: float = 25.0, hit_seconds: float = 0.5,
                  trans_seconds: float = 1.0) -> "Timing":
        """Header row required: 'frame' plus any of intensity, <effect names>, hit, hit_len,
        appear, vanish. Cells may be empty; each column interpolates over its own keys. A
        positive 'hit' value marks a hit of that strength (length from 'hit_len' seconds, else
        --hit-len). 'appear' / 'vanish' mark a transition; the value is its length in seconds
        (0 or 'x'-less blank = --trans-len)."""
        names, rows = _read_table(path)
        if "frame" not in names:
            raise ValueError(f"{path}: header needs a 'frame' column")
        fi = names.index("frame")
        keys, hits, trans = {}, [], []
        for r in rows:
            fr = r[fi]
            if not math.isfinite(fr):
                continue
            for col, kind in (("appear", "in"), ("vanish", "out")):
                if col in names:
                    v = r[names.index(col)]
                    if math.isfinite(v):
                        trans.append((fr, kind, v if v > 0 else trans_seconds))
            for i, n in enumerate(names):
                if n in ("frame", "hit", "hit_len", "appear", "vanish") or i >= len(r) or not math.isfinite(r[i]):
                    continue
                keys.setdefault(n, []).append((fr, r[i]))
            if "hit" in names:
                hv = r[names.index("hit")]
                if math.isfinite(hv) and hv > 0:
                    hl = r[names.index("hit_len")] if "hit_len" in names else math.nan
                    hits.append((fr, hv, hl if math.isfinite(hl) and hl > 0 else hit_seconds))
        return cls(keys, hits, fps, trans)

    def to_csv(self, path):
        """Write keys and hits in the format from_file reads."""
        cols = ["intensity"] + [k for k in EFFECT_ORDER if k in self.keys]
        rows = {}
        for c in cols:
            for f, v in self.keys.get(c, []):
                rows.setdefault(f, {})[c] = v
        for f, st, sec in self.hits:
            rows.setdefault(f, {}).update(hit=st, hit_len=sec)
        for f, kind, sec in self.trans:
            rows.setdefault(f, {})["appear" if kind == "in" else "vanish"] = sec
        head = ["frame"] + cols + ["hit", "hit_len"] + (["appear", "vanish"] if self.trans else [])
        fmt = lambda v: "" if v is None else f"{v:g}"
        lines = [",".join(head)] + [",".join([f"{f:g}"] + [fmt(rows[f].get(c)) for c in head[1:]])
                                    for f in sorted(rows)]
        Path(path).write_text("\n".join(lines) + "\n")


# --------------------------------------------------------------------------- materialise / vanish
_RV_D = 0.22                        # how long a pixel stays "in transit", in units of the whole ramp
_RV_U = (-0.25, 1.25)               # reveal map coverage, in layout-box widths / heights
_RV_V = (-0.15, 1.15)
_RV_GLOW = ((90, 255, 235), (255, 255, 255), (255, 80, 230))


@lru_cache(maxsize=16)
def _reveal_map(seed: int, idx: int):
    """Arrival time 0..1 per cell (uniformly spread, so progress q reveals a q share of the area),
    slice offsets per cell row (box widths) and a glow colour. Fixed for a transition, laid
    over the layout box, so the build-up is coherent frame to frame and follows the subject."""
    rng = np.random.default_rng([int(seed), 7919, int(idx)])
    FH, FW = 144, 96
    v = np.linspace(_RV_V[0], _RV_V[1], FH)[:, None] * np.ones((1, FW))
    u = np.ones((FH, 1)) * np.linspace(_RV_U[0], _RV_U[1], FW)[None, :]
    style = rng.choice(4, p=[0.35, 0.2, 0.3, 0.15])
    sweep = [v, 1 - v, np.hypot((u - 0.5) * 0.7, v - 0.45), u if rng.random() < 0.5 else 1 - u][style]

    def blocks(b):
        g = rng.random((-(-FH // b), -(-FW // b)))
        return np.repeat(np.repeat(g, b, 0), b, 1)[:FH, :FW]
    b = int(rng.choice([4, 6, 8, 12]))
    blk = 0.65 * blocks(b) + 0.35 * blocks(max(2, b // 2))
    rows = np.repeat(rng.random((FH + 1) // 2), 2)[:FH, None] * np.ones((1, FW))
    ws, wb = rng.uniform(0.35, 0.7), rng.uniform(0.25, 0.55)
    r = ws * (sweep - sweep.min()) / max(1e-6, np.ptp(sweep)) + wb * blk + 0.15 * rows \
        + 0.05 * rng.random((FH, FW))
    rank = np.empty(r.size, np.float32)
    rank[np.argsort(r, axis=None, kind="stable")] = np.arange(r.size, dtype=np.float32) / (r.size - 1)
    band = 3
    dx = rng.uniform(-1, 1, -(-FH // band)) * rng.uniform(0.08, 0.28) * (rng.random(-(-FH // band)) < 0.6)
    dx = np.repeat(dx, band)[:FH].astype(np.float32)
    glow = _RV_GLOW[int(rng.integers(len(_RV_GLOW)))]
    return rank.reshape(FH, FW), dx, glow


def _reveal_px(seed, idx, box, shape):
    """Per-pixel arrival time (1 outside the map), per-row slice offset in px, glow colour."""
    r, dxr, glow = _reveal_map(seed, idx)
    FH, FW = r.shape
    x0, y0, x1, y1 = box
    bw, bh = max(1, x1 - x0), max(1, y1 - y0)
    H, W = shape
    fy = ((np.arange(H) - y0) / bh - _RV_V[0]) / (_RV_V[1] - _RV_V[0]) * FH
    fx = ((np.arange(W) - x0) / bw - _RV_U[0]) / (_RV_U[1] - _RV_U[0]) * FW
    iy, ix = np.floor(fy).astype(np.int64), np.floor(fx).astype(np.int64)
    oky, okx = (iy >= 0) & (iy < FH), (ix >= 0) & (ix < FW)
    rp = r[np.clip(iy, 0, FH - 1)][:, np.clip(ix, 0, FW - 1)]
    rp = np.where(oky[:, None] & okx[None, :], rp, 1.0).astype(np.float32)
    dx = np.where(oky, dxr[np.clip(iy, 0, FH - 1)], 0.0) * bw
    return rp, dx, glow


def _reveal_start(a, kind, q, rp, dx, glow, box):
    """Show the subject partly built (or partly gone): pixels in transit are shoved sideways in
    slices that land as they settle, shown as flat mosaic blocks and lit up; the rest is either
    there or not."""
    s = (q * (1 + _RV_D) - rp) / _RV_D                      # <0 not yet, 0..1 in transit, >1 done
    if kind == "in":
        amul = (s >= 0).astype(np.float32)
        k = np.where(s >= 0, np.clip(1 - s, 0, 1), 0.0)     # 1 = just arrived
    else:
        amul = np.where(s >= 0, np.clip(1 - s, 0, 1), 1.0)
        k = np.where((s >= 0) & (amul > 0), np.clip(s, 0, 1), 0.0)   # 1 = about to go
    H, W = a.shape[:2]
    touched = k > 0
    out = a.copy()
    if touched.any():
        bw = max(1, box[2] - box[0])
        Y, X = np.nonzero(touched)
        kk = k[Y, X]
        sx = np.clip(X - np.round(dx[Y] * kk * kk).astype(np.int64), 0, W - 1)
        src = a[Y, sx].astype(np.float32)
        m = max(4, int(bw / 45))                            # mosaic block: premultiplied mean
        hh, ww = -(-H // m) * m, -(-W // m) * m
        pm = np.zeros((hh, ww, 4), np.float32)
        al = a[..., 3:4].astype(np.float32) / 255
        pm[:H, :W, :3] = a[..., :3] * al
        pm[:H, :W, 3] = a[..., 3]
        bm = pm.reshape(hh // m, m, ww // m, m, 4).mean(axis=(1, 3))
        mb = bm[Y // m, sx // m]
        mrgb = np.where(mb[:, 3:4] > 1e-3, mb[:, :3] / np.maximum(mb[:, 3:4] / 255, 1e-6), 0)
        w = np.clip(kk * 1.4 - 0.2, 0, 1)[:, None]
        rgb = src[:, :3] * (1 - w) + mrgb * w
        alpha = src[:, 3] * (1 - w[:, 0]) + mb[:, 3] * w[:, 0]
        g = (kk ** 3 * 0.65)[:, None]
        rgb = rgb * (1 - g) + np.asarray(glow, np.float32) * g
        if a.dtype == np.uint8:
            rgb, alpha = np.clip(rgb + 0.5, 0, 255), np.clip(alpha + 0.5, 0, 255)
        out[Y, X, :3] = rgb.astype(a.dtype)
        out[Y, X, 3] = alpha.astype(a.dtype)
    return _reveal_alpha(out, amul)


def _reveal_alpha(a, mul):
    """alpha *= mul (exact where mul is 1), colour cleared where nothing is left."""
    part = mul < 1
    if not part.any():
        return a
    out = a.copy()
    al = out[..., 3].astype(np.float32) * mul
    out[..., 3] = (np.clip(al + 0.5, 0, 255) if a.dtype == np.uint8 else al).astype(a.dtype)
    out[out[..., 3] == 0] = 0
    return out


def _reveal_end_mul(kind, q, rp):
    s = (q * (1 + _RV_D) - rp) / _RV_D
    return (s >= -0.35).astype(np.float32) if kind == "in" else np.clip(1.35 - s, 0, 1)


def _reveal_end(a, kind, q, rp):
    """After the effects: glitch fragments only where the subject exists (a little ahead of the
    front, so the build-up sparkles)."""
    return _reveal_alpha(a, _reveal_end_mul(kind, q, rp))


# --------------------------------------------------------------------------- background plate
def _over(top, bottom):
    """Straight-alpha 'over' for working arrays of the same dtype (uint8 or float 0..255).
    Opaque top pixels are the top, empty ones over an opaque bottom are the bottom (exactly);
    only the rest - edges, soft alpha - is blended."""
    out = np.where(top[..., 3:4] >= 255, top, bottom).astype(top.dtype)
    part = (top[..., 3] < 255) & ~((top[..., 3] <= 0) & (bottom[..., 3] >= 255))
    if not part.any():
        return out
    t, b = top[part].astype(np.float32), bottom[part].astype(np.float32)
    ta, ba = t[:, 3:4] / 255, b[:, 3:4] / 255
    oa = ta + ba * (1 - ta)
    rgb = (t[:, :3] * ta + b[:, :3] * ba * (1 - ta)) / np.maximum(oa, 1e-6)
    v = np.concatenate([rgb, oa * 255], -1)
    out[part] = np.clip(v + 0.5, 0, 255).astype(np.uint8) if top.dtype == np.uint8 else v
    return out


def _spill_mask(alpha, spill: float, box):
    """Where the glitch may change the picture once the subject sits on a plate: the subject
    itself, plus a soft falloff around it that reaches ~spill x box size (0 = subject only)."""
    al = np.clip(alpha.astype(np.float32) / 255, 0, 1)
    if spill <= 0:
        return al
    H, W = al.shape
    q = 8
    h, w = -(-H // q), -(-W // q)
    small = np.zeros((h * q, w * q), np.float32)
    small[:H, :W] = al
    small = small.reshape(h, q, w, q).max(axis=(1, 3))
    R = max(1, int(round(spill * 0.5 * max(box[2] - box[0], box[3] - box[1]) / q)))
    # distance from the subject in cells, by growing its outline one ring per step (8-connected)
    cur = small > 0.5
    dist = np.where(cur, 0.0, np.inf).astype(np.float32)
    for step in range(1, R + 1):
        grown = cur.copy()
        grown[1:] |= cur[:-1]
        grown[:-1] |= cur[1:]
        grown[:, 1:] |= cur[:, :-1]
        grown[:, :-1] |= cur[:, 1:]
        grown[1:, 1:] |= cur[:-1, :-1]
        grown[1:, :-1] |= cur[:-1, 1:]
        grown[:-1, 1:] |= cur[1:, :-1]
        grown[:-1, :-1] |= cur[1:, 1:]
        dist[grown & ~cur] = step
        cur = grown
    b = np.clip(1 - dist / R, 0, 1)
    b = (b * b * (3 - 2 * b)).astype(np.float32)          # full next to the body, fades to 0 at R
    big = np.asarray(Image.fromarray(b, "F").resize((w * q, h * q), Image.BILINEAR), np.float32)[:H, :W]
    return np.maximum(al, big)


def load_plate(path, io, like, size):
    """A background plate as a working image of the same kind as `like` (PIL / float array),
    resized to `size` (W, H) when it differs."""
    img, _ = load_frame(path, io)
    if isinstance(like, np.ndarray) and isinstance(img, Image.Image):
        img = np.asarray(img.convert("RGBA"), np.float32)
    elif isinstance(like, Image.Image) and isinstance(img, np.ndarray):
        img = to_display8(img)
    cur = (img.shape[1], img.shape[0]) if isinstance(img, np.ndarray) else img.size
    return resize_work(img, size) if tuple(cur) != tuple(size) else img


def composite_on(cut, plate):
    """Subject over plate, as a working image of the cutout's kind (opaque if the plate is)."""
    is_float = isinstance(cut, np.ndarray)
    c = _to_work(cut, is_float)
    pl = _to_work(plate, is_float)
    out = _over(c, pl)
    return out if is_float else Image.fromarray(out, "RGBA")


# --------------------------------------------------------------------------- body-part targeting
PARTS = ("face", "head", "torso", "arms", "hands", "legs", "feet")
PART_GROUPS = {"upper": ("head", "torso", "arms", "hands"), "lower": ("legs", "feet"),
               "body": PARTS}


def parse_parts(text) -> tuple:
    """'head,hands' / ['upper'] -> ('head', 'hands') ...; groups expand; validated."""
    items = text.split(",") if isinstance(text, str) else list(text or ())
    out = []
    for it in (x.strip().lower() for x in items):
        if not it or it == "all":
            continue
        grp = PART_GROUPS.get(it, (it,))
        for g in grp:
            if g not in PARTS:
                raise ValueError(f"unknown body part {it!r}; use {', '.join(PARTS)} or "
                                 f"{', '.join(PART_GROUPS)}")
            if g not in out:
                out.append(g)
    return tuple(out)


class Landmarks:
    """Per-frame MediaPipe body landmarks from estimate_pose.py (<pose>_landmarks.npz): x, y in
    source pixels + visibility. Frames without a person are filled from their neighbours, and the
    track is lightly smoothed (MediaPipe jitters a few pixels frame to frame)."""

    def __init__(self, path):
        if str(path).lower().endswith(".csv"):
            fr, lm = self._read_csv(path)
        else:
            with np.load(path, allow_pickle=False) as z:
                fr = np.asarray(z["frames"], np.float64)
                lm = np.asarray(z["lm"], np.float64)
        if lm.ndim != 3 or lm.shape[1] < 33 or len(fr) != len(lm):
            raise ValueError(f"{path}: not a landmarks file from estimate_pose.py")
        ok = np.isfinite(lm[:, 0, 0])
        if not ok.any():
            raise ValueError(f"{path}: no person in any frame")
        fr, lm = fr[ok], lm[ok]
        if len(lm) >= 3:                                   # [1 2 1] smoothing of positions
            sm = lm.copy()
            sm[1:-1, :, :2] = (lm[:-2, :, :2] + 2 * lm[1:-1, :, :2] + lm[2:, :, :2]) / 4
            lm = sm
        self.frames, self.lm = fr, lm

    @staticmethod
    def _read_csv(path):
        """estimate_pose.py's _landmarks.csv (the same data as the .npz, as text)."""
        fr, lm = [], []
        for ln in Path(path).read_text(encoding="utf-8-sig").splitlines():
            ln = ln.split("#")[0].strip()
            if not ln or ln.lower().startswith(("size", "frame")):
                continue
            v = [float(x) if x.strip() else math.nan for x in ln.split(",")]
            row = (v[1:] + [math.nan] * 99)[:99]
            fr.append(v[0])
            lm.append(np.asarray(row, np.float64).reshape(33, 3))
        if not fr:
            raise ValueError(f"{path}: no landmark rows")
        return np.asarray(fr, np.float64), np.stack(lm)

    def at(self, frame: float) -> np.ndarray:
        """(33, 3) at this frame number, interpolated between known frames, held beyond them."""
        f = self.frames
        if frame <= f[0]:
            return self.lm[0].astype(np.float32)
        if frame >= f[-1]:
            return self.lm[-1].astype(np.float32)
        j = int(np.searchsorted(f, frame))
        w = (frame - f[j - 1]) / max(1e-9, f[j] - f[j - 1])
        return ((1 - w) * self.lm[j - 1] + w * self.lm[j]).astype(np.float32)


def _part_shapes(lm, part):
    """Capsules (p, q, radius) for a body part; a circle is a capsule with p == q."""
    P = lambda *ix: np.mean([lm[i, :2] for i in ix], axis=0)   # noqa: E731
    ls, rs, lh, rh = P(11), P(12), P(23), P(24)
    ms, mh = (ls + rs) / 2, (lh + rh) / 2
    S = max(float(np.linalg.norm(ls - rs)), 0.55 * float(np.linalg.norm(ms - mh)), 4.0)
    rh_ = max(0.62 * float(np.linalg.norm(P(7) - P(8))), 0.3 * S)       # head radius
    hc = P(0, 7, 8) - np.array([0.0, 0.25 * rh_])                      # a little up: hair, crown
    if part == "head":
        return [(hc, hc, 1.4 * rh_), (hc, ms, 0.18 * S)]
    if part == "face":
        fc = P(0, 2, 5, 9, 10)
        return [(fc, fc, 0.72 * rh_)]
    if part == "torso":
        return [(ms, mh, 0.42 * S), (ls, rs, 0.14 * S), (lh, rh, 0.16 * S), (ls, lh, 0.15 * S),
                (rs, rh, 0.15 * S)]
    if part == "arms":
        r = 0.14 * S
        return [(P(11), P(13), r), (P(13), P(15), r), (P(12), P(14), r), (P(14), P(16), r)]
    if part == "hands":
        r = 0.11 * S
        return [(P(15), P(17, 19), r), (P(15, 17, 19, 21), P(15, 17, 19, 21), 0.13 * S),
                (P(16), P(18, 20), r), (P(16, 18, 20, 22), P(16, 18, 20, 22), 0.13 * S)]
    if part == "legs":
        r = 0.21 * S
        return [(P(23), P(25), r), (P(25), P(27), r), (P(24), P(26), r), (P(26), P(28), r)]
    if part == "feet":
        r = 0.08 * S
        return [(P(27), P(29), r), (P(29), P(31), r), (P(27), P(31), r),
                (P(28), P(30), r), (P(30), P(32), r), (P(28), P(32), r)]
    raise ValueError(part)


def part_mask(lm, parts, shape, offset=(0, 0), feather: float = 1.0) -> np.ndarray:
    """Soft 0..1 mask (float32, `shape`) covering the parts; lm in source px, offset = padding.
    Drawn at quarter resolution and scaled up (soft anyway, and 16x cheaper)."""
    H, W = shape
    lm = np.asarray(lm, np.float64).copy()
    if lm.shape[0] < 33 or not np.isfinite(lm[:33, :2]).all():
        return np.zeros((H, W), np.float32)                # no usable pose for this frame
    lm[:, 0] += offset[0]
    lm[:, 1] += offset[1]
    ls, rs = lm[11, :2], lm[12, :2]
    ms, mh = (lm[11, :2] + lm[12, :2]) / 2, (lm[23, :2] + lm[24, :2]) / 2
    S = max(float(np.linalg.norm(ls - rs)), 0.55 * float(np.linalg.norm(ms - mh)), 4.0)
    fe = max(2.0, 0.08 * S * max(0.0, float(feather)))
    q = 4
    h, w = -(-H // q), -(-W // q)
    gy = (np.arange(h, dtype=np.float32) * q + q / 2)[:, None]
    gx = (np.arange(w, dtype=np.float32) * q + q / 2)[None, :]
    m = np.zeros((h, w), np.float32)
    for part in parts:
        for p0, p1, r in _part_shapes(lm, part):
            d = p1 - p0
            L2 = float(d @ d)
            x0, x1 = min(p0[0], p1[0]) - r - fe, max(p0[0], p1[0]) + r + fe   # only the bounding box
            y0, y1 = min(p0[1], p1[1]) - r - fe, max(p0[1], p1[1]) + r + fe
            i0, i1 = max(0, int(y0 // q)), min(h, int(y1 // q) + 2)
            j0, j1 = max(0, int(x0 // q)), min(w, int(x1 // q) + 2)
            if i0 >= i1 or j0 >= j1:
                continue
            yy, xx = gy[i0:i1], gx[:, j0:j1]
            if L2 < 1e-6:
                dist = np.hypot(xx - p0[0], yy - p0[1])
            else:
                t = np.clip(((xx - p0[0]) * d[0] + (yy - p0[1]) * d[1]) / L2, 0, 1)
                dist = np.hypot(xx - (p0[0] + t * d[0]), yy - (p0[1] + t * d[1]))
            v = np.clip((r + fe / 2 - dist) / fe, 0, 1)
            np.maximum(m[i0:i1, j0:j1], v, out=m[i0:i1, j0:j1])
    if not m.any():
        return np.zeros((H, W), np.float32)
    big = np.asarray(Image.fromarray(m, "F").resize((w * q, h * q), Image.BILINEAR), np.float32)
    return np.clip(big[:H, :W], 0, 1)


def _blend(before, after, m):
    """before where m = 0, after where m = 1 (both exact), mixed in between."""
    if m is None:
        return after
    if after is before:
        return after
    full, none = m >= 0.999, m <= 0.001
    if full.all():
        return after
    out = np.where(full[..., None], after, before)
    mid = ~(full | none)
    if mid.any():
        w = m[mid][:, None]
        v = before[mid].astype(np.float32) * (1 - w) + after[mid].astype(np.float32) * w
        out[mid] = np.clip(v + 0.5, 0, 255).astype(np.uint8) if out.dtype == np.uint8 else v
    return out


# --------------------------------------------------------------------------- motion-reactive
_MOTION_FEAT: dict = {}            # (path, size, mtime, io) -> low-res image, per process


def _motion_feat(path, io):
    st = Path(path).stat()
    key = (str(path), st.st_size, st.st_mtime_ns, json.dumps(io or {}, sort_keys=True))
    f = _MOTION_FEAT.get(key)
    if f is None:
        img, _ = load_frame(path, io)
        if isinstance(img, Image.Image):                  # Pillow's box reduce: fast, in C
            im = img.convert("RGBA")
            sm = np.asarray(im.reduce(max(1, int(round(max(im.size) / 160)))), np.float32)
        else:
            a = np.asarray(img, np.float32)
            s = max(1, int(round(max(a.shape[:2]) / 160)))
            h, w = (a.shape[0] // s) * s, (a.shape[1] // s) * s
            sm = a[:h, :w].reshape(h // s, s, w // s, s, 4).mean(axis=(1, 3))
        al = sm[..., 3] / 255
        f = np.stack([np.minimum(_lum(sm[..., :3]), 255) * al, sm[..., 3]], -1).astype(np.float32)
        if len(_MOTION_FEAT) > 4000:
            _MOTION_FEAT.clear()
        _MOTION_FEAT[key] = f
    return f


def motion_curve(paths, io=None) -> np.ndarray:
    """How fast the subject moves at each frame, 0..~1.5, from the frame-to-frame change of a
    small brightness+alpha image inside the subject (body, hair, clothes and travel all count;
    no pose needed). Scaled to the shot: 1 = its fast moments (95th percentile)."""
    n = len(paths)
    if n < 2:
        return np.zeros(n)
    feats = [_motion_feat(p, io) for p in paths]
    m = np.zeros(n)
    for i in range(1, n):
        a, b = feats[i - 1], feats[i]
        if a.shape != b.shape:
            continue
        area = np.maximum(a[..., 1], b[..., 1]) > 8
        if area.any():
            m[i] = float(np.abs(a - b)[area].mean())
    m[0] = m[1]
    if n >= 3:                                           # [1 2 1]: frame-difference jitter
        m = np.convolve(np.pad(m, 1, mode="edge"), [0.25, 0.5, 0.25], mode="valid")
    ref = float(np.percentile(m, 95))
    return np.clip(m / ref, 0, 1.5) if ref > 1e-6 else np.zeros(n)


def apply_motion(fo: dict, intensity: float, v: float, amount: float) -> dict:
    """Fold motion v (motion_curve) into a frame's overrides: with amount 1 the glitch drops to
    a sixth when the subject is still and goes past full on fast moves (and forces activity)."""
    if amount <= 0:
        return fo
    k = 1 - amount + amount * (0.15 + 1.6 * min(v, 1.2))
    out = dict(fo)
    out["intensity"] = float(np.clip(fo.get("intensity", intensity) * k, 0, 1))
    out["boost"] = max(fo.get("boost", 0.0), amount * float(np.clip((v - 0.55) * 2.2, 0, 1)))
    return out


# --------------------------------------------------------------------------- audio-reactive
AUDIO_SR = 22050                     # compressed formats are decoded at this rate (plenty for analysis)
_AUDIO_CACHE: dict = {}


def load_audio(path):
    """-> (mono float32 samples in -1..1, sample rate). WAV through the standard library; MP3,
    FLAC, OGG (and WAVs it can't read, e.g. 32-bit float) through the miniaudio package, or
    ffmpeg if that is installed. Cached per file while it is unchanged."""
    p = Path(path)
    st = p.stat()
    key = (str(p.resolve()), st.st_size, st.st_mtime_ns)
    if key in _AUDIO_CACHE:
        return _AUDIO_CACHE[key]
    x, sr = None, None
    if p.suffix.lower() == ".wav":
        import wave
        try:
            with wave.open(str(p), "rb") as w:
                n, ch, sw, sr = w.getnframes(), w.getnchannels(), w.getsampwidth(), w.getframerate()
                raw = w.readframes(n)
            if sw == 1:
                x = (np.frombuffer(raw, np.uint8).astype(np.float32) - 128) / 128
            elif sw == 2:
                x = np.frombuffer(raw, "<i2").astype(np.float32) / 32768
            elif sw == 3:
                b = np.frombuffer(raw, np.uint8).reshape(-1, 3).astype(np.int32)
                v = b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16)
                x = np.where(v >= 1 << 23, v - (1 << 24), v).astype(np.float32) / (1 << 23)
            elif sw == 4:
                x = np.frombuffer(raw, "<i4").astype(np.float32) / 2 ** 31
            if x is not None:
                x = x[: len(x) // ch * ch].reshape(-1, ch).mean(axis=1)
        except (wave.Error, EOFError, ValueError):
            x = None
    if x is None:
        try:
            import miniaudio
        except ImportError:
            miniaudio = None
        if miniaudio is not None:
            try:
                d = miniaudio.decode_file(str(p), output_format=miniaudio.SampleFormat.SIGNED16,
                                          nchannels=1, sample_rate=AUDIO_SR)
            except miniaudio.DecodeError as ex:
                raise ValueError(f"{p.name}: cannot decode audio ({ex})") from None
            x, sr = np.frombuffer(d.samples, np.int16).astype(np.float32) / 32768, d.sample_rate
        else:
            ff = shutil.which("ffmpeg")
            if not ff:
                raise ValueError(f"{p.name}: reading {p.suffix or 'this'} audio needs the miniaudio "
                                 f"package (pip install miniaudio) or ffmpeg; WAV works without either")
            r = subprocess.run([ff, "-v", "error", "-i", str(p), "-f", "s16le", "-ac", "1",
                                "-ar", str(AUDIO_SR), "-"], capture_output=True, stdin=subprocess.DEVNULL)
            if r.returncode != 0:
                raise ValueError(f"{p.name}: ffmpeg could not decode it: {r.stderr.decode(errors='replace')[-200:]}")
            x, sr = np.frombuffer(r.stdout, np.int16).astype(np.float32) / 32768, AUDIO_SR
    if x is None or x.size == 0 or not sr:
        raise ValueError(f"{p.name}: no audio samples")
    x = np.ascontiguousarray(x, np.float32)
    if len(_AUDIO_CACHE) >= 4:
        _AUDIO_CACHE.clear()
    _AUDIO_CACHE[key] = (x, int(sr))
    return x, int(sr)


def _onset_flux(x, sr):
    """Spectral flux (sum of rises in log-magnitude across frequency bins) on a 100 Hz grid ->
    (times s, flux). Rises in energy across many bands = drum hits, plucks, attacks."""
    win = 1 << int(round(math.log2(max(256, sr * 0.046))))
    hop = max(1, int(sr / 100))
    if len(x) < win + hop:
        return np.zeros(0), np.zeros(0)
    m = 1 + (len(x) - win) // hop
    w = np.hanning(win).astype(np.float32)
    flux = np.zeros(m, np.float32)
    prev = None
    for c0 in range(0, m, 2048):                         # chunks: long songs stay small in memory
        c1 = min(m, c0 + 2048)
        idx = (np.arange(c0, c1) * hop)[:, None] + np.arange(win)[None, :]
        mag = np.log1p(100 * np.abs(np.fft.rfft(x[idx] * w, axis=1))).astype(np.float32)
        d = np.diff(np.vstack([mag[:1] if prev is None else prev, mag]), axis=0)
        flux[c0:c1] = np.maximum(d, 0).sum(axis=1)
        prev = mag[-1:]
    flux[0] = 0
    return (np.arange(m) * hop + win / 2) / sr, flux


def audio_curves(path, frame_times, start: float = 0.0, end: float | None = None,
                 sens: float = 0.5, fps: float = 25.0):
    """Per-frame audio drive for frames at the given audio times (s):
    loud  loudness, 0..1.5 (1 = the passage's loud parts; more sensitivity = quieter counts)
    hit   beat envelope 0..1: jumps at each detected onset, falls off over ~0.35 s
    beats onset times (s) found inside the passage. Frames outside [start, end] get 0."""
    x, sr = load_audio(path)
    dur = len(x) / sr
    end = dur if end is None or end <= 0 else min(float(end), dur)
    start = float(np.clip(start, 0, max(0.0, end - 1e-3)))
    sens = float(np.clip(sens, 0, 1))
    ft = np.asarray(frame_times, np.float64)
    # loudness: RMS on a 200 Hz grid, averaged over each frame's 1/fps slice
    h = max(1, int(sr / 200))
    nb = len(x) // h
    rms = np.sqrt((x[:nb * h].reshape(nb, h).astype(np.float64) ** 2).mean(axis=1) + 1e-12)
    cs = np.concatenate([[0.0], np.cumsum(rms)])
    i0 = np.clip((ft * sr / h).astype(np.int64), 0, nb)
    i1 = np.clip(((ft + 1.0 / fps) * sr / h).astype(np.int64), 0, nb)
    i1 = np.maximum(i1, np.minimum(i0 + 1, nb))
    fr_rms = (cs[i1] - cs[i0]) / np.maximum(i1 - i0, 1)
    a, b = int(start * sr / h), max(int(start * sr / h) + 1, int(end * sr / h))
    ref = float(np.percentile(rms[a:b], 97 - 40 * sens)) if b > a else 0.0
    loud = np.clip(fr_rms / ref, 0, 1.5) if ref > 1e-6 else np.zeros(len(ft))
    # beats: peaks of spectral flux standing out from the passage's typical level
    tt, flux = _onset_flux(x, sr)
    inp = (tt >= start) & (tt <= end)
    beats = np.zeros(0)
    hit = np.zeros(len(ft))
    if inp.sum() > 8:
        f = flux[inp]
        # adaptive threshold: the local average (+-0.3 s, so busy sections need more to count)
        # plus a margin scaled to the passage's strong onsets; sensitivity shrinks the margin
        scale = float(np.percentile(f, 98) - np.median(f)) + 1e-6
        k = 30                                           # 0.3 s on the 100 Hz grid
        cs_f = np.concatenate([[0.0], np.cumsum(flux, dtype=np.float64)])
        idx = np.arange(len(flux))
        lo_i, hi_i = np.clip(idx - k, 0, len(flux)), np.clip(idx + k + 1, 0, len(flux))
        local = (cs_f[hi_i] - cs_f[lo_i]) / (hi_i - lo_i)
        lam = 0.75 - 0.65 * sens                         # sens 0 -> only the biggest hits, 1 -> most
        z = (flux - local) / scale
        thr = lam
        pk = np.flatnonzero(inp[1:-1] & (z[1:-1] > thr) & (z[1:-1] >= z[:-2]) & (z[1:-1] > z[2:])) + 1
        keep, last = [], -1e9
        for k in pk:                                     # at most one beat per 0.1 s (the strongest)
            if tt[k] - last >= 0.1:
                keep.append(k)
                last = tt[k]
            elif z[k] > z[keep[-1]]:
                keep[-1] = k
                last = tt[k]
        beats = tt[keep]
        strength = np.clip(0.5 + (z[keep] - thr) / (2 * max(thr, 0.1)), 0.5, 1.0)
        for bt, stg in zip(beats, strength):            # a beat lands on the frame that contains it
            dt = ft - bt
            on = (dt > -1.0 / fps) & (dt < 0.35)
            hit[on] = np.maximum(hit[on], stg * (1 - np.clip(dt[on], 0, None) / 0.35) ** 2)
    out = (ft < start) | (ft >= end)
    loud[out] = 0
    hit[out] = 0
    return loud, hit, beats


def apply_audio(fo: dict, intensity: float, gains: dict, loud: float, hit: float, amount: float) -> dict:
    """Fold audio into a frame's overrides: loudness scales intensity like the motion slider
    (amount 1: quiet -> a sixth, loud -> past full); beats force activity and up to twice the
    events for a moment."""
    if amount <= 0:
        return fo
    k = 1 - amount + amount * (0.15 + 1.6 * min(loud, 1.2))
    e = amount * float(hit)
    out = dict(fo)
    out["intensity"] = max(float(np.clip(fo.get("intensity", intensity) * k, 0, 1)), e)
    out["boost"] = max(fo.get("boost", 0.0), e)
    if e > 0:
        base = fo.get("gains") or {fx: gains.get(fx, 1.0) for fx in EFFECT_ORDER}
        out["gains"] = {fx: g * (1 + e) for fx, g in base.items()}
    return out


def activity_level(t: float, *, seed: int = 0, intensity: float = 0.5, bursts: float = 0.5,
                   boost: float = 0.0, **_ignored) -> float:
    """How busy the glitch is at time t, 0..1 (the calm/burst curve that gates events, times the
    intensity fade-in). Takes a render job's kw (+ its timing overrides); used by the GUI graph."""
    p = Params(seed=int(seed) & 0x7FFFFFFF, intensity=float(np.clip(intensity, 0, 1)),
               bursts=float(np.clip(bursts, 0, 1)), boost=float(np.clip(boost, 0, 1)))
    return _activity(p, t) * min(1.0, p.intensity * 5)


# --------------------------------------------------------------------------- public API
def _to_work(img, is_float: bool):
    """PIL image or float array -> working RGBA array (uint8, or float32 0..255 scale), invisible
    pixels zeroed."""
    if is_float:
        a = np.array(img, dtype=np.float32)                 # copy; float pipeline, no 8-bit steps
        if a.ndim != 3 or a.shape[2] != 4:
            raise ValueError("float input must be an (h, w, 4) RGBA array")
        a[..., 3] = np.clip(a[..., 3], 0, 255)
    else:
        if img.mode in ("I;16", "I;16L", "I;16B", "I;16N", "I"):     # 16/32-bit gray: Pillow would clip
            g16 = np.asarray(img, dtype=np.int64)
            img = Image.fromarray(np.clip(g16 >> 8 if g16.max() > 255 else g16, 0, 255).astype(np.uint8), "L")
        a = np.array(img.convert("RGBA"))
    np.copyto(a, 0, where=(a[..., 3:4] == 0))              # drop junk RGB hiding under alpha 0
    return a


def glitch(img, t: float = 0.0, *, seed: int | None = None, intensity: float = 0.5,
           fps: float = 25.0, tempo: float = 1.0, density: float = 1.0, bursts: float = 0.5,
           pad: float = 0.0, tint: float = 0.0, skip=(), bleed: int | None = None,
           quality: int | None = None, bbox=None, shift=(0.0, 0.0), frame: int | None = None,
           trim: bool = False, yaw: float | None = None, lock: float = 1.0,
           cam_dist: float = 3.0, depth: float = 0.35, pitch: float | None = None,
           roll: float | None = None, tilt: float = 0.25, gains: dict | None = None,
           boost: float = 0.0, history=None, history_key=None, reveal=None, landmarks=None,
           only=(), protect=(), targets=None, feather: float = 1.0, plate=None,
           spill: float = 0.05):
    """Return a glitched RGBA copy of `img` as it looks at time `t` (seconds).

    `img` is a PIL image (processed in 8 bit, returns a PIL image) or a float32 numpy array of
    shape (h, w, 4) with straight alpha and values on a 0..255 scale - values above 255 are HDR
    and are kept (returns a float32 array). load_frame() gives you either from a file.

    For a sequence, call once per frame with t = frame/fps, the same seed, and the same `bbox`
    (x0, y0, x1, y1 in the *input's* pixel coordinates: the area the glitch is laid out in; omit
    to use this image's own subject box). Output is padded by `pad` per side so spills fit.
    intensity 0..1 | tempo >1 faster events | density event-count multiplier | bursts 0..1
    calm/burst contrast | bleed px that JPEG junk may spill past the silhouette |
    gains = {effect: strength 0..3} (1 default, 0 off; event effects get more and bigger events,
    split/tint/scanlines/noise scale their strength).
    3D mode (on when yaw, pitch or roll is given): yaw/pitch/roll in degrees, see Space3D |
    lock = share of the subject's turn (yaw) the glitch volume follows (1 glued, 0 ignores) |
    cam_dist = camera distance in box heights (perspective strength) | depth = half-thickness
    of the volume in box widths | tilt = share of planes tilted toward horizontal (seen from
    above/below only).
    Time effects (echo, stutter, mosh) need `history`: a callable k -> the *source* image k frames
    earlier (same type and size as img, None before the start), or a list [1 back, 2 back, ...];
    up to MAX_BACK frames are asked for. A `key(k)` method on it (and `history_key` for this
    frame) lets motion between frames be cached. Without history these effects do nothing.
    reveal = ('in' | 'out', progress 0..1, index) materialises / dematerialises the subject (the
    index picks the build-up pattern), ('hidden',) returns an empty canvas; Timing.reveal()
    gives it per frame.
    Body parts (need `landmarks`: this frame's (33, 3) MediaPipe landmarks in img pixels, see
    Landmarks): only = parts the glitch may touch, protect = parts kept clean, targets =
    {effect: parts} for single effects. Parts: face head torso arms hands legs feet (+ groups
    upper, lower). feather scales the soft edge.
    plate = background (same kind and size as img): the subject is composited onto it and the
    glitch may damage the plate around the subject, out to ~spill x subject size (0 = only the
    subject). The result is the opaque composite, unpadded; history frames should then be
    composites too (SeqHistory(plates=...) does that).
    """
    if seed is None:
        seed = int(np.random.SeedSequence().entropy % 2**31)
    skip = tuple(skip)
    unknown = set(skip) - set(EFFECT_ORDER)
    if unknown:
        raise ValueError(f"unknown effect(s): {sorted(unknown)}; valid: {EFFECT_ORDER}")
    p = Params(seed=int(seed) & 0x7FFFFFFF, intensity=float(np.clip(intensity, 0, 1)),
               fps=float(fps), tempo=max(0.1, float(tempo)), density=max(0.0, float(density)),
               bursts=float(np.clip(bursts, 0, 1)), pad=max(0.0, float(pad)),
               tint=float(np.clip(tint, 0, 1)), skip=skip, bleed=bleed, quality=quality,
               gains=tuple(sorted((k, max(0.0, float(v))) for k, v in (gains or {}).items()
                                  if float(v) != 1.0)),
               boost=float(np.clip(boost, 0, 1)))
    bad = {k for k, _ in p.gains} - set(EFFECT_ORDER)
    if bad:
        raise ValueError(f"unknown effect(s) in gains: {sorted(bad)}; valid: {EFFECT_ORDER}")
    t = max(0.0, float(t))
    fi = int(frame) if frame is not None else int(math.floor(t * p.fps + 1e-6))
    if fi < 0:
        raise ValueError("frame index must be >= 0")

    is_float = isinstance(img, np.ndarray)
    a = _to_work(img, is_float)
    H, W = a.shape[:2]
    pl = None
    if plate is not None:                                   # comp mode: frame = plate, no padding
        pl = _to_work(plate, is_float)
        if pl.shape != a.shape:
            raise ValueError(f"plate is {pl.shape[1]}x{pl.shape[0]}, frame is {W}x{H}: sizes must match")
        p = _dc_replace(p, pad=0.0)
    py, px = int(H * p.pad), int(W * p.pad)
    a = np.pad(a, ((py, py), (px, px), (0, 0)))
    Hp, Wp = a.shape[:2]
    if reveal is not None and reveal[0] == "hidden":
        a[:] = 0
    if not (a[..., 3] > 0).any():                           # empty frame (subject out of shot)
        if pl is not None:
            return pl if is_float else Image.fromarray(pl, "RGBA")
        return a if is_float else Image.fromarray(a, "RGBA")

    if bbox is None:
        ys, xs = np.nonzero(a[..., 3] > 8)
        if ys.size == 0:                                    # only faint alpha (e.g. motion blur)
            ys, xs = np.nonzero(a[..., 3] > 0)
        bb = (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)
    else:
        bb = (bbox[0] + px + shift[0], bbox[1] + py + shift[1],
              bbox[2] + px + shift[0], bbox[3] + py + shift[1])
    # The layout box may extend past the canvas (a follow box when the subject leaves frame);
    # clipping it would shrink and shift the layout, making glitches jump. Effects clip their own
    # pixel coordinates.
    x0, y0 = int(round(bb[0])), int(round(bb[1]))
    x1, y1 = max(x0 + 2, int(round(bb[2]))), max(y0 + 2, int(round(bb[3])))
    rv = None
    if reveal is not None:
        kind, q = str(reveal[0]), float(np.clip(reveal[1], 0, 1))
        if kind not in ("in", "out"):
            raise ValueError(f"reveal: expected ('in'|'out', progress, index) or ('hidden',), got {reveal!r}")
        rp, rdx, glow = _reveal_px(p.seed, int(reveal[2]) if len(reveal) > 2 else 0,
                                   (x0, y0, x1, y1), (Hp, Wp))
        a = _reveal_start(a, kind, q, rp, rdx, glow, (x0, y0, x1, y1))
        rv = (kind, q, rp)
    # body parts: which pixels each effect (and the glitch as a whole) may change
    only, protect = parse_parts(only), parse_parts(protect)
    tmasks, allow = {}, None
    if landmarks is not None and (only or protect or targets):
        lmk = np.asarray(landmarks, np.float32)
        mk = lambda parts: part_mask(lmk, parts, (Hp, Wp), (px, py), feather)   # noqa: E731
        if only:
            allow = mk(only)
        if protect:
            keep = 1 - mk(protect)
            allow = keep if allow is None else allow * keep
        for fx, parts in (targets or {}).items():
            if fx not in EFFECT_ORDER:
                raise ValueError(f"targets: unknown effect {fx!r}")
            parts = parse_parts(parts)
            if parts:
                tmasks[fx] = mk(parts)
    spill_m = None
    if pl is not None:                                      # subject onto the plate: effects see both
        spill_m = _spill_mask(a[..., 3], float(spill), (x0, y0, x1, y1))
        a = _over(a, pl)
    clean = a

    def run(name, img, c=None, mask=None):
        out = _FX[name](img, c or ctx)
        m = tmasks.get(name) if mask is None else mask
        return _blend(img, out, m) if m is not None else out
    space = None
    if yaw is not None or pitch or roll:
        space = Space3D(math.radians(float(yaw or 0.0)) * float(lock), cam_dist, depth,
                        math.radians(float(pitch or 0.0)), math.radians(float(roll or 0.0)), tilt)
    hist = None
    if history is not None and any(n not in skip for n in TIME_FX):
        def prep(im):
            if isinstance(im, np.ndarray) != is_float:
                raise ValueError("history frames must be the same type as img")
            return np.pad(_to_work(im, is_float), ((py, py), (px, px), (0, 0)))
        hist = _History(history, prep, a, history_key, fi)
    ctx = Ctx(p, t, fi, (x0, y0, x1, y1), int(Wp * 0.03) if bleed is None else int(bleed), space,
              hist=hist)
    if hist is not None:                                    # time effects: on the clean frame, first
        for name in TIME_FX:
            if name not in skip:
                a = run(name, a)

    if space is None:
        for name in EFFECT_ORDER:
            if name not in skip and name not in TIME_FX:
                a = run(name, a)
    else:
        # 3-D effects run in an upright working frame (roll undone), screen effects afterwards
        rolled = abs(math.degrees(space.roll)) > 1e-3
        centre = ((x0 + x1) / 2, (y0 + y1) / 2)
        work = _rotate_rgba(a, math.degrees(space.roll), centre) if rolled else a
        base = work
        for name in EFFECT_ORDER:
            if name in _LAYERED and name not in skip:
                before = work
                front = _FX[name](work, ctx.layered("front"))
                back = _FX[name](work, ctx.layered("back"))
                work = front if back is work else _under(front, _ghost(back, work))
                if name in tmasks:                          # mask turned into the upright frame
                    m = tmasks[name]
                    if rolled:
                        m = np.asarray(Image.fromarray(m, "F").rotate(
                            math.degrees(space.roll), resample=Image.BILINEAR, center=centre), np.float32)
                    work = _blend(before, work, m)
        if rolled:      # rotate only what the glitch changed back, untouched pixels stay exact
            changed = np.any(work != base, axis=2)
            if changed.any():
                ret = _rotate_rgba(work, -math.degrees(space.roll), centre)
                m = Image.fromarray(changed.astype(np.uint8) * 255).rotate(
                    -math.degrees(space.roll), resample=Image.BILINEAR, center=centre)
                a = np.where((np.asarray(m) > 0)[..., None], ret, a)
        else:
            a = work
        for name in EFFECT_ORDER:
            if name not in _LAYERED and name not in skip and name not in TIME_FX:
                a = run(name, a)
    if allow is not None:
        a = _blend(clean, a, allow)
    if pl is not None:
        a = _blend(clean, a, spill_m)                       # plate damage only near the subject
        if rv is not None:                                  # not built yet / gone: the clean plate
            a = _blend(pl, a, _reveal_end_mul(*rv))
        a = _over(a, pl)                                    # no holes (e.g. rotated corners)
    elif rv is not None:
        a = _reveal_end(a, *rv)

    if is_float:
        if trim:
            ys, xs = np.nonzero(a[..., 3] > 0)
            if ys.size:
                a = a[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
        return a
    out = Image.fromarray(a, "RGBA")
    if trim:
        box = out.getchannel("A").getbbox()
        if box:
            out = out.crop(box)
    return out


# --------------------------------------------------------------------------- sequence plumbing
def _natural_key(path) -> list:
    return [int(s) if s.isdigit() else s.lower() for s in re.split(r"(\d+)", Path(path).name)]


# --------------------------------------------------------------------------- frame I/O (PNG / EXR)
EXR_COMPRESSIONS = {"none": "NO_COMPRESSION", "zip": "ZIP_COMPRESSION", "zips": "ZIPS_COMPRESSION",
                    "piz": "PIZ_COMPRESSION", "dwaa": "DWAA_COMPRESSION", "dwab": "DWAB_COMPRESSION",
                    "rle": "RLE_COMPRESSION"}
IO_DEFAULTS = dict(exr_space="linear", exr_alpha="premultiplied", exr_depth="half",
                   exr_compression="zip")


def _openexr():
    try:
        import OpenEXR
        return OpenEXR
    except ImportError:
        raise RuntimeError("EXR files need the OpenEXR package:  pip install OpenEXR") from None


def srgb_encode(x):
    """Linear -> sRGB-encoded. Continues the curve above 1 (HDR survives) and mirrors negatives."""
    ax = np.abs(x)
    return np.sign(x) * np.where(ax <= 0.0031308, 12.92 * ax, 1.055 * np.power(ax, 1 / 2.4) - 0.055)


def srgb_decode(y):
    ay = np.abs(y)
    return np.sign(y) * np.where(ay <= 0.04045, ay / 12.92, np.power((ay + 0.055) / 1.055, 2.4))


def _exr_rgba(path):
    """-> (raw RGBA float32 (H, W, 4) as stored, alpha present?) in the file's display window.
    Finds R, G, B, A channels, plain or inside a layer (e.g. 'beauty.R'); data windows smaller or
    larger than the display window are placed correctly."""
    OpenEXR = _openexr()
    f = OpenEXR.File(str(path), separate_channels=True)
    ch = f.channels()
    low = {k.lower(): k for k in ch}
    layer = None
    for pref in ("", "rgba.", "beauty.", "rgb."):
        if all(pref + c in low for c in "rgb"):
            layer = pref
            break
    if layer is None:
        for k in sorted(low):
            if k.endswith(".r") and all(k[:-1] + c in low for c in "gb"):
                layer = k[:-1]
                break
    if layer is None:
        raise ValueError(f"{path}: no R/G/B channels found (has: {', '.join(sorted(ch))})")
    get = lambda c: np.asarray(ch[low[layer + c]].pixels, dtype=np.float32)
    has_a = layer + "a" in low
    planes = [get("r"), get("g"), get("b"), get("a") if has_a else np.ones_like(get("r"))]
    data = np.stack(planes, -1)
    hdr = f.header()
    try:
        dw = np.asarray(hdr["dataWindow"], dtype=np.int64).reshape(2, 2)
        disp = np.asarray(hdr["displayWindow"], dtype=np.int64).reshape(2, 2)
    except (KeyError, ValueError, TypeError):
        return data, has_a
    W, H = int(disp[1, 0] - disp[0, 0] + 1), int(disp[1, 1] - disp[0, 1] + 1)
    if data.shape[:2] == (H, W) and (dw == disp).all():
        return data, has_a
    out = np.zeros((H, W, 4), np.float32)                  # place the data window in the frame
    ox, oy = int(dw[0, 0] - disp[0, 0]), int(dw[0, 1] - disp[0, 1])
    sx0, sy0 = max(0, -ox), max(0, -oy)
    dx0, dy0 = max(0, ox), max(0, oy)
    w_ = min(data.shape[1] - sx0, W - dx0)
    h_ = min(data.shape[0] - sy0, H - dy0)
    if w_ > 0 and h_ > 0:
        out[dy0:dy0 + h_, dx0:dx0 + w_] = data[sy0:sy0 + h_, sx0:sx0 + w_]
    return out, has_a


def exr_to_work(raw, space="linear", alpha="premultiplied"):
    """File values -> the glitch's working image: straight alpha, display-encoded, 0..255 scale
    (values above 255 are HDR highlights and are kept)."""
    a = np.clip(raw[..., 3], 0, 1)
    rgb = raw[..., :3].astype(np.float32)
    if alpha == "premultiplied":
        rgb = np.where(a[..., None] > 1e-6, rgb / np.maximum(a[..., None], 1e-6), 0)
    if space == "linear":
        rgb = srgb_encode(rgb)
    return np.dstack([rgb * 255.0, a * 255.0]).astype(np.float32)


def work_to_exr(img, space="linear", alpha="premultiplied"):
    """Inverse of exr_to_work."""
    rgb = img[..., :3].astype(np.float32) / 255.0
    a = np.clip(img[..., 3].astype(np.float32) / 255.0, 0, 1)
    if space == "linear":
        rgb = srgb_decode(rgb)
    if alpha == "premultiplied":
        rgb = rgb * a[..., None]
    return np.dstack([rgb, a]).astype(np.float32)


def load_frame(path, io=None):
    """-> (image, raw): a PIL RGBA image for 8-bit formats, or for EXR a float32 working array
    (see glitch()) plus the raw file values (for exact pass-through of untouched pixels)."""
    io = {**IO_DEFAULTS, **(io or {})}
    if Path(path).suffix.lower() == ".exr":
        raw, _ = _exr_rgba(path)
        return exr_to_work(raw, io["exr_space"], io["exr_alpha"]), raw
    with Image.open(path) as im:
        im.load()
        return im, None


def to_display8(img) -> Image.Image:
    """Working image (PIL or float array) -> 8-bit RGBA PIL image for viewing / PNG."""
    if isinstance(img, Image.Image):
        return img.convert("RGBA")
    return Image.fromarray(np.clip(np.asarray(img) + 0.5, 0, 255).astype(np.uint8), "RGBA")


def resize_work(img, size):
    """Resize a working image (PIL or float array) with premultiplied filtering."""
    if isinstance(img, Image.Image):
        return img.resize(size, Image.LANCZOS)
    a = np.clip(img[..., 3:4], 0, 255) / 255.0
    pm = np.concatenate([img[..., :3] * a, img[..., 3:4]], -1).astype(np.float32)
    r = np.stack([np.asarray(Image.fromarray(np.ascontiguousarray(pm[..., i]), "F")
                             .resize(size, Image.LANCZOS)) for i in range(4)], -1)
    al = np.clip(r[..., 3], 0, 255)
    rgb = np.where(al[..., None] > 0.5, r[..., :3] * 255.0 / np.maximum(al[..., None], 1e-3), 0)
    return np.dstack([np.clip(rgb, 0, float(img[..., :3].max())), al]).astype(np.float32)


def save_frame(img, dst, io=None, raw=None, src=None, pad=(0, 0)):
    """Write a glitched frame. PNG/TIFF/WebP: 8-bit, straight alpha. EXR: half (or 32-bit) float,
    linear + premultiplied unless told otherwise. With `raw` (the source EXR's values) and `src`
    (the working input), pixels the glitch left untouched are copied from the source file exactly."""
    io = {**IO_DEFAULTS, **(io or {})}
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.stem + ".tmp" + dst.suffix)
    if dst.suffix.lower() != ".exr":
        to_display8(img).save(tmp, compress_level=3) if dst.suffix.lower() == ".png" else to_display8(img).save(tmp)
        os.replace(tmp, dst)
        return dst
    OpenEXR = _openexr()
    work = np.asarray(img, dtype=np.float32)                 # PIL 8-bit converts to 0..255 floats
    out = work_to_exr(work, io["exr_space"], io["exr_alpha"])
    if raw is not None and src is not None:
        py, px = pad
        rawp = np.pad(raw, ((py, py), (px, px), (0, 0)))
        srcp = np.pad(np.where(src[..., 3:4] > 0, src, 0), ((py, py), (px, px), (0, 0)))
        if rawp.shape == out.shape:
            same = np.all(work == srcp, axis=2)
            out[same] = rawp[same]
    dt = np.float16 if io["exr_depth"] == "half" else np.float32
    comp = getattr(OpenEXR, EXR_COMPRESSIONS[io["exr_compression"]])
    chans = {c: np.ascontiguousarray(out[..., i].astype(dt)) for i, c in enumerate("RGBA")}
    OpenEXR.File({"compression": comp, "type": OpenEXR.scanlineimage}, chans).write(str(tmp))
    os.replace(tmp, dst)
    return dst


def _scan(path):
    """(subject bbox or None, (W,H)) for one frame."""
    if Path(path).suffix.lower() == ".exr":
        raw, _ = _exr_rgba(path)
        al = Image.fromarray((np.clip(raw[..., 3], 0, 1) * 255 + 0.5).astype(np.uint8), "L")
        return al.point(lambda v: 255 if v > 8 else 0).getbbox(), al.size
    with Image.open(path) as im:
        al = im.convert("RGBA").getchannel("A")
        return al.point(lambda v: 255 if v > 8 else 0).getbbox(), im.size


def _plan_boxes(scans, fps: float, follow: float, override, scales=None):
    """Per-frame layout box. Fixed union box by default; with `follow`, a box that tracks the
    subject's (smoothed) centre. Its size is constant, or - with `scales` (relative apparent size
    per frame, e.g. from estimate_pose.py) - a constant body-relative size times that scale, so the
    glitch grows and shrinks with the person."""
    n = len(scans)
    sizes = {s for _, s in scans}
    if len(sizes) != 1:
        raise ValueError(f"all frames must have the same size, found: {sorted(sizes)[:4]}")
    W, H = next(iter(sizes))
    if override:
        return [tuple(override)] * n, (W, H)
    boxes = [b for b, _ in scans]
    valid = [b for b in boxes if b]
    if not valid:
        return [(0, 0, W, H)] * n, (W, H)
    union = (min(b[0] for b in valid), min(b[1] for b in valid),
             max(b[2] for b in valid), max(b[3] for b in valid))
    if follow <= 0:
        return [union] * n, (W, H)
    arr = np.array([[(b[0] + b[2]) / 2, (b[1] + b[3]) / 2, b[2] - b[0], b[3] - b[1]]
                    if b else [np.nan] * 4 for b in boxes], float)
    idx = np.arange(n)
    ok = ~np.isnan(arr[:, 0])
    for j in range(4):
        arr[:, j] = np.interp(idx, idx[ok], arr[ok, j])     # hold/interpolate over empty frames
    sig = follow * fps
    if sig > 0.3 and n > 1:
        r = int(3 * sig) + 1
        k = np.exp(-0.5 * (np.arange(-r, r + 1) / sig) ** 2)
        k /= k.sum()
        for j in (0, 1):
            arr[:, j] = np.convolve(np.pad(arr[:, j], r, mode="edge"), k, mode="valid")
    if scales is None:
        wr, hr = np.percentile(arr[:, 2], 90), np.percentile(arr[:, 3], 90)
        return [(cx - wr / 2, cy - hr / 2, cx + wr / 2, cy + hr / 2) for cx, cy in arr[:, :2]], (W, H)
    sc = np.asarray(scales, float)
    wr, hr = np.percentile(arr[:, 2] / sc, 90), np.percentile(arr[:, 3] / sc, 90)   # size at scale 1
    return [(cx - wr * k / 2, cy - hr * k / 2, cx + wr * k / 2, cy + hr * k / 2)
            for (cx, cy), k in zip(arr[:, :2], sc)], (W, H)


class SeqHistory:
    """history= for glitch(): earlier source files of the sequence (nearest first), loaded on
    demand; optionally resized (GUI previews)."""

    _cache: "OrderedDict" = None          # per process: neighbouring frames get asked for again
    _bytes = 0
    BUDGET = 100_000_000

    def __init__(self, paths, io=None, size=None, plates=None, frames=None):
        self.paths, self.io, self.size = list(paths), io, size
        self.frames = list(frames) if frames is not None else None   # their frame numbers
        self.plates = list(plates) if plates else None

    def __call__(self, k: int):
        if not 1 <= k <= len(self.paths):
            return None
        cls = SeqHistory
        if cls._cache is None:
            from collections import OrderedDict
            cls._cache = OrderedDict()
        pk = str(self.plates[k - 1]) if self.plates and k - 1 < len(self.plates) else None
        ck = (str(self.paths[k - 1]), self.size, json.dumps(self.io, sort_keys=True), pk)
        if ck in cls._cache:
            cls._cache.move_to_end(ck)
            return cls._cache[ck]
        img, _ = load_frame(self.paths[k - 1], self.io)
        if pk is not None:                                  # comp mode: earlier frames on their plates
            nat = (img.shape[1], img.shape[0]) if isinstance(img, np.ndarray) else img.size
            img = composite_on(img, load_plate(pk, self.io, img, nat))
        img = resize_work(img, self.size) if self.size else img
        nb = img.nbytes if isinstance(img, np.ndarray) else img.width * img.height * 4
        cls._cache[ck] = img
        cls._bytes += nb
        while cls._bytes > cls.BUDGET and len(cls._cache) > 1:
            _k, old = cls._cache.popitem(last=False)
            cls._bytes -= old.nbytes if isinstance(old, np.ndarray) else old.width * old.height * 4
        return img

    def key(self, k: int):
        if not 1 <= k <= len(self.paths):
            return None
        pk = str(self.plates[k - 1]) if self.plates and k - 1 < len(self.plates) else None
        return (str(self.paths[k - 1]), self.size, pk)


# --------------------------------------------------------------------------- glitch pass (matte)
PASS_LO, PASS_HI = 2.0, 64.0      # 8-bit difference: <= LO unchanged (rounding), >= HI white


def glitch_pass(out, img, plate=None, pad=(0, 0)) -> np.ndarray:
    """Matte of the pixels the glitch made, float32 0..1 at the output's size.

    A pixel counts as glitch when the output is neither the clean subject (the input on the same
    canvas, over the plate in comp mode) nor the clean background (transparency, or the plate).
    So blocks, smears, echoes, held frames, transitions and spill into the plate are white;
    the untouched subject, holes that show the background, and empty frames are black.
    Grey levels say how strongly a pixel changed (linear from PASS_LO to PASS_HI levels of 255),
    so the comp can pick its own threshold: a Levels / Threshold on the pass keeps only the hard
    glitches (displaced blocks, fills, smears) and drops subtle compression haze. Measured on
    premultiplied 8-bit display values, so EXR and PNG renders give the same matte."""
    o = np.asarray(to_display8(out), np.float32)
    if plate is not None:
        clean = np.asarray(to_display8(composite_on(img, plate)), np.float32)
        bg = np.asarray(to_display8(plate), np.float32)
    else:
        py, px = pad
        clean = np.pad(np.asarray(to_display8(img), np.float32), ((py, py), (px, px), (0, 0)))
        bg = np.zeros_like(clean)
    if clean.shape != o.shape:
        raise ValueError(f"glitch pass: output is {o.shape[1]}x{o.shape[0]}, the clean frame "
                         f"{clean.shape[1]}x{clean.shape[0]} (--trim can't be combined with it)")

    def pm(x):
        return np.concatenate([x[..., :3] * (x[..., 3:] / 255.0), x[..., 3:]], -1)

    O = pm(o)
    d = np.minimum(np.abs(O - pm(clean)).max(-1), np.abs(O - pm(bg)).max(-1))
    return np.clip((d - PASS_LO) / (PASS_HI - PASS_LO), 0.0, 1.0).astype(np.float32)


def save_pass(m, dst, io=None):
    """Write a matte: 8-bit greyscale PNG/TIFF/WebP, or EXR with R=G=B=A=matte (so Fusion /
    Nuke can use it as luminance or alpha)."""
    io = {**IO_DEFAULTS, **(io or {})}
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.stem + ".tmp" + dst.suffix)
    if dst.suffix.lower() != ".exr":
        Image.fromarray((np.clip(m, 0, 1) * 255 + 0.5).astype(np.uint8), "L").save(tmp)
    else:
        OpenEXR = _openexr()
        dt = np.float16 if io["exr_depth"] == "half" else np.float32
        ch = np.ascontiguousarray(m.astype(dt))
        comp = getattr(OpenEXR, EXR_COMPRESSIONS[io["exr_compression"]])
        OpenEXR.File({"compression": comp, "type": OpenEXR.scanlineimage},
                     {c: ch for c in "RGBA"}).write(str(tmp))
    os.replace(tmp, dst)
    return dst


def _render(job):
    src, dst, frame, bbox, kw, ang, camd, io, fo, prev = job
    yaw, pitch, roll = ang if ang is not None else (None, None, None)
    img, raw = load_frame(src, io)
    fo = dict(fo)
    pass_dst = fo.pop("pass_dst", None)
    plate_path, prev_plates = fo.pop("plate", None), fo.pop("prev_plates", None)
    prev_frames = fo.pop("prev_frames", None)
    H, W = (img.shape[:2] if isinstance(img, np.ndarray) else (img.height, img.width))
    plate = load_plate(plate_path, io, img, (W, H)) if plate_path else None
    out = render(img, t=frame / kw["fps"], frame=frame, bbox=bbox, yaw=yaw, pitch=pitch,
                 roll=roll, cam_dist=camd, plate=plate,
                 history=SeqHistory(prev, io, plates=prev_plates, frames=prev_frames) if prev else None,
                 history_key=(str(src), None, str(plate_path) if plate_path else None), **{**kw, **fo})
    pad = (0, 0) if plate is not None else (int(H * kw["pad"]), int(W * kw["pad"]))
    if pass_dst:
        save_pass(glitch_pass(out, img, plate, pad), pass_dst, io)
    exact = raw is not None and not kw.get("trim") and plate is None
    return save_frame(out, dst, io, raw=raw if exact else None, src=img if exact else None, pad=pad)


def _render_chunk(jobs):
    """Render consecutive frames in one worker (time effects reuse loaded neighbours/motion)."""
    return [_render(j) for j in jobs]


# --------------------------------------------------------------------------- render pool
# Measured high-water mark of one render worker over a long run (time effects, glitch pass,
# EXR): ~650 MB at 1920x1080, ~1.5 GB at 3840x2160, i.e. about 360 MB + 140 bytes per canvas
# pixel. The estimate adds ~20 % headroom (plates, big look-backs).
WORKER_BASE_BYTES = 450_000_000
WORKER_BYTES_PER_PX = 170
MEMORY_SHARE = 0.75                   # of the memory free at the start, for the render workers
NUMBA_WORKER_BYTES = 90_000_000       # Numba + LLVM + the loaded kernels, per worker (measured ~78 MB)


def _numba_expected() -> bool:
    """Will the workers load the compiled kernels? (Asked without importing Numba here.)"""
    if os.environ.get("GLITCHER_NUMBA", "1").strip().lower() in ("0", "off", "no", "false"):
        return False
    import importlib.util
    try:
        return importlib.util.find_spec("numba") is not None
    except (ImportError, ValueError):
        return False


def worker_bytes(canvas_px: int) -> int:
    """Estimated high-water memory of one render worker."""
    return int(WORKER_BASE_BYTES + WORKER_BYTES_PER_PX * max(1, canvas_px)
               + (NUMBA_WORKER_BYTES if _numba_expected() else 0))


def available_memory() -> int | None:
    """Free physical memory in bytes, or None if this system won't say."""
    try:
        if sys.platform == "win32":
            import ctypes

            class MS(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
            ms = MS()
            ms.dwLength = ctypes.sizeof(MS)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(ms)):
                return int(ms.ullAvailPhys)
            return None
        if sys.platform.startswith("linux"):
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemAvailable:"):
                        return int(line.split()[1]) * 1024
            return None
        total = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")   # macOS: no "free" figure
        return int(total * 0.6)
    except (OSError, ValueError, AttributeError):
        return None


def safe_workers(requested: int, canvas_px: int, avail: int | None = None) -> int:
    """How many render workers fit in memory (at least 1, at most `requested`)."""
    avail = available_memory() if avail is None else avail
    if not avail:
        return max(1, requested)
    return max(1, min(requested, int(avail * MEMORY_SHARE // worker_bytes(canvas_px))))


class RenderCrash(RuntimeError):
    """A worker process died even when rendering alone: a crash, not a lack of memory."""


def _worker_init(crash_log, threads=None):
    """Each worker writes a Python trace to crash_log if it dies of a native crash; threads:
    the native engine's threads in this worker."""
    import faulthandler
    if threads:
        set_engine_threads(threads)
    try:
        faulthandler.enable(open(crash_log, "a", encoding="utf-8"))   # noqa: SIM115 - lives with the process
    except OSError:
        pass


def _tail(path, n=12):
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            return "".join(f.readlines()[-n:]).strip()
    except OSError:
        return ""


def canvas_px(work) -> int:
    """Pixels of one output canvas (input size grown by the pad), for the memory estimate."""
    if not work:
        return 1
    try:
        with Image.open(work[0][0]) as im:
            w, h = im.size
    except OSError:
        return 1
    pad = float((work[0][4] or {}).get("pad") or 0.0)
    return int(w * (1 + 2 * pad)) * int(h * (1 + 2 * pad))


def render_pool(work, workers: int, progress=None, cancel=None, log=None, chunk_fn=None) -> int:
    """Render `work` (planned jobs) with up to `workers` processes; returns the frames done.

    Consecutive frames go to the same worker (the time effects reuse loaded neighbours). If a
    worker process dies - usually the system ran out of memory and killed it - the frames not
    finished yet are rendered again with half the workers, down to one, then one frame at a
    time. Only a frame that crashes a lone worker is an error (RenderCrash, with the frame and
    the worker's trace). Frames are independent, so a retry gives the same pixels.
    chunk_fn replaces the per-worker function (tests use it to kill workers on purpose)."""
    log = log or (lambda m: print(m, file=sys.stderr))
    threads, ctx = None, None
    if chunk_fn is None and native():
        # The engine is multithreaded itself: a few workers (file reading and writing overlap)
        # sharing the cores, rather than one per core. Fresh processes, not forked ones: a fork
        # of a process whose engine threads have started would wait on threads it lacks.
        cpus = os.cpu_count() or 1
        workers = max(1, min(int(workers), max(1, cpus // 4)))
        threads = max(1, cpus // workers)
        import multiprocessing as mp
        ctx = mp.get_context("spawn")
    chunk_fn = chunk_fn or _render_chunk
    n = len(work)
    crash_log = os.path.join(tempfile.gettempdir(), f"glitcher_worker_{os.getpid()}.log")
    pending = list(range(n))                                  # indexes into work, in order
    workers, solo, cause = max(1, int(workers)), False, None
    try:
        while pending:
            ch = 1 if solo else max(1, min(8, -(-len(pending) // (workers * 3))))
            chunks = [pending[i:i + ch] for i in range(0, len(pending), ch)]
            ex = ProcessPoolExecutor(workers, mp_context=ctx, initializer=_worker_init,
                                     initargs=(crash_log, threads))
            futs = [(ex.submit(chunk_fn, [work[i] for i in c]), c) for c in chunks]
            broken = None
            try:
                for f, c in futs:
                    while not f.done():
                        if cancel is not None and cancel.is_set():
                            return n - len(pending)
                        time.sleep(0.05)
                    try:
                        f.result()                            # a frame's own error is raised here
                    except (BrokenProcessPool, MemoryError) as e:
                        broken, cause = c, e                  # killed, or out of memory: fewer workers
                        break
                    pending = [i for i in pending if i not in set(c)]
                    if progress:
                        progress(n - len(pending))
            finally:
                ex.shutdown(wait=broken is None and not (cancel is not None and cancel.is_set()),
                            cancel_futures=True)
            if broken is None:
                continue
            finished = {i for f, c in futs if f.done() and not f.cancelled() and f.exception() is None
                        for i in c}
            pending = [i for i in pending if i not in finished]
            if progress:
                progress(n - len(pending))
            if solo:
                name = Path(work[broken[0]][0]).name
                if isinstance(cause, MemoryError):
                    raise RenderCrash(f"not enough memory to render {name} even one frame at a time "
                                      f"({cause}). Close other programs, or lower the canvas padding.")
                trace = _tail(crash_log)
                raise RenderCrash(f"rendering {name} crashed the render process even on its own, so "
                                  f"it is not a memory problem."
                                  + (f"\n\nWorker trace:\n{trace}" if trace else ""))
            if workers == 1:
                solo = True
            workers = max(1, workers // 2)
            log(("a render worker ran out of memory" if isinstance(cause, MemoryError) else
                 "a render worker died (most likely out of memory)") + "; rendering the remaining "
                f"{len(pending)} frame(s) with {workers} worker(s)" + (", one frame at a time" if solo else ""))
        return n
    finally:
        try:
            os.remove(crash_log)
        except OSError:
            pass


def _gather(inp: str):
    """-> (kind, [paths]) where kind is 'dir', 'glob' or 'file'."""
    pth = Path(inp)
    if pth.is_file():
        return "file", [pth]
    if pth.is_dir():
        files = [f for f in pth.iterdir() if f.suffix.lower() in IMG_EXT]
        return "dir", sorted(files, key=_natural_key)
    if any(ch in inp for ch in "*?["):
        files = [Path(f) for f in _glob.glob(inp) if Path(f).suffix.lower() in IMG_EXT]
        return "glob", sorted(files, key=_natural_key)
    return "file", [pth]


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="glitch.py",
        description="Temporally coherent glitch for transparent PNG cutouts / image sequences.",
        epilog=f"{APP_NAME} {__version__} - {PROJECT_URL} (downloads: {PROJECT_URL}/releases/latest)")
    ap.add_argument("--version", action="version", version=f"{APP_NAME} {__version__} (glitch.py)")
    ap.add_argument("input", help="PNG file, folder of frames, or quoted glob such as 'f_*.png'")
    ap.add_argument("-o", "--output", type=Path,
                    help="output folder (sequences) or file (single still); default: next to input")
    ap.add_argument("-i", "--intensity", type=float, default=0.5, help="0..1 (default 0.5)")
    ap.add_argument("-s", "--seed", type=int, default=None, help="RNG seed (default: random, printed)")
    ap.add_argument("--fps", type=float, default=25.0, help="frame rate of the sequence (default 25)")
    ap.add_argument("--tempo", type=float, default=1.0, help=">1 shorter/snappier events (default 1)")
    ap.add_argument("--density", type=float, default=1.0, help="event-count multiplier (default 1)")
    ap.add_argument("--motion", type=float, default=0.0, metavar="0..1",
                    help="motion-reactive: the glitch settles when the subject holds still and "
                         "spikes on fast movement (0 = off, 1 = fully driven by motion)")
    ap.add_argument("--bursts", type=float, default=0.5,
                    help="0 = constant activity, 1 = strong calm/burst cycles (default 0.5)")
    ap.add_argument("--follow", type=float, default=None, metavar="SEC",
                    help="glitch layout follows the subject, smoothed over SEC seconds "
                         "(0 = fixed on screen; default 0, or 0.4 with --pose)")
    ap.add_argument("--bbox", help="x0,y0,x1,y1 fixed layout box in input pixels (overrides "
                                   "detection and follow mode)")
    ap.add_argument("--boxes", metavar="FILE",
                    help="per-frame layout boxes: read from FILE if it exists, else save them there. "
                         "Render the whole shot once with it; partial re-renders then match exactly, "
                         "also in follow / 3D mode")
    ap.add_argument("--offset", type=int, default=0, help="add to every frame index (default 0)")
    ap.add_argument("--use-numbers", action="store_true",
                    help="take frame index from the trailing number in each filename")
    ap.add_argument("-n", "--frames", type=int, default=1,
                    help="for a single still: render this many time steps (default 1)")
    ap.add_argument("--pad", type=float, default=0.0,
                    help="canvas padding per side, as a share of the frame (default 0: output = input "
                         "size; e.g. 0.12 leaves room for glitch reaching past the frame edge)")
    ap.add_argument("--tint", type=float, default=0.0, help="purple/cyan blend 0..1 (default 0)")
    ap.add_argument("--bleed", type=int, default=None, help="px JPEG junk may spill outside body")
    ap.add_argument("--jpeg-quality", type=int, default=None,
                    help="1..100, lower = blockier (unset or 0 = automatic, from intensity)")
    ap.add_argument("--skip", default="", help=f"comma list from: {','.join(EFFECT_ORDER)}")
    ap.add_argument("--gain", default="", metavar="FX=S,...",
                    help="per-effect strength 0..3, e.g. jpeg=1.5,slices=0.5,noise=0 (1 = default, "
                         "0 = off); event effects get more and bigger events")
    ap.add_argument("--trim", action="store_true", help="crop to visible pixels (single frame only)")
    ap.add_argument("--range", metavar="FIRST-LAST",
                    help="render only these frame numbers, planned with the whole sequence: layout and "
                         "time effects (which read earlier frames) match a full render exactly")
    ga = ap.add_argument_group("sound-reactive")
    ga.add_argument("--audio", metavar="FILE",
                    help="WAV or MP3 (MP3/FLAC/OGG need the miniaudio package or ffmpeg): loudness "
                         "drives intensity, beats trigger bursts")
    ga.add_argument("--audio-start", type=float, default=0.0, metavar="SEC",
                    help="where in the audio the passage starts; it lines up with the first frame")
    ga.add_argument("--audio-end", type=float, default=None, metavar="SEC",
                    help="end of the passage (default: end of the file); after it the audio is ignored")
    ga.add_argument("--audio-strength", type=float, default=0.8, metavar="0..1",
                    help="how strongly the audio drives the glitch (0 = off, default 0.8)")
    ga.add_argument("--audio-sens", type=float, default=0.5, metavar="0..1",
                    help="sensitivity: higher = quieter passages count as loud and more onsets count "
                         "as beats (default 0.5)")
    gp = ap.add_argument_group("background plate")
    gp.add_argument("--plate", metavar="PATH",
                    help="clean background (folder of frames, glob or one image): the subject is "
                         "composited onto it and the glitch spills into it; output is the opaque comp")
    gp.add_argument("--spill", type=float, default=0.05, metavar="0..1",
                    help="how far the glitch may damage the plate around the subject (0 = subject only)")
    gb = ap.add_argument_group("body parts (need landmarks from estimate_pose.py)")
    gb.add_argument("--only", default="", metavar="PARTS",
                    help="glitch only these parts: face, head, torso, arms, hands, legs, feet, "
                         "upper, lower (comma-separated)")
    gb.add_argument("--protect", default="", metavar="PARTS", help="keep these parts clean, e.g. face")
    gb.add_argument("--target", action="append", default=[], metavar="EFFECT=PARTS",
                    help="limit one effect to parts, repeatable: --target mosh=head --target echo=arms,hands")
    gb.add_argument("--part-feather", type=float, default=1.0, metavar="X",
                    help="softness of the part edges (1 = default)")
    gb.add_argument("--landmarks", metavar="FILE",
                    help="landmarks file (default: <pose>_landmarks.npz next to --pose)")
    gt = ap.add_argument_group("timing")
    gt.add_argument("--curve", metavar="FILE",
                    help="keyframes over frames: CSV with a header 'frame' plus any of intensity, effect "
                         "names (strength multipliers, 1 = as set), hit, hit_len; linear between keys")
    gt.add_argument("--hits", default="", metavar="F[:S[:SEC]],...",
                    help="forced bursts at frames, e.g. 48,112:0.6,200:1:0.8 (strength 0..1, length s)")
    gt.add_argument("--hit-len", type=float, default=0.5, metavar="SEC",
                    help="default hit length in seconds (default 0.5)")
    gt.add_argument("--appear", default="", metavar="F[:SEC],...",
                    help="materialise at these frames: the subject builds up out of glitch "
                         "structure (invisible before the first appear), e.g. 0 or 12:1.5")
    gt.add_argument("--vanish", default="", metavar="F[:SEC],...",
                    help="dematerialise at these frames: breaks apart and is gone afterwards")
    gt.add_argument("--trans-len", type=float, default=1.0, metavar="SEC",
                    help="default appear / vanish length in seconds (default 1.0)")
    ap.add_argument("-j", "--jobs", type=int, default=0,
                    help="worker processes (0 = auto: every CPU, fewer if memory is short)")
    ge = ap.add_argument_group("output format / EXR")
    ge.add_argument("--format", choices=["auto", "png", "exr"], default="auto",
                    help="output file type (default auto: same as the input, EXR stays EXR)")
    ge.add_argument("--exr-space", choices=["linear", "display"], default="linear",
                    help="EXR colour: linear (default, the EXR convention; glitch maths happens in "
                         "sRGB-encoded values and is converted back) or display (already encoded)")
    ge.add_argument("--exr-alpha", choices=["premultiplied", "straight"], default="premultiplied",
                    help="EXR alpha convention for reading and writing (default premultiplied)")
    ge.add_argument("--exr-depth", choices=["half", "float"], default="half",
                    help="EXR output precision: 16-bit half (default) or 32-bit float")
    ge.add_argument("--exr-compression", choices=sorted(EXR_COMPRESSIONS), default="zip",
                    help="EXR compression (default zip, lossless; dwaa/dwab are lossy and small)")
    ge.add_argument("--glitch-pass", nargs="?", const="", default=None, metavar="DIR",
                    help="also write a glitch matte for every frame: white where the glitch made "
                         "the pixel (blocks, smears, echoes, spill...), black where it is the clean "
                         "subject or background, grey = changed a little. Use it to glow, grade or "
                         "regrain only the glitch in the comp. Same file names, in DIR (default: a "
                         "glitch_pass folder inside the output folder)")
    g3 = ap.add_argument_group("3D mode")
    g3.add_argument("--pose", type=Path,
                    help="file with 'frame, yaw[, pitch[, roll]]' rows in degrees (a header may name "
                         "the columns, e.g. estimate_pose.py output); enables 3D mode")
    g3.add_argument("--pitch", type=float, default=0.0,
                    help="add to pitch, deg (+ = camera above subject); alone it enables 3D mode")
    g3.add_argument("--roll", type=float, default=0.0,
                    help="add to roll, deg (+ = subject appears rotated clockwise); alone enables 3D")
    g3.add_argument("--no-pose-pitch", action="store_true",
                    help="ignore the pose file's pitch column (use --pitch alone)")
    g3.add_argument("--no-pose-roll", action="store_true",
                    help="ignore the pose file's roll column (use --roll alone)")
    g3.add_argument("--tilt", type=float, default=0.25,
                    help="share of glitch planes tilted toward horizontal, visible from high/low "
                         "angles (default 0.25; 0 = all vertical)")
    g3.add_argument("--pose-smooth", type=float, default=0.0, metavar="SEC",
                    help="Gaussian smoothing of all pose curves (yaw, pitch, roll, scale, cam_dist) "
                         "in seconds (default 0)")
    g3.add_argument("--pose-unwrap", action="store_true",
                    help="treat yaw jumps > 180 deg between keys as wrap-around (tracker data); "
                         "default takes keys literally, like animation curves. Only changes the "
                         "result where angles are blended: --pose-smooth, or keys not on every frame")
    g3.add_argument("--lock", type=float, default=1.0,
                    help="share of the turn the glitch volume follows: 1 glued, 0 ignores (default 1)")
    g3.add_argument("--cam-dist", type=float, default=None,
                    help="camera distance in layout-box heights (= subject heights in follow/3D mode); "
                         "smaller = stronger perspective "
                         "(default: the pose file's cam_dist column if present, else 3)")
    g3.add_argument("--no-pose-scale", action="store_true",
                    help="ignore the pose file's scale column (constant-size follow box)")
    g3.add_argument("--depth", type=float, default=0.35,
                    help="half-thickness of the glitch volume in subject widths (default 0.35)")
    return ap


class PlanError(ValueError):
    """Bad arguments or inputs, reported to the user (CLI: as a usage error)."""


def parse_gains(text: str) -> dict:
    """'jpeg=1.5,slices=0' -> {'jpeg': 1.5, 'slices': 0.0}"""
    out = {}
    for part in filter(None, (t.strip() for t in (text or "").split(","))):
        try:
            k, v = part.split("=")
            k, v = k.strip(), float(v)
        except ValueError:
            raise PlanError(f"--gain: expected effect=strength, got {part!r}") from None
        if k not in EFFECT_ORDER:
            raise PlanError(f"--gain: unknown effect {k!r}; valid: {','.join(EFFECT_ORDER)}")
        if v < 0:
            raise PlanError(f"--gain: strength must be >= 0 ({part})")
        out[k] = v
    return out


def plan_render(args, notes: list | None = None, scan_cache: dict | None = None,
                check_alpha: bool = True) -> dict:
    """Turn parsed arguments into render jobs (one per output frame). Shared by the CLI and the
    GUI, so both render exactly the same thing. `scan_cache` (path -> alpha scan) lets a caller
    re-plan cheaply after changing settings. Raises PlanError for bad inputs."""
    notes = [] if notes is None else notes
    if not args.fps > 0:
        raise PlanError("--fps must be > 0")
    for name, lo, hi in (("motion", 0, 1), ("spill", 0, 1), ("part_feather", 0, 10),
                         ("audio_strength", 0, 1), ("audio_sens", 0, 1)):
        v = getattr(args, name, None)
        if v is not None and not lo <= v <= hi:
            raise PlanError(f"--{name.replace('_', '-')} must be between {lo} and {hi}, got {v:g}")

    skip = [s.strip() for s in args.skip.split(",") if s.strip()]
    bad = [s for s in skip if s not in EFFECT_ORDER]
    if bad:
        raise PlanError(f"unknown effect(s) in --skip: {bad}; valid: {','.join(EFFECT_ORDER)}")
    gains = parse_gains(args.gain)
    io = dict(exr_space=args.exr_space, exr_alpha=args.exr_alpha, exr_depth=args.exr_depth,
              exr_compression=args.exr_compression)
    override = None
    if args.bbox:
        try:
            override = tuple(float(v) for v in args.bbox.split(","))
            assert len(override) == 4
        except Exception:
            raise PlanError("--bbox needs four comma-separated numbers: x0,y0,x1,y1") from None
    pose = None
    if args.pose:
        try:
            pose = Pose.from_file(args.pose, args.fps, args.pose_smooth, args.pose_unwrap)
        except (OSError, ValueError) as ex:
            raise PlanError(f"--pose: {ex}") from None
    is3d = pose is not None or args.pitch != 0 or args.roll != 0
    follow = args.follow if args.follow is not None else (0.4 if is3d else 0.0)

    kind, files = _gather(args.input)
    if not files or (kind == "file" and not files[0].is_file()):
        raise PlanError(f"no input images found for: {args.input}")
    seed = args.seed if args.seed is not None else int(np.random.SeedSequence().entropy % 2**31)

    # ---- items: (src, dst, frame_index)
    items = []
    in_ext = files[0].suffix.lower()
    ext = {"png": ".png", "exr": ".exr"}.get(args.format) or (".exr" if in_ext == ".exr" else ".png")

    def number(stem):
        m = re.findall(r"\d+", stem)
        return int(m[-1]) if m else None

    if kind == "file":
        src = files[0]
        n = max(1, args.frames)
        base = args.offset
        if args.use_numbers and number(src.stem) is not None:
            base = number(src.stem) + args.offset
        out = args.output
        if n == 1:
            if out is not None and (out.is_dir() or out.suffix == ""):
                out = out / (src.stem + "_glitch" + ext)          # -o names a folder
            elif out is not None and out.suffix.lower() not in IMG_EXT:
                raise PlanError(f"-o {out}: unsupported image type (use .png, .exr, .webp or .tif)")
            items.append((src, out or src.with_name(src.stem + "_glitch" + ext), base))
        else:
            if out is not None and out.suffix.lower() in IMG_EXT and not out.is_dir():
                raise PlanError(f"-o {out}: with --frames, -o must be a folder")
            odir = out or src.with_name(src.stem + "_glitch")
            items = [(src, odir / f"{src.stem}_{i + 1:04d}{ext}", base + i) for i in range(n)]
    else:
        here = Path(args.input).resolve() if kind == "dir" else None     # works for '.' too
        odir = args.output or (here.with_name(here.name + "_glitch") if kind == "dir"
                               else Path("glitch_out"))
        for i, f in enumerate(files):
            fr = i + args.offset
            if args.use_numbers and number(f.stem) is not None:
                fr = number(f.stem) + args.offset
            items.append((f, odir / (f.stem + ext), fr))
    if min(fr for _, _, fr in items) < 0:
        raise PlanError("frame indices must be >= 0")
    seen = {}
    for s_, d_, _ in items:
        key = str(Path(d_)).lower() if os.name == "nt" else str(d_)
        if key in seen:
            raise PlanError(f"two inputs would write the same output {d_}: {seen[key].name} and "
                            f"{Path(s_).name} (rename one, they differ only by extension/folder)")
        seen[key] = Path(s_)
    # never write over source material: the frames themselves (output folder = input folder,
    # or -o pointing at the input file)
    _ident = lambda q: os.path.normcase(os.path.realpath(q))          # noqa: E731
    sources = {_ident(s_) for s_, _, _ in items}
    for _s, d_, _ in items:
        if _ident(d_) in sources:
            raise PlanError(f"refusing to overwrite the input {d_}: choose an output folder that is "
                            f"not the input folder")
    if args.trim and len(items) > 1:
        raise PlanError("--trim changes frame size; use it for a single frame only")

    # ---- pass 1: layout boxes (cheap: alpha only)
    # every CPU by default (the pool fits the count to free memory); Windows pools max out at 61
    jobs = args.jobs or max(1, min(61, os.cpu_count() or 1))
    jobs = 1 if len(items) < 3 else jobs
    uniq = list(dict.fromkeys(s for s, _, _ in items))
    scan_of = {} if scan_cache is None else scan_cache
    todo = [u for u in uniq if u not in scan_of]
    if jobs > 1 and len(todo) > 1:
        with ProcessPoolExecutor(jobs) as ex:
            scan_of.update(zip(todo, ex.map(_scan, todo, chunksize=8)))
    else:
        scan_of.update((u, _scan(u)) for u in todo)
    scales = None
    if pose is not None and pose.has_scale and not args.no_pose_scale:
        scales = [pose.scale(fr) for _, _, fr in items]
        if follow <= 0:
            notes.append("note: pose scale is used by the follow box only; add --follow")
    try:
        boxes, (W, H) = _plan_boxes([scan_of[s] for s, _, _ in items], args.fps, follow, override, scales)
    except ValueError as ex:
        raise PlanError(str(ex)) from None
    if getattr(args, "boxes", None):
        bf = Path(args.boxes)
        if bf.is_file():                    # reuse the full render's layout -> partial renders match
            try:
                saved = {int(k): tuple(v) for k, v in json.loads(bf.read_text()).items()}
                boxes = [saved[fr] for _, _, fr in items]
            except KeyError as ex:
                raise PlanError(f"--boxes {bf}: no layout box for frame {ex}") from None
            except (ValueError, TypeError) as ex:
                raise PlanError(f"--boxes {bf}: unreadable ({ex})") from None
            notes.append(f"note: layout boxes read from {bf}")
        else:
            bf.parent.mkdir(parents=True, exist_ok=True)
            bf.write_text(json.dumps({str(fr): [float(v) for v in b]
                                      for (_, _, fr), b in zip(items, boxes)}))
            notes.append(f"note: layout boxes saved to {bf}")
    if follow > 0 and override:
        notes.append("note: --bbox given, --follow ignored")
    if check_alpha:
        try:
            img0, _ = load_frame(uniq[0], io)
        except RuntimeError as ex:                     # OpenEXR missing
            raise PlanError(str(ex)) from None
        except (OSError, ValueError) as ex:
            raise PlanError(f"cannot read {uniq[0]}: {ex}") from None
        amin = (float(np.asarray(img0)[..., 3].min()) if isinstance(img0, np.ndarray)
                else img0.convert("RGBA").getchannel("A").getextrema()[0])
        if amin >= 255:
            notes.append("warning: first frame has no transparency - the whole frame is treated "
                         "as the subject")

    kw = dict(seed=seed, intensity=args.intensity, fps=args.fps, tempo=args.tempo,
              density=args.density, bursts=args.bursts, pad=args.pad, tint=args.tint,
              skip=tuple(skip), bleed=args.bleed, quality=args.jpeg_quality, trim=args.trim,
              lock=args.lock, depth=args.depth, tilt=args.tilt, gains=gains)
    angs = []
    for _, _, fr in items:
        if not is3d:
            angs.append(None)
            continue
        y, pt, rl = pose.angles(fr) if pose else (0.0, 0.0, 0.0)
        pt = 0.0 if args.no_pose_pitch else pt
        rl = 0.0 if args.no_pose_roll else rl
        angs.append((y, pt + args.pitch, rl + args.roll))
    def camd(fr):
        if args.cam_dist is not None:
            return args.cam_dist
        c = pose.cam_dist(fr) if pose is not None else None
        return c if c is not None else 3.0
    cams = [camd(fr) for _, _, fr in items]
    timing = Timing(fps=args.fps)
    try:
        hits = Timing.parse_hits(args.hits, args.hit_len)
        trans = (Timing.parse_trans(args.appear, "in", args.trans_len)
                 + Timing.parse_trans(args.vanish, "out", args.trans_len))
        if args.curve:
            timing = Timing.from_file(args.curve, args.fps, args.hit_len, args.trans_len)
        timing = Timing(timing.keys, timing.hits + hits, args.fps, timing.trans + trans)
    except OSError as ex:
        raise PlanError(f"--curve: {ex}") from None
    except ValueError as ex:
        raise PlanError(str(ex)) from None
    fos = [timing.overrides(fr, args.intensity, gains) for _, _, fr in items]
    plates = None
    if args.plate:                                       # background plate: one still or a sequence
        pp = Path(args.plate)
        plist = ([pp] if pp.is_file() else
                 sorted((q for q in (pp.iterdir() if pp.is_dir() else map(Path, _glob.glob(args.plate)))
                         if q.suffix.lower() in IMG_EXT), key=_natural_key))
        if not plist:
            raise PlanError(f"--plate: no images found in {args.plate}")
        plates = [plist[min(i, len(plist) - 1)] for i in range(len(items))]
        hit = {_ident(q) for q in plist} & {_ident(d_) for _, d_, _ in items}
        if hit:
            raise PlanError(f"refusing to overwrite the background plate {sorted(hit)[0]}: choose an "
                            f"output folder that is not the plate folder")
        if 1 < len(plist) != len(items):
            notes.append(f"note: {len(plist)} plate frame(s) for {len(items)} frame(s) - matched in "
                         f"order, the last plate held")
        kw.update(spill=args.spill)
    motion = None
    if args.motion > 0 and kind != "file":               # glitch follows how fast the subject moves
        motion = motion_curve([it[0] for it in items], io)
        fos = [apply_motion(fo, args.intensity, float(v), args.motion) for fo, v in zip(fos, motion)]
    audio = None
    if args.audio and args.audio_strength > 0:           # sound-reactive: passage lined up with frame 0
        f0 = items[0][2]
        times = [args.audio_start + (fr - f0) / args.fps for _, _, fr in items]
        try:
            loud, hit, beats = audio_curves(args.audio, times, args.audio_start, args.audio_end,
                                            args.audio_sens, args.fps)
        except (OSError, ValueError) as ex:
            raise PlanError(f"--audio: {ex}") from None
        fos = [apply_audio(fo, args.intensity, gains, float(lv), float(hv), args.audio_strength)
               for fo, lv, hv in zip(fos, loud, hit)]
        audio = dict(loud=loud, hit=hit, beats=beats, times=np.asarray(times))
    # body-part targeting: landmarks per frame (from estimate_pose.py) into each job
    try:
        only, protect = parse_parts(args.only), parse_parts(args.protect)
        targets = {}
        for t_ in args.target:
            fx, sep, parts = t_.partition("=")
            if not sep or fx.strip() not in EFFECT_ORDER:
                raise ValueError(f"--target {t_!r}: expected EFFECT=PARTS, e.g. mosh=head")
            targets[fx.strip()] = parse_parts(parts)
    except ValueError as ex:
        raise PlanError(str(ex)) from None
    if only or protect or targets:
        lpath = args.landmarks
        if not lpath and args.pose:
            pp = Path(args.pose)
            lpath = str(pp.with_name(pp.stem + "_landmarks.npz"))
        if not lpath or not Path(lpath).is_file():
            raise PlanError("--only / --protect / --target need body landmarks: run estimate_pose.py "
                            "(it writes <pose>_landmarks.npz) and pass --pose or --landmarks"
                            + (f" (looked for {lpath})" if lpath else ""))
        try:
            lmk = Landmarks(lpath)
        except (OSError, ValueError, KeyError) as ex:
            raise PlanError(f"--landmarks: {ex}") from None
        kw.update(only=only, protect=protect, targets=targets, feather=args.part_feather)
        fos = [dict(fo, landmarks=lmk.at(fr)) for fo, (_, _, fr) in zip(fos, items)]
    # time effects: each frame gets the source files before it (nearest first); a still has none
    prevs = [tuple(items[j][0] for j in range(i - 1, max(-1, i - 1 - MAX_BACK), -1))
             if kind != "file" else () for i in range(len(items))]
    gappy = kind != "file" and any(items[i][2] - items[i - 1][2] != 1 for i in range(1, len(items)))
    if gappy:                                            # time effects need the real frame distances
        fos = [dict(fo, prev_frames=tuple(items[j][2] for j in range(i - 1, max(-1, i - 1 - MAX_BACK), -1)))
               for i, fo in enumerate(fos)]
    if plates is not None:
        fos = [dict(fo, plate=str(plates[i]),
                    prev_plates=tuple(str(plates[j]) for j in range(i - 1, max(-1, i - 1 - MAX_BACK), -1))
                    if kind != "file" else ())
               for i, fo in enumerate(fos)]
    if getattr(args, "glitch_pass", None) is not None:
        if args.trim:
            raise PlanError("--glitch-pass can't be combined with --trim (the matte must match the "
                            "frame's size)")
        pdir = Path(args.glitch_pass) if args.glitch_pass else None
        pdsts = [(pdir or Path(d_).parent / "glitch_pass") / Path(d_).name for _, d_, _ in items]
        mains = {_ident(d_) for _, d_, _ in items}
        protected = sources | ({_ident(q) for q in plist} if plates is not None else set())
        for q in pdsts:
            if _ident(q) in mains:
                raise PlanError(f"--glitch-pass would overwrite the rendered frame {q}: give it a "
                                f"folder other than the output folder")
            if _ident(q) in protected:
                raise PlanError(f"refusing to overwrite source material {q} with the glitch pass")
        fos = [dict(fo, pass_dst=str(q)) for fo, q in zip(fos, pdsts)]
    work = [(s, d, fr, b, kw, g, cm, io, fo, pv)
            for (s, d, fr), b, g, cm, fo, pv in zip(items, boxes, angs, cams, fos, prevs)]
    if getattr(args, "range", None):
        try:
            lo_s, _, hi_s = args.range.partition("-")
            lo_r, hi_r = int(lo_s), int(hi_s or lo_s)
        except ValueError:
            raise PlanError(f"--range: expected FIRST-LAST frame numbers, got {args.range!r}") from None
        keep = [i for i, w_ in enumerate(work) if lo_r <= w_[2] <= hi_r]
        if not keep:
            raise PlanError(f"--range {args.range}: no frames in that range")
        work = [work[i] for i in keep]
        boxes, angs, cams = [boxes[i] for i in keep], [angs[i] for i in keep], [cams[i] for i in keep]
        scales = None if scales is None else [scales[i] for i in keep]
        motion = None if motion is None else motion[keep]
        if audio is not None:
            audio = dict(audio, loud=audio["loud"][keep], hit=audio["hit"][keep], times=audio["times"][keep])
    if ext == ".exr" or in_ext == ".exr":
        try:
            _openexr()
        except RuntimeError as ex:
            raise PlanError(str(ex)) from None

    return dict(work=work, boxes=boxes, seed=seed, follow=follow, is3d=is3d, angs=angs,
                cams=cams, scales=scales, jobs=jobs, pose=pose, size=(W, H), timing=timing,
                motion=motion, audio=audio)


def main(argv=None) -> int:
    ap = build_parser()
    args = ap.parse_args(argv)
    notes = []
    try:
        P = plan_render(args, notes)
    except PlanError as ex:
        ap.error(str(ex))
    for n_ in notes:
        print(n_, file=sys.stderr)
    work, boxes, seed, follow, is3d = P["work"], P["boxes"], P["seed"], P["follow"], P["is3d"]
    angs, cams, scales, jobs = P["angs"], P["cams"], P["scales"], P["jobs"]

    # ---- pass 2: render (every frame is independent, so this parallelises trivially)
    n = len(work)
    tty = sys.stderr.isatty()

    def tick(i):
        if tty:
            print(f"\r{i}/{n}", end="", file=sys.stderr, flush=True)

    try:
        native_on = native() is not None
    except RuntimeError as ex:                  # GLITCHER_ENGINE=native without the library
        print(f"error: {ex}", file=sys.stderr)
        return 1
    if not native_on:
        _fast()   # load (first time on a machine: compile) once here, so the workers just load
    print(f"render code: {renderer() if native_on else FAST_STATUS}", file=sys.stderr)
    if jobs > 1:
        px = canvas_px(work)
        fit = safe_workers(jobs, px)
        if fit < jobs:
            print(f"note: {fit} worker(s) instead of {jobs}: each needs about "
                  f"{worker_bytes(px) / 1e9:.1f} GB", file=sys.stderr)
        try:
            render_pool(work, fit, progress=tick)
        except RenderCrash as ex:
            print(f"error: {ex}", file=sys.stderr)
            return 1
    else:
        for i, w_ in enumerate(work, 1):
            _render(w_)
            tick(i)
    if tty:
        print(file=sys.stderr)
    first, last = work[0][1], work[-1][1]
    print(f"{n} frame(s) -> {first}" + (f" ... {last}" if n > 1 else ""))
    ub = boxes[0]
    print(f"seed {seed}, layout box {','.join(str(int(round(v))) for v in ub)}"
          + ("" if follow <= 0 else " (first frame; following subject)"))
    if is3d:
        A = np.array(angs)
        print(f"3D mode: yaw {A[:, 0].min():.1f}..{A[:, 0].max():.1f}, pitch {A[:, 1].min():.1f}.."
              f"{A[:, 1].max():.1f}, roll {A[:, 2].min():.1f}..{A[:, 2].max():.1f} deg; lock {args.lock}, "
              f"cam-dist {min(cams):.2f}..{max(cams):.2f}, depth {args.depth}, tilt {args.tilt}"
              + (f", scale {min(scales):.2f}..{max(scales):.2f}" if scales else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
