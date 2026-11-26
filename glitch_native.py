"""Python access to the native engine (engine/, glitch_core) through ctypes.

    import glitch_native
    eng = glitch_native.Engine()                       # finds the built library
    out = eng.render(frames, i, seed=3, intensity=0.8, gains={"echo": 2})

`frames` is a list of RGBA images (PIL images or uint8 / float arrays, straight alpha, like
glitch.py takes); frame i is rendered with the frames before it as history. The engine works
in premultiplied float 0..1; this module converts both ways. The engine has all 13 effects,
3D mode (mode3d=1 with yaw / pitch / roll and / or pose_path), body parts (only / protect /
targets with landmarks_path, estimate_pose.py's _landmarks.csv), plates (plate=image), the
app's whole-shot layout box (layout="shot"), hits (hits="48, 112:0.6") and appear / vanish
(appear="0", vanish="120:1.5").

    out = eng.render(frames, i, mode3d=1, pose_path="shot_pose.csv", protect="head",
                     landmarks_path="shot_pose_landmarks.csv", plate=bg, spill=0.1)
"""
from __future__ import annotations

import ctypes as C
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
EFFECTS = ["jpeg", "blocks", "slices", "wave", "sort", "smear", "split", "tint", "scanlines",
           "noise", "echo", "stutter", "mosh"]
MAX_BACK = 24
ABI = 8
PARTS = ("face", "head", "torso", "arms", "hands", "legs", "feet")


class GcParams(C.Structure):
    _fields_ = [("abi", C.c_uint32), ("seed", C.c_uint32), ("intensity", C.c_double), ("fps", C.c_double),
                ("tempo", C.c_double), ("density", C.c_double), ("bursts", C.c_double),
                ("boost", C.c_double), ("gains", C.c_double * 13), ("tint", C.c_double),
                ("quality", C.c_int32), ("bleed", C.c_int32),
                # 3D
                ("mode3d", C.c_int32), ("pose_flags", C.c_int32), ("yaw", C.c_double), ("pitch", C.c_double),
                ("roll", C.c_double), ("lock", C.c_double), ("cam_dist", C.c_double), ("depth", C.c_double),
                ("tilt", C.c_double), ("pose_smooth", C.c_double), ("track_offset", C.c_double),
                ("pose_path", C.c_char_p), ("landmarks_path", C.c_char_p),
                # body parts, plate
                ("only", C.c_uint32), ("protect", C.c_uint32), ("targets", C.c_uint32 * 13), ("overlay", C.c_uint32),
                ("feather", C.c_double), ("spill", C.c_double),
                # layout box and hits
                ("layout_mode", C.c_int32), ("_pad2", C.c_int32), ("layout", C.c_double * 4),
                ("box_grow", C.c_double), ("hits", C.c_char_p), ("hit_len", C.c_double),
                # appear / vanish
                ("appear", C.c_char_p), ("vanish", C.c_char_p), ("trans_len", C.c_double),
                # from the shot scan; a frame planned by the host
                ("follow", C.c_double), ("motion", C.c_double), ("reveal_set", C.c_int32),
                ("reveal_idx", C.c_int32), ("reveal_set_q", C.c_double), ("landmarks", C.POINTER(C.c_double))]


class GcImage(C.Structure):
    _fields_ = [("ptr", C.c_void_p), ("width", C.c_int32), ("height", C.c_int32),
                ("stride", C.c_int64), ("format", C.c_int32), ("reserved", C.c_int32)]


class GcStats(C.Structure):
    _fields_ = [("ms", C.c_double), ("echo", C.c_int32), ("stutter", C.c_int32), ("mosh", C.c_int32),
                ("has_box", C.c_int32), ("bbox", C.c_int32 * 4), ("frame_fx", C.c_int32 * 10),
                ("flags", C.c_int32), ("warn", C.c_int32), ("has_view", C.c_int32), ("_pad", C.c_int32),
                ("view", C.c_double * 4), ("hit", C.c_double), ("reveal", C.c_int32), ("_pad3", C.c_int32),
                ("reveal_q", C.c_double), ("motion", C.c_double), ("layout_from", C.c_int32),
                ("_pad4", C.c_int32)]


LIB_NAME = {"win32": "glitch_core.dll", "darwin": "libglitch_core.dylib"}.get(sys.platform, "libglitch_core.so")


def library_path() -> Path | None:
    """GLITCHER_CORE, else the release build in engine/target, else next to this file (or, in
    the packaged app, in its bundle / next to the executable)."""
    cands = [os.environ.get("GLITCHER_CORE"), HERE / "engine" / "target" / "release" / LIB_NAME, HERE / LIB_NAME]
    if getattr(sys, "frozen", False):
        cands += [Path(getattr(sys, "_MEIPASS", HERE)) / LIB_NAME, Path(sys.executable).parent / LIB_NAME]
    for p in cands:
        if p and Path(p).is_file():
            return Path(p)
    return None


def load(path=None):
    path = Path(path) if path else library_path()
    if path is None:
        raise FileNotFoundError("glitch_core library not found: build it (cd engine; cargo build --release) "
                                "or set GLITCHER_CORE")
    lib = C.CDLL(str(path))
    lib.gc_version.restype = C.c_char_p
    lib.gc_engine_new.restype = C.c_void_p
    lib.gc_engine_free.argtypes = [C.c_void_p]
    lib.gc_needed.argtypes = [C.c_void_p, C.POINTER(GcParams), C.c_int64, C.c_int32, C.c_int32, C.c_int32,
                              C.POINTER(C.c_int32)]
    lib.gc_render.argtypes = [C.c_void_p, C.POINTER(GcParams), C.c_int64, C.POINTER(GcImage), C.POINTER(GcImage),
                              C.POINTER(GcImage), C.POINTER(GcImage), C.POINTER(GcImage), C.POINTER(GcStats)]
    lib.gc_sizes.argtypes = [C.POINTER(C.c_int32)]
    sz = (C.c_int32 * 3)()
    lib.gc_sizes(sz)
    mine = (C.sizeof(GcParams), C.sizeof(GcImage), C.sizeof(GcStats))
    if tuple(sz) != mine:
        raise RuntimeError(f"glitch_core {path.name}: struct sizes {tuple(sz)} differ from this module's {mine} "
                           "(library and glitch_native.py from different versions)")
    lib.gc_last_error.argtypes = [C.c_void_p]
    lib.gc_last_error.restype = C.c_char_p
    lib.gc_premultiply.argtypes = [C.c_void_p, C.c_int32, C.c_int32, C.c_int64, C.c_int32, C.POINTER(GcImage),
                                   C.c_int32, C.c_int32]
    lib.gc_unpremultiply.argtypes = [C.POINTER(GcImage), C.c_int32, C.c_int32, C.c_int32, C.c_int32, C.c_void_p,
                                     C.c_int64, C.c_int32]
    lib.gc_parse_parts.argtypes = [C.c_void_p, C.c_char_p]
    lib.gc_parse_targets.argtypes = [C.c_void_p, C.c_char_p, C.POINTER(C.c_uint32)]
    return lib


def part_bits(parts) -> int:
    """'head,hands' / ('head', 'hands') -> the engine's part bits (groups upper/lower/body too)."""
    items = parts.replace("+", ",").split(",") if isinstance(parts, str) else list(parts or ())
    groups = {"upper": ("head", "torso", "arms", "hands"), "lower": ("legs", "feet"), "body": PARTS}
    bits = 0
    for it in (x.strip().lower() for x in items):
        if not it or it == "all":
            continue
        for g in groups.get(it, (it,)):
            if g not in PARTS:
                raise ValueError(f"unknown body part {it!r}")
            bits |= 1 << PARTS.index(g)
    return bits


def to_premul(img) -> np.ndarray:
    """RGBA (PIL / uint8 0..255 / float 0..255, straight alpha) -> float32 premultiplied 0..1."""
    a = np.asarray(img.convert("RGBA") if hasattr(img, "convert") else img, dtype=np.float32) / 255.0
    out = np.empty_like(a)
    out[..., :3] = a[..., :3] * a[..., 3:4]
    out[..., 3] = a[..., 3]
    return np.ascontiguousarray(out)


def from_premul(p: np.ndarray, as_uint8: bool = True) -> np.ndarray:
    """float32 premultiplied 0..1 -> straight RGBA 0..255 (uint8, or float32 for HDR)."""
    a = p[..., 3:4]
    rgb = np.where(a > 0, p[..., :3] / np.maximum(a, 1e-8), 0.0)
    out = np.concatenate([rgb, a], axis=-1) * 255.0
    if as_uint8:
        return np.clip(out + 0.5, 0, 255).astype(np.uint8)
    return out.astype(np.float32)


def _image(arr: np.ndarray) -> GcImage:
    assert arr.dtype == np.float32 and arr.flags.c_contiguous and arr.ndim == 3 and arr.shape[2] == 4
    h, w = arr.shape[:2]
    return GcImage(arr.ctypes.data, w, h, 4 * w, 0, 0)


class Engine:
    def __init__(self, path=None):
        self.lib = load(path)
        self.version = self.lib.gc_version().decode()
        self._e = self.lib.gc_engine_new()

    def __del__(self):
        if getattr(self, "_e", None):
            self.lib.gc_engine_free(self._e)
            self._e = None

    @staticmethod
    def params(seed=0, intensity=0.5, fps=25.0, tempo=1.0, density=1.0, bursts=0.5, boost=0.0, gains=None,
               tint=0.0, quality=0, bleed=-1, mode3d=0, yaw=0.0, pitch=0.0, roll=0.0, lock=1.0, cam_dist=0.0,
               depth=0.35, tilt=0.25, pose_path=None, pose_smooth=0.0, pose_unwrap=False, no_pose_pitch=False,
               no_pose_roll=False, track_offset=0.0, landmarks_path=None, only=(), protect=(), targets=None,
               feather=1.0, spill=0.05, overlay=False, layout=None, box_grow=0.0, hits="", hit_len=0.5,
               appear="", vanish="", trans_len=1.0, reveal=None, landmarks=None):
        g = [1.0] * 13
        for k, v in (gains or {}).items():
            g[EFFECTS.index(k)] = float(v)
        t = [0] * 13
        for k, v in (targets or {}).items():
            t[EFFECTS.index(k)] = part_bits(v)
        enc = lambda s: None if not s else str(s).encode("utf-8")       # noqa: E731
        flags = (1 if no_pose_pitch else 0) | (2 if no_pose_roll else 0) | (4 if pose_unwrap else 0)
        p = GcParams(ABI, int(seed) & 0xFFFFFFFF, intensity, fps, tempo, density, bursts, boost,
                     (C.c_double * 13)(*g), tint, int(quality or 0), -1 if bleed is None else int(bleed),
                     int(bool(mode3d)), flags, yaw, pitch, roll, lock, cam_dist or 0.0, depth, tilt, pose_smooth,
                     track_offset, enc(pose_path), enc(landmarks_path), part_bits(only), part_bits(protect),
                     (C.c_uint32 * 13)(*t), int(bool(overlay)), feather, spill,
                     0 if layout is None else 1, 0, (C.c_double * 4)(*(layout or (0, 0, 0, 0))), box_grow,
                     enc(hits), hit_len, enc(appear), enc(vanish), trans_len)
        if reveal is not None:                            # glitch.py's reveal tuple, as planned
            kind = str(reveal[0])
            p.reveal_set = {"hidden": 4, "in": 2, "out": 3}.get(kind, 1)
            if kind in ("in", "out"):
                p.reveal_set_q = float(reveal[1])
                p.reveal_idx = int(reveal[2]) if len(reveal) > 2 else 0
        if landmarks is not None:
            lm = np.ascontiguousarray(np.asarray(landmarks, np.float64).reshape(33, 3))
            p._lm = lm                                    # kept alive with the params
            p.landmarks = lm.ctypes.data_as(C.POINTER(C.c_double))
        return p

    def needed(self, p: GcParams, frame: int, w: int, h: int, max_back: int) -> list[int]:
        buf = (C.c_int32 * MAX_BACK)()
        n = self.lib.gc_needed(self._e, C.byref(p), frame, w, h, max_back, buf)
        if n < 0:
            raise RuntimeError(f"gc_needed failed ({n})")
        return list(buf[:n])

    def last_error(self) -> str:
        return (self.lib.gc_last_error(self._e) or b"").decode("utf-8", "replace")

    def render_premul(self, cur: np.ndarray, back: dict, frame: int, p: GcParams, plate=None, matte=False):
        """cur: premultiplied float32 (h, w, 4); back: {k: premultiplied frame k back}; plate:
        premultiplied float32 background or None. matte=True: also the glitch matte (h, w)."""
        hist = (GcImage * MAX_BACK)()
        for k, arr in back.items():
            if 1 <= k <= MAX_BACK and arr is not None:
                hist[k - 1] = _image(arr)
        out = np.empty_like(cur)
        st = GcStats()
        ci, oi = _image(cur), _image(out)
        pi = C.byref(_image(plate)) if plate is not None else None
        m = np.zeros_like(cur) if matte else None
        mi = C.byref(_image(m)) if matte else None
        r = self.lib.gc_render(self._e, C.byref(p), frame, C.byref(ci), hist, pi, C.byref(oi), mi, C.byref(st))
        if r != 0:
            raise RuntimeError(f"gc_render failed ({r})")
        if st.warn:
            self.warning = self.last_error()
        if matte:
            return out, st, m[..., 0].copy()
        return out, st

    @staticmethod
    def shot_box(frames):
        """The union of the subject's boxes (alpha > 8) over frames - the layout box glitch.py's
        app uses by default; None if no frame has a subject."""
        box = None
        for f in frames:
            a = np.asarray(f)[..., 3]
            ys, xs = np.nonzero(a > 8)
            if ys.size == 0:
                ys, xs = np.nonzero(a > 0)
            if ys.size:
                b = (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)
                box = b if box is None else (min(box[0], b[0]), min(box[1], b[1]), max(box[2], b[2]), max(box[3], b[3]))
        return None if box is None else tuple(float(v) for v in box)

    def render(self, frames, i: int, plate=None, **kw):
        """Frame i of a list of straight-alpha RGBA frames -> straight uint8 RGBA array. plate =
        background image (straight RGBA) or None. layout="shot": the whole list's box."""
        if kw.get("layout") == "shot":
            if getattr(self, "_shot_of", None) is not frames:
                self._shot_of, self._shot = frames, self.shot_box(frames)
            kw["layout"] = self._shot
        p = self.params(**kw)
        h, w = np.asarray(frames[i]).shape[:2]
        need = self.needed(p, i, w, h, min(i, MAX_BACK))
        cur = to_premul(frames[i])
        pl = to_premul(plate) if plate is not None else None
        out, _ = self.render_premul(cur, {k: to_premul(frames[i - k]) for k in need}, i, p, pl)
        return from_premul(out)


# --------------------------------------------------------------------------- glitch.py's API
_ENGINE: Engine | None = None
_FAILED: str | None = None


def engine() -> Engine:
    """This process's engine (one per process: its motion cache serves consecutive frames)."""
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = Engine()
    return _ENGINE


def available() -> bool:
    """Whether the native engine can be used here (and why not: unavailable_reason())."""
    global _FAILED
    if _ENGINE is not None:
        return True
    if _FAILED is not None:
        return False
    try:
        engine()
        return True
    except (OSError, RuntimeError, FileNotFoundError) as ex:
        _FAILED = f"{type(ex).__name__}: {ex}"
        return False


def unavailable_reason() -> str:
    return _FAILED or ""


def _straight(img, is_float: bool) -> np.ndarray:
    """glitch.py's frame (PIL image, or float32 0..255 array) as a contiguous straight RGBA array."""
    if is_float:
        return np.ascontiguousarray(np.asarray(img, np.float32))
    return np.ascontiguousarray(np.asarray(img if img.mode == "RGBA" else img.convert("RGBA")))


def _canvas(lib, img, size, at) -> np.ndarray:
    """img premultiplied into a zero float32 canvas of `size` (w, h) at `at` (x, y): the padding
    and the conversion in one pass, on the engine's threads."""
    is_float = isinstance(img, np.ndarray)
    a = _straight(img, is_float)
    h, w = a.shape[:2]
    full = (w, h) == tuple(size)
    out = (np.empty if full else np.zeros)((size[1], size[0], 4), np.float32)   # padding stays zero
    r = lib.gc_premultiply(a.ctypes.data, w, h, 4 * w, 0 if is_float else 3, C.byref(_image(out)), at[0], at[1])
    if r != 0:
        raise ValueError(f"frame of {w}x{h} doesn't fit the {size[0]}x{size[1]} canvas at {at}")
    return out


def _from_canvas(lib, p: np.ndarray, box, as_float: bool) -> np.ndarray:
    """The (x0, y0, x1, y1) part of a premultiplied canvas as straight RGBA (uint8, or float32 0..255)."""
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    out = np.empty((h, w, 4), np.float32 if as_float else np.uint8)
    r = lib.gc_unpremultiply(C.byref(_image(p)), x0, y0, w, h, out.ctypes.data, 4 * w, 0 if as_float else 3)
    if r != 0:
        raise ValueError("bad output box")
    return out


def _history_frame(history, k):
    if history is None:
        return None
    if callable(history):
        return history(k)
    return history[k - 1] if 1 <= k <= len(history) else None


def glitch(img, t: float = 0.0, *, seed=None, intensity: float = 0.5, fps: float = 25.0, tempo: float = 1.0,
           density: float = 1.0, bursts: float = 0.5, pad: float = 0.0, tint: float = 0.0, skip=(), bleed=None,
           quality=None, bbox=None, shift=(0.0, 0.0), frame=None, trim: bool = False, yaw=None, lock: float = 1.0,
           cam_dist: float = 3.0, depth: float = 0.35, pitch=None, roll=None, tilt: float = 0.25, gains=None,
           boost: float = 0.0, history=None, history_key=None, reveal=None, landmarks=None, only=(), protect=(),
           targets=None, feather: float = 1.0, plate=None, spill: float = 0.05, matte: bool = False):
    """glitch.glitch() rendered by the native engine: same arguments, same frame types in and out
    (PIL image -> PIL image, float32 0..255 straight array -> float32 array), same layout box,
    3D angles, reveals, body parts and plate. The look is the engine's (its own random events,
    the time effects on the subject before a plate). matte=True also returns the glitch matte
    (float32 0..1, the output's size): (image, matte)."""
    from PIL import Image
    is_float = isinstance(img, np.ndarray)
    eng = engine()
    lib = eng.lib
    if seed is None:
        seed = int(np.random.SeedSequence().entropy % 2**31)
    if frame is None:
        frame = int(round(t * fps))
    g = {fx: float(v) for fx, v in (gains or {}).items()}
    for fx in skip:
        g[fx] = 0.0
    W, H = (img.shape[1], img.shape[0]) if is_float else img.size
    pl = None
    if plate is not None:
        pw, ph = (plate.shape[1], plate.shape[0]) if isinstance(plate, np.ndarray) else plate.size
        if (pw, ph) != (W, H):
            raise ValueError(f"plate is {pw}x{ph}, frame is {W}x{H}: sizes must match")
        pl = _canvas(lib, plate, (W, H), (0, 0))
    py, px = (0, 0) if pl is not None else (int(H * pad), int(W * pad))
    Wp, Hp = W + 2 * px, H + 2 * py
    cur = _canvas(lib, img, (Wp, Hp), (px, py))
    lay = None
    if bbox is not None:
        lay = (bbox[0] + px + shift[0], bbox[1] + py + shift[1], bbox[2] + px + shift[0], bbox[3] + py + shift[1])
    lm = None
    if landmarks is not None:
        lm = np.array(landmarks, np.float64).reshape(33, 3)
        lm[:, 0] += px
        lm[:, 1] += py
    is3d = yaw is not None or pitch is not None or roll is not None
    p = eng.params(seed=seed, intensity=float(np.clip(intensity, 0, 1)), fps=fps, tempo=tempo, density=density,
                   bursts=bursts, boost=boost, gains=g, tint=tint, quality=int(quality or 0),
                   bleed=-1 if bleed is None else int(bleed), mode3d=int(is3d), yaw=float(yaw or 0.0),
                   pitch=float(pitch or 0.0), roll=float(roll or 0.0), lock=lock, cam_dist=cam_dist or 3.0,
                   depth=depth, tilt=tilt, only=only or (), protect=protect or (),
                   targets={fx: v for fx, v in (targets or {}).items()}, feather=feather, spill=spill,
                   layout=lay, reveal=reveal if reveal is not None else ("visible",), landmarks=lm)
    if history is not None and getattr(history, "plates", None):
        # glitch.py's comp mode hands earlier frames already on their plates; the engine runs
        # the time effects on the subject alone and composites afterwards, so it wants them bare.
        import copy
        history = copy.copy(history)
        history.plates = None
    back = {}
    for k in eng.needed(p, int(frame), Wp, Hp, MAX_BACK):
        h = _history_frame(history, k)
        if h is None:
            continue
        hw, hh = (h.shape[1], h.shape[0]) if isinstance(h, np.ndarray) else h.size
        if (hw, hh) == (W, H):
            back[k] = _canvas(lib, h, (Wp, Hp), (px, py))
    res = eng.render_premul(cur, back, int(frame), p, pl, matte=matte)
    out, m = res[0], (res[2] if matte else None)
    box = (0, 0, Wp, Hp)
    if trim:
        ys, xs = np.nonzero(out[..., 3] * 255.0 + 0.5 >= 1.0)   # alpha > 0 once 8-bit
        if ys.size:
            box = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
            m = None if m is None else m[box[1]:box[3], box[0]:box[2]]
    a = _from_canvas(lib, out, box, is_float)
    if not is_float:
        a = Image.fromarray(a, "RGBA")
    return (a, m) if matte else a
