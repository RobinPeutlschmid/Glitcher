#!/usr/bin/env python3
"""
estimate_pose.py (Glitcher 1.2) - estimate a person's orientation relative to the camera per frame (yaw, pitch,
roll) and write the pose file that glitch.py --pose reads.

Dependencies: numpy, Pillow, mediapipe   (pip install numpy pillow mediapipe)
The MediaPipe pose model (~30 MB) is downloaded on first run, or pass --model.

Usage
-----
    python estimate_pose.py frames/ -o pose.csv --plot pose.png
    python estimate_pose.py frames/ -o pose.csv --hint 0:0 --hint 120:180   # pin known yaws
    python estimate_pose.py frames/ -o pose.csv --pitch-hint 0:30 --pitch-hint 96:20  # fix pitch
    python glitch.py frames/ -o out/ --pose pose.csv

Output: CSV "frame,yaw,pitch,roll,scale[,cam_dist],confidence" (angles in degrees), same
conventions as glitch.py:
    yaw    0 = facing the camera, + = the front turns toward screen right; continuous (a full
           turn reads 0 -> 360)
    pitch  + = camera above the subject (high angle), - = below (low angle)
    roll   + = subject appears rotated clockwise on screen (dutch angle or leaning)
    scale  apparent size relative to the clip's median frame; with --focal-mm also
    cam_dist  camera distance in subject heights (drives perspective strength in glitch.py)
All three describe the *torso* relative to the camera, so a person leaning forward and a camera
looking down read the same - which is what the glitch volume needs. Frame numbering matches
glitch.py (position in the sequence + --offset, or the filename number with --use-numbers).

How it works
------------
1. Landmarks. MediaPipe Pose (VIDEO mode, tracks between frames) on each frame, cropped to the
   union of the alpha boxes and composited on gray.
2. Roll from the 2-D torso axis (hip centre -> shoulder centre) in the image: rotations about the
   view axis are exactly what that axis shows. Image measurements below are un-rolled first.
3. Yaw, fused per frame from: the shoulder and hip lines in 3-D (world landmarks), each made
   into a full rotation with the torso axis and decomposed exactly like glitch.py composes it
   (R = Rroll.Rpitch.Ryaw), plus a 2-D width cue (apparent shoulder width / torso length ->
   |cos yaw|). Then solved for the whole clip: Viterbi over (5-degree bin, mirrored?) states.
   Front/back confusion in a pose model is a mirror (reading = 180 - true), smooth through
   profile, so the mirrored state is sticky and penalised and --hint FRAME:YAW settles it.
4. Pitch from two cues: the torso axis tilting toward/away from the camera in 3-D (depth), and
   the slant of the shoulder and hip lines in the image, which for a turned body equals
   -sin(pitch) * tan(yaw) - strong at oblique yaws, absent when facing the camera or in profile.
   Weighted by where each cue is informative, then smoothed.
5. Pitch and roll are smoothed more than yaw (--smooth-tilt): they change slowly in real shots
   and their per-frame measurements are noisier.
6. Scale: a weak-perspective fit of the image joints to MediaPipe's metric 3-D skeleton gives
   pixels per metre, which does not change when the person turns, bends or lifts an arm (unlike
   the bounding box). A dolly and a zoom look the same in scale; only with the lens known
   (--focal-mm, --sensor-mm) can it be turned into camera distance = focal / stature in px.

Limits (measured on rendered characters with known angles): torso only, not the head.
  yaw    ~6-11 deg median error with a level camera; worse on steep high angles, where the
         pose model loses frames and front/back gets ambiguous - check the plot, add --hint.
  roll   reliable: ~2-5 deg median.
  pitch  the weakest: MediaPipe's depth is biased toward "upright", so low angles read well
         (~3 deg median at -25) but high angles read low (+30 came out ~+13). If you know the
         camera height, override it in glitch.py: --no-pose-pitch --pitch 30.
A subject with no face cues may come out mirrored in yaw for a stretch - add a --hint.

Hints
  --hint F:YAW         pins the yaw solver at that frame (and settles front/back around it);
                       between hints the measurements still drive the curve.
  --pitch-hint F:DEG   correct pitch / roll: the curve is shifted so it passes through every hint,
  --roll-hint F:DEG    the shift interpolated linearly between hints and held beyond them. Pitch is
                       usually biased rather than noisy, so one or two hints fix a whole shot.
glitch_gui.py's Pose tab edits these on its timeline and re-solves from the landmark cache.
"""
from __future__ import annotations

import argparse
import csv
import json
import glob as _glob
import math
import os
import re
import sys
import urllib.request
from pathlib import Path

os.environ.setdefault("GLOG_minloglevel", "2")          # quiet MediaPipe / TFLite logging
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
from PIL import Image

IMG_EXT = {".png", ".webp", ".tif", ".tiff", ".exr"}   # same as glitch.py: frame numbering matches


def open_rgba(path, io=None) -> Image.Image:
    """8-bit RGBA image of a frame. EXR goes through glitch.py's reader (needs glitch.py next to
    this script and `pip install OpenEXR`) and is shown display-encoded, as MediaPipe expects."""
    if Path(path).suffix.lower() == ".exr":
        try:
            sys.path.insert(0, str(Path(__file__).resolve().parent))
            import glitch
        except ImportError:
            raise RuntimeError("EXR input needs glitch.py in the same folder as estimate_pose.py") from None
        img, _raw = glitch.load_frame(path, io)
        return glitch.to_display8(img)
    with Image.open(path) as im:
        return im.convert("RGBA")
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
             "pose_landmarker_{v}/float16/latest/pose_landmarker_{v}.task")
LSH, RSH, LHIP, RHIP = 11, 12, 23, 24
NBINS = 72                                                # 5-degree states
JOINTS = {"nose": 0, "ls": 11, "rs": 12, "le": 13, "re": 14, "lh": 23, "rh": 24,
          "lk": 25, "rk": 26, "la": 27, "ra": 28}          # MediaPipe pose landmark indices
FEAT = ["ok"] + [f"{q}_{j}" for j in JOINTS for q in ("wx", "wy", "wz", "ix", "iy", "v")]


# --------------------------------------------------------------------------- frames
def _natural_key(path) -> list:
    return [int(s) if s.isdigit() else s.lower() for s in re.split(r"(\d+)", Path(path).name)]


def gather(inp: str) -> list[Path]:
    p = Path(inp)
    if p.is_file():
        return [p]
    if p.is_dir():
        return sorted([f for f in p.iterdir() if f.suffix.lower() in IMG_EXT], key=_natural_key)
    if any(ch in inp for ch in "*?["):
        return sorted([Path(f) for f in _glob.glob(inp) if Path(f).suffix.lower() in IMG_EXT],
                      key=_natural_key)
    return [p] if p.is_file() else []


def _alpha_box(f, io=None):
    """(bbox of alpha > 8 or the whole frame if there is no alpha, frame size)."""
    if Path(f).suffix.lower() == ".exr":
        im = open_rgba(f, io)
        return im.getchannel("A").point(lambda v: 255 if v > 8 else 0).getbbox(), im.size
    with Image.open(f) as im:
        if "A" not in im.getbands():                  # no alpha: the whole frame counts
            return (0, 0, *im.size), im.size
        return im.getchannel("A").point(lambda v: 255 if v > 8 else 0).getbbox(), im.size


def union_crop(files, margin: float = 0.15, io=None):
    """Union of alpha boxes over the clip, grown by `margin` (a fixed crop keeps tracking stable)."""
    box = None
    size = None
    for f in files:
        b = _alpha_box(f, io)
        size = size or b[1]
        b = b[0]
        if b:
            box = b if box is None else (min(box[0], b[0]), min(box[1], b[1]),
                                         max(box[2], b[2]), max(box[3], b[3]))
    if box is None:
        return (0, 0, *size)
    w, h = box[2] - box[0], box[3] - box[1]
    m = int(max(w, h) * margin)
    return (max(0, box[0] - m), max(0, box[1] - m), min(size[0], box[2] + m), min(size[1], box[3] + m))


# --------------------------------------------------------------------------- landmarks
MODEL_MEMBERS = {"pose_detector.tflite", "pose_landmarks_detector.tflite"}


def model_ok(path) -> bool:
    """A .task file is a zip holding the detector and landmark models; anything else (an
    interrupted download, an HTML error page) is rejected before MediaPipe sees it."""
    import zipfile
    try:
        with zipfile.ZipFile(path) as z:
            return MODEL_MEMBERS <= set(z.namelist()) and z.testzip() is None
    except (OSError, zipfile.BadZipFile):
        return False


def _download(url: str, dst: Path, attempts: int = 30, stall: float = 20.0):
    if not url.lower().startswith("https://"):            # never file:// or plain http
        raise ValueError(f"refusing to download from a non-https address: {url}")
    """Download to dst.part with resume: if the connection stalls (no data for `stall` seconds) or
    drops, it reconnects and asks only for the missing bytes (HTTP Range), up to `attempts` times.
    The file is verified before it is moved into place, so a failed download never leaves a broken
    model behind. Retries with certifi's CA bundle on certificate errors."""
    import shutil
    import ssl
    import time
    part = dst.with_name(dst.name + ".part")       # kept between runs: the next run resumes it
    ctx = None
    total = 0
    last = None
    prev = -1
    for attempt in range(1, attempts + 1):
        done = part.stat().st_size if part.exists() else 0
        if attempt > 1:              # back off only when the last attempt got nothing
            time.sleep(0.5 if done > prev else min(2.0 * attempt, 10.0))
        prev = done
        if total and done >= total:
            break
        headers = {"User-Agent": "estimate_pose.py"}
        if done:
            headers["Range"] = f"bytes={done}-"
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=stall, context=ctx) as r:  # nosec B310 - https only, checked above
                if done and r.status != 206:           # server ignored the range: start over
                    done = 0
                    part.unlink(missing_ok=True)
                if not total:
                    cr = r.headers.get("Content-Range")  # "bytes a-b/total"
                    total = int(cr.rsplit("/", 1)[1]) if cr else int(r.headers.get("Content-Length") or 0)
                with open(part, "ab") as fh:
                    while True:
                        chunk = r.read(1 << 18)          # each read times out after `stall` s
                        if not chunk:
                            break
                        fh.write(chunk)
                        done += len(chunk)
                        if sys.stderr.isatty():
                            msg = f"{done / 1e6:5.1f} / {total / 1e6:.1f} MB" if total else f"{done / 1e6:5.1f} MB"
                            print(f"\r  {msg}" + (f"  (resume {attempt - 1})" if attempt > 1 else ""),
                                  end="", file=sys.stderr, flush=True)
            if not total or done >= total:
                break
            last = IOError(f"connection closed at {done} of {total} bytes")
        except Exception as ex:                          # noqa: BLE001 - stalls, drops, resets
            last = ex
            if isinstance(getattr(ex, "reason", ex), ssl.SSLError) and ctx is None:
                try:
                    import certifi
                    ctx = ssl.create_default_context(cafile=certifi.where())
                    continue
                except ImportError:
                    pass
            code = getattr(ex, "code", None)
            if code is not None and code not in (408, 429, 500, 502, 503, 504):
                break                                    # 404 and friends will not get better
    if sys.stderr.isatty():
        print(file=sys.stderr)
    done = part.stat().st_size if part.exists() else 0
    if not part.exists() or (total and done < total):
        part.unlink(missing_ok=True)
        raise last or IOError("download failed")
    if not model_ok(part):
        part.unlink(missing_ok=True)
        raise IOError("downloaded file is not a valid MediaPipe model bundle")
    shutil.move(str(part), str(dst))


NO_MEDIAPIPE = ("pose estimation needs MediaPipe, which this build doesn't include (built with "
                "--no-pose, or not installed: pip install mediapipe). Re-solving hints from an "
                "existing landmark cache still works.")


_MP_ERROR: ImportError | None = None


def _have_mediapipe() -> bool:
    global _MP_ERROR
    try:
        import mediapipe  # noqa: F401
        return True
    except ImportError as ex:
        _MP_ERROR = ex
        return False


def _no_mediapipe() -> str:
    """Why pose estimation is unavailable: not installed / left out of the build, or installed
    but failing to load (then the real error is what helps, e.g. a missing library)."""
    ex = _MP_ERROR
    if ex is not None and ex.name != "mediapipe":
        return f"MediaPipe is installed but failed to load: {type(ex).__name__}: {ex}"
    return NO_MEDIAPIPE


def resolve_model(path: str | None, variant: str) -> str:
    if path:
        if not Path(path).is_file():
            raise FileNotFoundError(f"model file not found: {path}")
        if not model_ok(path):
            raise ValueError(f"{path} is not a valid MediaPipe .task model (incomplete download?)")
        return path
    name = f"pose_landmarker_{variant}.task"
    cache = Path.home() / ".cache" / "estimate_pose" / name
    for cand in (Path(__file__).with_name(name), Path(sys.executable).with_name(name), cache):
        if cand.is_file():
            if model_ok(cand):
                return str(cand)
            print(f"removing broken model file {cand}", file=sys.stderr)
            cand.unlink(missing_ok=True)
    cache.parent.mkdir(parents=True, exist_ok=True)
    url = MODEL_URL.format(v=variant)
    print(f"downloading {url}", file=sys.stderr)
    try:
        _download(url, cache)
    except Exception as ex:                           # noqa: BLE001
        raise RuntimeError(
            f"download failed: {type(ex).__name__}: {ex}\n"
            f"  Try the smaller model (--variant full, ~9 MB), or fetch this one with a resuming\n"
            f"  downloader, e.g.:  curl -L -C - --retry 30 --retry-all-errors -o {name} {url}\n"
            f"  URL:\n    {url}\n"
            f"  and either save it as\n    {cache}\n"
            f"  or next to estimate_pose.py, or pass --model <path>.") from None
    return str(cache)


def self_check(model_path: str | None = None, variant: str = "heavy") -> int:
    """--check: load MediaPipe and the model and run one detection on a blank image. Used by the
    release builds to prove a packaged app can really estimate poses (a missing native library
    or model only shows up when the detector is created). 0 = works, 3 = doesn't."""
    if not _have_mediapipe():
        print("pose check FAILED: " + _no_mediapipe(), file=sys.stderr)
        return 3
    try:
        import mediapipe as mp
        from mediapipe.tasks import python as mpt
        from mediapipe.tasks.python import vision
        model = resolve_model(model_path, variant)
        det = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
            base_options=mpt.BaseOptions(model_asset_path=model),
            running_mode=vision.RunningMode.IMAGE, num_poses=1))
        try:
            img = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.full((64, 64, 3), 128, np.uint8))
            res = det.detect(img)
        finally:
            det.close()
        if res is None or not hasattr(res, "pose_landmarks"):
            raise RuntimeError("the detector returned no result")
    except Exception as ex:                               # noqa: BLE001 - report anything, exit 3
        print(f"pose check FAILED: {type(ex).__name__}: {ex}", file=sys.stderr)
        return 3
    print(f"pose check OK: MediaPipe {getattr(mp, '__version__', '?')}, model {model}")
    return 0


def extract(files, model: str, fps: float, bg: int = 128, io=None, progress: bool = False):
    """-> (F, LM). F: per-frame landmark features, shape (N, len(FEAT)); row 'ok'=0 where no person
    was found. LM: all 33 MediaPipe landmarks per frame, (N, 33, 3) = x, y in full-frame pixels and
    visibility, NaN where no person was found (glitch.py's body-part targeting reads these)."""
    import mediapipe as mp
    from mediapipe.tasks import python as mpt
    from mediapipe.tasks.python import vision

    crop = union_crop(files, io=io)
    opts = vision.PoseLandmarkerOptions(
        base_options=mpt.BaseOptions(model_asset_path=model),
        running_mode=vision.RunningMode.VIDEO, num_poses=1,
        min_pose_detection_confidence=0.3, min_pose_presence_confidence=0.3,
        min_tracking_confidence=0.3)
    det = vision.PoseLandmarker.create_from_options(opts)
    still = None                                   # single-image detector for re-tries, made lazily
    out = np.zeros((len(files), len(FEAT)), np.float64)
    lm = np.full((len(files), 33, 3), np.nan, np.float32)
    retried = 0

    def upside_down(r):
        L = r.pose_landmarks[0]
        sy = (L[LSH].y + L[RSH].y) / 2
        hy = (L[LHIP].y + L[RHIP].y) / 2
        return sy > hy                              # shoulders below hips: an inverted fit
    try:
        for i, f in enumerate(files):
            im = open_rgba(f, io).crop(crop)
            bgim = Image.new("RGBA", im.size, (bg, bg, bg, 255))
            bgim.alpha_composite(im)
            rgb = np.ascontiguousarray(np.asarray(bgim.convert("RGB")))
            mpimg = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            r = det.detect_for_video(mpimg, int(round(i * 1000.0 / fps)))
            if r.pose_landmarks and upside_down(r):
                # tracking can lock onto an upside-down fit (seen on high-angle shots) and carry it
                # forward; detect this frame from scratch instead
                if still is None:
                    still = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
                        base_options=mpt.BaseOptions(model_asset_path=model),
                        running_mode=vision.RunningMode.IMAGE, num_poses=1,
                        min_pose_detection_confidence=0.3, min_pose_presence_confidence=0.3))
                r2 = still.detect(mpimg)
                retried += 1
                r = r2 if (r2.pose_landmarks and not upside_down(r2)) else None
            if r is not None and r.pose_landmarks:
                L, W = r.pose_landmarks[0], r.pose_world_landmarks[0]
                h, w = rgb.shape[:2]
                row = [1.0]
                for k in JOINTS.values():                  # image coords in pixels (crop = no resize)
                    row += [W[k].x, W[k].y, W[k].z, L[k].x * w, L[k].y * h, L[k].visibility]
                out[i] = row
                lm[i] = [(p.x * w + crop[0], p.y * h + crop[1], p.visibility) for p in L[:33]]
            if progress:                                   # one line per frame, for the GUI
                print(f"landmarks {i + 1}/{len(files)}", file=sys.stderr, flush=True)
            elif sys.stderr.isatty():
                print(f"\rlandmarks {i + 1}/{len(files)}", end="", file=sys.stderr, flush=True)
    finally:
        det.close()
        if still is not None:
            still.close()
        if sys.stderr.isatty():
            print(file=sys.stderr)
    if retried:
        print(f"re-detected {retried} upside-down frame(s); "
              f"{int((out[:, 0] == 0).sum())} frame(s) without a usable pose", file=sys.stderr)
    return out, lm


# --------------------------------------------------------------------------- per-frame measurement
def _wrap(d):
    return (np.asarray(d) + 180.0) % 360.0 - 180.0


def decompose(x_axis, down):
    """Subject rotation from its left-pointing axis and down axis (camera coords: x right, y down,
    z toward camera), decomposed as R = Rroll.Rpitch.Ryaw -> (yaw, pitch, roll) in degrees.
    Works on arrays of shape (..., 3). Same convention as glitch.Space3D."""
    y = down / np.maximum(np.linalg.norm(down, axis=-1, keepdims=True), 1e-9)
    x = x_axis - np.sum(x_axis * y, axis=-1, keepdims=True) * y
    x = x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-9)
    z = np.cross(x, y)
    # R columns are x, y, z;  R[2] = [-cp sy, -sp, cp cy], R[0,1] = -sr cp, R[1,1] = cr cp
    yaw = np.degrees(np.arctan2(-x[..., 2], z[..., 2]))
    pitch = np.degrees(np.arcsin(np.clip(-y[..., 2], -1, 1)))
    roll = np.degrees(np.arctan2(-y[..., 0], y[..., 1]))
    return yaw, pitch, roll


def measure(F: np.ndarray):
    """Per-frame measurements: dict of arrays (degrees) and weights (0 where unusable)."""
    c = {n: F[:, i] for i, n in enumerate(FEAT)}
    ok = c["ok"] > 0
    vs = np.clip(np.minimum(c["v_ls"], c["v_rs"]), 0, 1)
    vh = np.clip(np.minimum(c["v_lh"], c["v_rh"]), 0, 1)

    # world landmarks -> camera coords (x right, y down, z toward camera; MediaPipe z points away)
    def W(j):
        return np.stack([c[f"wx_{j}"], c[f"wy_{j}"], -c[f"wz_{j}"]], 1)
    ls, rs, lh, rh = W("ls"), W("rs"), W("lh"), W("rh")
    down = (lh + rh) / 2 - (ls + rs) / 2
    yaw_sh, pitch_d, _ = decompose(ls - rs, down)
    yaw_hp, _, _ = decompose(lh - rh, down)

    # image landmarks; roll from the 2-D torso axis
    def I(j):
        return np.stack([c[f"ix_{j}"], c[f"iy_{j}"]], 1)
    ils, irs, ilh, irh = I("ls"), I("rs"), I("lh"), I("rh")
    axis2d = (ils + irs) / 2 - (ilh + irh) / 2                    # hip centre -> shoulder centre
    roll = np.degrees(np.arctan2(axis2d[:, 0], -axis2d[:, 1]))
    torso = np.linalg.norm(axis2d, axis=1)
    ok = ok & (np.abs(roll) < 100)                                # inverted fits are not usable
    vs, vh = np.where(ok, vs, 0.0), np.where(ok, vh, 0.0)
    roll_w = np.where(ok & (torso > 1), np.minimum(vs, vh), 0.0)

    # un-roll image vectors so 2-D cues see an upright figure
    r = np.radians(roll)
    cr, sr = np.cos(r), np.sin(r)
    def unroll(v):                                                # undo a clockwise roll (y down)
        return np.stack([v[:, 0] * cr + v[:, 1] * sr, -v[:, 0] * sr + v[:, 1] * cr], 1)
    sh2d, hp2d = unroll(ils - irs), unroll(ilh - irh)

    # 2-D width cue: |cos yaw| from apparent shoulder width over torso length, against the same
    # ratio of the metric 3-D skeleton (how this body looks frontal). That reference used to be
    # the clip's widest view, which is only frontal if the person faces the camera at some point:
    # a person held at 45 deg read as ~31 deg. The skeleton's own proportions need no frontal view.
    sw = sh2d[:, 0]                                               # signed: >0 front-ish
    ratio = np.where(ok & (torso > 1), np.abs(sw) / np.maximum(torso, 1), np.nan)
    w_width = np.linalg.norm(ls - rs, axis=1)
    w_torso = np.linalg.norm(down, axis=1)
    k_body = np.where(ok & (w_torso > 1e-3), w_width / np.maximum(w_torso, 1e-3), np.nan)
    if np.isfinite(k_body).any():
        k = float(np.nanmedian(k_body))                           # bone proportions: one per person
    else:
        k = np.nanpercentile(ratio, 95) if np.isfinite(ratio).any() else 1.0
    cosm = np.nan_to_num(np.clip(ratio / max(k, 1e-6), 0, 1))
    a_sh, a_hp = np.radians(yaw_sh), np.radians(yaw_hp)
    side = np.sign(np.sin(a_sh) + 0.6 * np.sin(a_hp))             # which side: from depth
    front = np.sign(sw + 1e-9)                                    # front/back: from the labels
    a_geo = np.arctan2(side * np.sqrt(1 - cosm ** 2), front * cosm)

    w_sh, w_hp, w_geo = vs * 1.0, vh * 0.7, vs * 0.8
    X = w_sh * np.cos(a_sh) + w_hp * np.cos(a_hp) + w_geo * np.cos(a_geo)
    Y = w_sh * np.sin(a_sh) + w_hp * np.sin(a_hp) + w_geo * np.sin(a_geo)
    agree = np.hypot(X, Y) / np.maximum(w_sh + w_hp + w_geo, 1e-6)
    return dict(yaw=np.degrees(np.arctan2(Y, X)),
                yaw_w=np.where(ok, np.clip(agree, 0, 1) * np.maximum(vs, vh), 0.0),
                roll=roll, roll_w=roll_w,
                pitch_d=pitch_d, pitch_d_w=np.where(ok, np.minimum(vs, vh), 0.0),
                sh2d=sh2d, hp2d=hp2d, vs=vs, vh=vh, ok=ok)


def pitch_from_slant(m, yaw_deg):
    """sin(pitch) from the image slant of the shoulder/hip lines given yaw: an un-rolled
    horizontal line projects as (cos y, -sin p sin y), so sin p = -(dy/dx) * cos y / sin y."""
    yr = np.radians(yaw_deg)
    cy, sy = np.cos(yr), np.sin(yr)
    est, wts = [], []
    for v, vis in ((m["sh2d"], m["vs"]), (m["hp2d"], m["vh"])):
        dx, dy = v[:, 0], v[:, 1]
        with np.errstate(divide="ignore", invalid="ignore"):
            spv = -(dy / dx) * (cy / sy)
        good = np.isfinite(spv) & (np.abs(dx) > 2) & m["ok"]
        est.append(np.degrees(np.arcsin(np.clip(np.nan_to_num(spv), -1, 1))))
        # informative only at oblique yaws: weight ~ sin^2(2 yaw)
        wts.append(np.where(good & (np.abs(spv) < 1.2), vis * np.sin(2 * yr) ** 2, 0.0))
    W = wts[0] + 0.7 * wts[1]
    P = np.where(W > 0, (est[0] * wts[0] + 0.7 * est[1] * wts[1]) / np.maximum(W, 1e-9), 0.0)
    return P, np.clip(W, 0, 1)


def body_scale(F):
    """Apparent size of the person, independent of pose: a weak-perspective fit of the image
    joints to MediaPipe's metric 3-D skeleton (image ~ s * world_xy + t). Because the world
    skeleton already carries the body's rotation and limb positions, s (pixels per metre) does
    not change when the person turns, bends or raises an arm - only when they get nearer or
    further (or the lens zooms). Also returns the skeleton's ankle-to-shoulder chain length (m)."""
    c = {n: F[:, i] for i, n in enumerate(FEAT)}
    js = ["ls", "rs", "le", "re", "lh", "rh", "lk", "rk", "la", "ra"]
    Wd = np.stack([np.stack([c[f"wx_{j}"], c[f"wy_{j}"]], 1) for j in js], 1)       # (n,J,2)
    Im = np.stack([np.stack([c[f"ix_{j}"], c[f"iy_{j}"]], 1) for j in js], 1)
    V = np.stack([c[f"v_{j}"] for j in js], 1) * (c["ok"] > 0)[:, None]
    w = np.where(V > 0.5, V, 0.0)
    n_ok = (w > 0).sum(1)
    wsum = np.maximum(w.sum(1, keepdims=True), 1e-9)
    Wc = Wd - (w[..., None] * Wd).sum(1, keepdims=True) / wsum[..., None]
    Ic = Im - (w[..., None] * Im).sum(1, keepdims=True) / wsum[..., None]
    num = (w * (Ic * Wc).sum(2)).sum(1)
    den = (w * (Wc * Wc).sum(2)).sum(1)
    s = np.where((n_ok >= 4) & (den > 1e-6), num / np.maximum(den, 1e-9), np.nan)
    wt = np.where(np.isfinite(s) & (s > 0), np.clip(n_ok / len(js), 0, 1), 0.0)

    def P(j):
        return np.stack([c[f"wx_{j}"], c[f"wy_{j}"], c[f"wz_{j}"]], 1)
    seg = lambda a, b: np.linalg.norm(P(a) - P(b), axis=1)
    legs = (seg("la", "lk") + seg("lk", "lh") + seg("ra", "rk") + seg("rk", "rh")) / 2
    torso = np.linalg.norm((P("ls") + P("rs")) / 2 - (P("lh") + P("rh")) / 2, axis=1)
    return np.nan_to_num(s), wt, legs + torso


def smooth_series(vals, w, fps: float, sec: float):
    """Fill gaps, 5-frame median against spikes, then confidence-weighted Gaussian."""
    n = len(vals)
    v = np.asarray(vals, float).copy()
    good = w > 0
    if not good.any():
        return np.zeros(n)
    idx = np.arange(n)
    v[~good] = np.interp(idx[~good], idx[good], v[good])
    if n >= 5:
        pad = np.pad(v, 2, mode="edge")
        v = np.median(np.stack([pad[i:i + n] for i in range(5)]), axis=0)
    sig = sec * fps
    if sig > 0.3 and n > 1:
        r = int(3 * sig) + 1
        k = np.exp(-0.5 * (np.arange(-r, r + 1) / sig) ** 2)
        ww = np.maximum(w, 0.05)
        v = (np.convolve(np.pad(v * ww, r, mode="edge"), k, mode="valid")
             / np.convolve(np.pad(ww, r, mode="edge"), k, mode="valid"))
    return v


# --------------------------------------------------------------------------- whole-clip solve
def solve(yaw_obs, weight, fps: float, max_speed: float = 720.0, hints: dict | None = None,
          p_flip: float = 0.04, p_out: float = 0.10, mirror_cost: float = 0.25,
          switch_cost: float = 4.0):
    """Viterbi over (5-degree yaw bin, mirrored?) states.

    Front/back confusion in a pose model is a *mirror*, not a half turn: when it believes it sees
    the back it swaps left/right labels but reads depth from the same image, so the measurement
    becomes 180 - true (reflected across profile). Both readings meet at +-90, so the confusion
    is smooth and continuity alone cannot detect it. Hence a 'mirrored' state that is sticky
    (MediaPipe's confusions last for stretches), slightly penalised (it is usually right), and
    settled by --hint frames. Each measurement may also be a plain 180 flip or an outlier.
    Turn speed is limited by a prior on the per-frame change. Returns the bin-centre track."""
    n = len(yaw_obs)
    centres = np.arange(NBINS) * (360.0 / NBINS) - 180.0
    kappa = 1.0 / math.radians(18.0) ** 2                        # ~18 deg measurement spread
    sig_v = max(5.0, max_speed / fps / 2.0)                       # deg per frame, ~2 sigma = max
    k_tr = 1.0 / math.radians(sig_v) ** 2

    def vm(d):
        return np.exp(kappa * (np.cos(np.radians(d)) - 1))
    S = 2 * NBINS                                                 # [0:NBINS) normal, [NBINS:) mirrored
    U = np.zeros((n, S))
    for i in range(n):
        if weight[i] <= 0:
            continue
        for m, obs in ((0, yaw_obs[i]), (1, 180.0 - yaw_obs[i])):
            d = centres - obs
            lik = (1 - p_flip - p_out) * vm(d) + p_flip * vm(d - 180) + p_out
            U[i, m * NBINS:(m + 1) * NBINS] = (-np.log(lik) + m * mirror_cost) * weight[i]
    for f, y in (hints or {}).items():
        if 0 <= f < n:
            pin = 50.0 * (1 - np.cos(np.radians(centres - y)))     # strong pin, either state
            U[f] += np.concatenate([pin, pin])
    dd = centres[None, :] - centres[:, None]
    Tb = k_tr * (1 - np.cos(np.radians(dd)))                      # cost from bin a -> bin b
    T = np.block([[Tb, Tb + switch_cost], [Tb + switch_cost, Tb]])
    cost = U[0].copy()
    back = np.zeros((n, S), np.int32)
    for i in range(1, n):
        tot = cost[:, None] + T
        back[i] = np.argmin(tot, axis=0)
        cost = tot[back[i], np.arange(S)] + U[i]
    path = np.zeros(n, np.int32)
    path[-1] = int(np.argmin(cost))
    for i in range(n - 1, 0, -1):
        path[i - 1] = back[i][path[i]]
    return centres[path % NBINS]


def refine(track, yaw_obs, weight, fps: float, smooth: float = 0.1, hints: dict | None = None):
    """Pull the coarse track to the closest of the measurement / its mirror / its flip where they
    agree, pin --hint frames to their given value, then unwrap and smooth (confidence-weighted)."""
    n = len(track)
    val = track.astype(float).copy()
    conf = np.zeros(n)
    for i in range(n):
        if weight[i] <= 0:
            continue
        ds = [float(_wrap(c - track[i])) for c in (yaw_obs[i], 180.0 - yaw_obs[i], yaw_obs[i] + 180.0)]
        d = min(ds, key=abs)
        if abs(d) <= 25.0:
            val[i] = track[i] + d
            conf[i] = weight[i] * (1 - abs(d) / 25.0 * 0.5)
    for i, y in (hints or {}).items():                           # a hint is ground truth
        if 0 <= i < n:
            val[i] = track[i] + float(_wrap(y - track[i]))
            conf[i] = 20.0                                       # dominates the smoothing window
    u = np.degrees(np.unwrap(np.radians(val)))
    sig = smooth * fps
    if sig > 0.3 and n > 1:
        r = int(3 * sig) + 1
        k = np.exp(-0.5 * (np.arange(-r, r + 1) / sig) ** 2)
        w = np.maximum(conf, 0.05)                                   # never fully ignore a frame
        num = np.convolve(np.pad(u * w, r, mode="edge"), k, mode="valid")
        den = np.convolve(np.pad(w, r, mode="edge"), k, mode="valid")
        u = num / den
    if u.size and abs(u[0]) > 180:                                   # start in -180..180
        u -= 360.0 * round(u[0] / 360.0)
    return u, conf


def offset_through(series, hints: dict | None):
    """Shift a curve so it passes through hints {index: value}: the correction (hint - curve) is
    interpolated linearly between hinted frames and held before the first / after the last."""
    if not hints:
        return series
    idx = sorted(i for i in hints if 0 <= i < len(series))
    if not idx:
        return series
    off = [hints[i] - series[i] for i in idx]
    return series + np.interp(np.arange(len(series)), idx, off)


def estimate(F, fps, max_speed=720.0, hints=None, smooth=0.1, smooth_tilt=0.25,
             focal_px: float | None = None, pitch_hints=None, roll_hints=None):
    """-> dict with yaw, pitch, roll (degrees per frame), scale, optional cam_dist, confidence,
    and raw measurements."""
    m = measure(F)
    track = solve(m["yaw"], m["yaw_w"], fps, max_speed, hints)
    yaw, conf = refine(track, m["yaw"], m["yaw_w"], fps, smooth, hints)
    conf = np.minimum(conf, 1.0)
    roll = offset_through(smooth_series(m["roll"], m["roll_w"], fps, smooth_tilt), roll_hints)
    p_sl, w_sl = pitch_from_slant(m, yaw)
    w_d = 0.5 * m["pitch_d_w"]
    wsum = w_d + w_sl
    p_raw = np.where(wsum > 0, (m["pitch_d"] * w_d + p_sl * w_sl) / np.maximum(wsum, 1e-9), 0.0)
    pitch = offset_through(smooth_series(p_raw, np.clip(wsum, 0, 1), fps, smooth_tilt), pitch_hints)
    # scale: pixels per metre, smoothed in log space, reported relative to the clip's median
    s_px, s_w, chain = body_scale(F)
    good = s_w > 0
    ref = float(np.median(s_px[good])) if good.any() else 1.0
    log_s = np.where(good, np.log(np.maximum(s_px, 1e-9) / ref), 0.0)
    scale = np.exp(smooth_series(log_s, s_w, fps, smooth_tilt))
    out = dict(yaw=yaw, pitch=pitch, roll=roll, conf=conf, yaw_obs=m["yaw"], yaw_w=m["yaw_w"],
               pitch_raw=p_raw, pitch_w=np.clip(wsum, 0, 1), roll_raw=m["roll"], roll_w=m["roll_w"],
               scale=scale, scale_raw=np.where(good, s_px / ref, np.nan), scale_w=s_w)
    if focal_px and not good.any():
        out["warning"] = "no frame had enough visible joints for a scale fit: cam_dist not written"
    elif focal_px:
        # distance in subject heights = focal / (stature in px). Stature from the skeleton: the
        # ankle-to-shoulder chain is ~78% of standing height (anthropometric average).
        stature_m = float(np.median(chain[good])) / 0.78 if good.any() else 1.75
        out["cam_dist"] = focal_px / (ref * scale * stature_m)
        out["stature_m"] = stature_m
    return out


# --------------------------------------------------------------------------- output
def write_csv(path, frames, R, meta: str):
    with open(path, "w", newline="") as fh:
        fh.write(f"# {meta}\n# degrees. yaw: 0 = facing camera, + = front turns to screen right "
                 "(continuous); pitch: + = camera above; roll: + = subject rotated clockwise\n")
        fh.write("# scale: apparent size relative to the clip's median frame (1 = median)\n")
        if "cam_dist" in R:
            fh.write("# cam_dist: camera distance in subject heights (from --focal-mm)\n")
        w = csv.writer(fh)
        head = ["frame", "yaw", "pitch", "roll", "scale"] + (["cam_dist"] if "cam_dist" in R else []) + ["confidence"]
        w.writerow(head)
        for i, f in enumerate(frames):
            row = [f, f"{R['yaw'][i]:.2f}", f"{R['pitch'][i]:.2f}", f"{R['roll'][i]:.2f}", f"{R['scale'][i]:.4f}"]
            if "cam_dist" in R:
                row.append(f"{R['cam_dist'][i]:.3f}")
            w.writerow(row + [f"{R['conf'][i]:.2f}"])


def write_plot(path, frames, R):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fr = np.asarray(frames)
    fig, axs = plt.subplots(4, 1, figsize=(10, 10.5), dpi=110, sharex=True)
    for ax, key, raw, w, lab in ((axs[0], "yaw", "yaw_obs", "yaw_w", "yaw"),
                                 (axs[1], "pitch", "pitch_raw", "pitch_w", "pitch (+ camera above)"),
                                 (axs[2], "roll", "roll_raw", "roll_w", "roll (+ clockwise)")):
        ok = R[w] > 0
        rv = R[raw] if key != "yaw" else R["yaw"] + _wrap(R["yaw_obs"] - R["yaw"])
        ax.scatter(fr[ok], rv[ok], s=7, c=R[w][ok], cmap="viridis", vmin=0, vmax=1,
                   label="per-frame measurement (colour = weight)")
        ax.plot(fr, R[key], color="#d1495b", lw=2, label="estimate")
        ax.set_ylabel(lab + " (deg)")
        ax.grid(alpha=0.3)
    low = R["conf"] < 0.25
    if low.any():
        axs[0].scatter(fr[low], R["yaw"][low], marker="x", color="#333", s=18, label="low confidence")
    okS = R["scale_w"] > 0
    axs[3].scatter(fr[okS], R["scale_raw"][okS], s=7, c=R["scale_w"][okS], cmap="viridis", vmin=0, vmax=1)
    axs[3].plot(fr, R["scale"], color="#d1495b", lw=2)
    axs[3].set_ylabel("scale (rel. to median)")
    axs[3].grid(alpha=0.3)
    if "cam_dist" in R:
        ax2 = axs[3].twinx()
        ax2.plot(fr, R["cam_dist"], color="#00798c", lw=1.2, ls="--")
        ax2.set_ylabel("cam distance (subject heights)", color="#00798c")
    axs[0].legend(loc="best", fontsize=8)
    axs[3].set_xlabel("frame")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# --------------------------------------------------------------------------- CLI
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Estimate per-frame yaw/pitch/roll of a person for glitch.py --pose.")
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import glitch
    ap.add_argument("--version", action="version",
                    version=f"{getattr(glitch, 'APP_NAME', 'Glitcher')} {getattr(glitch, '__version__', '?')} (estimate_pose.py)")
    ap.add_argument("input", nargs="?", help="folder of frames, quoted glob, or a single image")
    ap.add_argument("--check", action="store_true",
                    help="only test that pose estimation works here (MediaPipe loads, the model is found "
                         "and runs on a blank image); exit code 0 = works, 3 = doesn't")
    ap.add_argument("-o", "--output", type=Path, default=Path("pose.csv"))
    ap.add_argument("--fps", type=float, default=25.0, help="frame rate (default 25, as glitch.py)")
    ap.add_argument("--offset", type=int, default=0, help="add to every frame index (match glitch.py)")
    ap.add_argument("--use-numbers", action="store_true", help="frame index from trailing filename number")
    ap.add_argument("--model", help="path to a pose_landmarker_*.task file (default: one next to this "
                                    "script or in ~/.cache/estimate_pose, else download)")
    ap.add_argument("--variant", choices=["heavy", "full", "lite"], default="heavy",
                    help="model size when downloading (default heavy: most accurate)")
    ap.add_argument("--max-speed", type=float, default=720.0,
                    help="fastest plausible turn, deg/s (default 720); lower = steadier, may lag fast spins")
    ap.add_argument("--smooth", type=float, default=0.1, help="yaw smoothing in seconds (default 0.1)")
    ap.add_argument("--smooth-tilt", type=float, default=0.25,
                    help="pitch/roll smoothing in seconds (default 0.25)")
    ap.add_argument("--hint", action="append", default=[], metavar="FRAME:YAW",
                    help="pin a frame to a known yaw (repeatable), e.g. --hint 0:0 --hint 96:180; also "
                         "settles front/back for the stretch around it")
    ap.add_argument("--pitch-hint", action="append", default=[], metavar="FRAME:DEG",
                    help="known pitch at a frame (repeatable); shifts the pitch curve through it")
    ap.add_argument("--roll-hint", action="append", default=[], metavar="FRAME:DEG",
                    help="known roll at a frame (repeatable); shifts the roll curve through it")
    ap.add_argument("--progress", action="store_true", help="print one progress line per frame")
    ap.add_argument("--landmarks-out", type=Path, metavar="FILE",
                    help="where to write all 33 body landmarks per frame for glitch.py --only/--protect "
                         "(default: <output>_landmarks.npz)")
    ap.add_argument("--focal-mm", type=float,
                    help="lens focal length in mm; enables camera-distance output (dolly vs zoom)")
    ap.add_argument("--sensor-mm", type=float, default=36.0,
                    help="sensor width in mm the image width maps to (default 36 = full frame)")
    ap.add_argument("--plot", type=Path, help="also save a PNG plot of the curves for review")
    ap.add_argument("--exr-space", choices=["linear", "display"], default="linear",
                    help="EXR input colour, as in glitch.py (default linear)")
    ap.add_argument("--exr-alpha", choices=["premultiplied", "straight"], default="premultiplied",
                    help="EXR input alpha, as in glitch.py (default premultiplied)")
    ap.add_argument("--cache", type=Path,
                    help="landmark cache (.npz); reused if it matches, so re-solving with new hints is instant")
    args = ap.parse_args(argv)
    if args.check:
        return self_check(args.model, args.variant)
    if not args.input:
        ap.error("the input (folder of frames, glob or image) is required")
    if not args.fps > 0:
        ap.error("--fps must be > 0")
    if args.cache and args.cache.suffix.lower() != ".npz":
        args.cache = args.cache.with_name(args.cache.name + ".npz")   # numpy would append it anyway

    files = gather(args.input)
    if not files:
        ap.error(f"no input images found for: {args.input}")
    frames = []
    for i, f in enumerate(files):
        fr = i + args.offset
        if args.use_numbers:
            m = re.findall(r"\d+", f.stem)
            fr = int(m[-1]) + args.offset if m else fr
        frames.append(fr)
    if len(set(frames)) != len(frames):
        dup = sorted({f for f in frames if frames.count(f) > 1})[:5]
        ap.error(f"duplicate frame numbers {dup} (check --use-numbers / file names)")
    def parse_hints(items, flag, what):
        out = {}
        for h in items:
            try:
                fs, ys = h.split(":")
                out[frames.index(int(float(fs)))] = float(ys)
            except ValueError:
                ap.error(f"{flag} {h!r}: expected FRAME:{what} with FRAME one of the sequence's frame numbers")
        return out
    hints = parse_hints(args.hint, "--hint", "YAW")
    pitch_hints = parse_hints(args.pitch_hint, "--pitch-hint", "DEG")
    roll_hints = parse_hints(args.roll_hint, "--roll-hint", "DEG")

    F = LM = None
    model = None
    if args.model or not (args.cache and args.cache.is_file()):
        if not _have_mediapipe():
            print("error: " + _no_mediapipe(), file=sys.stderr)
            return 2
        try:
            model = resolve_model(args.model, args.variant)
        except Exception as ex:                       # noqa: BLE001
            print(f"error: could not get the pose model.\n{ex}", file=sys.stderr)
            return 2
    # the cache is valid only for the same files (name, size, mtime), model and fps
    sig = json.dumps({"files": [[f.name, f.stat().st_size, int(f.stat().st_mtime)] for f in files],
                      "model": Path(model).name if model else f"pose_landmarker_{args.variant}.task",
                      "fps": args.fps, "feat": FEAT, "lm": 1})
    if args.cache and args.cache.is_file():
        try:
            with np.load(args.cache, allow_pickle=False) as z:
                if str(z["sig"]) == sig:
                    F, LM = z["F"], z["LM"]
                    print(f"using cached landmarks from {args.cache}", file=sys.stderr)
                else:
                    print("landmark cache is for different frames/model/fps - re-extracting", file=sys.stderr)
        except Exception as ex:                       # noqa: BLE001 - truncated or old-format cache
            print(f"ignoring unreadable cache {args.cache} ({type(ex).__name__})", file=sys.stderr)
    if F is None:
        if not _have_mediapipe():
            print("error: " + _no_mediapipe(), file=sys.stderr)
            return 2
        if model is None:
            try:
                model = resolve_model(args.model, args.variant)
            except Exception as ex:                   # noqa: BLE001
                print(f"error: could not get the pose model.\n{ex}", file=sys.stderr)
                return 2
        try:
            F, LM = extract(files, model, args.fps, io=dict(exr_space=args.exr_space, exr_alpha=args.exr_alpha),
                            progress=args.progress)
        except RuntimeError as ex:
            ap.error(str(ex))
        if args.cache:
            tmp = args.cache.with_name(args.cache.name + ".tmp")
            with open(tmp, "wb") as fh:                         # write fully, then swap in
                np.savez_compressed(fh, F=F, LM=LM, sig=np.array(sig))
            os.replace(tmp, args.cache)

    found = int((F[:, 0] > 0).sum())
    if found == 0:
        ap.error("no person detected in any frame")
    focal_px = None
    if args.focal_mm:
        focal_px = args.focal_mm / args.sensor_mm * open_rgba(files[0], dict(exr_space=args.exr_space)).size[0]
    R = estimate(F, args.fps, args.max_speed, hints, args.smooth, args.smooth_tilt, focal_px,
                 pitch_hints, roll_hints)
    if "warning" in R:
        print("warning: " + R["warning"], file=sys.stderr)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_csv(args.output, frames, R,
              f"estimate_pose.py: {len(files)} frames, person found in {found}, fps {args.fps}")
    if args.plot:
        write_plot(args.plot, frames, R)
    lm_out = args.landmarks_out or args.output.with_name(args.output.stem + "_landmarks.npz")
    write_landmarks(lm_out, frames, LM, open_rgba(files[0], dict(exr_space=args.exr_space)).size)
    low = int((R["conf"] < 0.25).sum())
    print(f"{len(files)} frames -> {args.output}  (person found in {found}; yaw {R['yaw'].min():.0f}.."
          f"{R['yaw'].max():.0f}, pitch {R['pitch'].min():.0f}..{R['pitch'].max():.0f}, roll "
          f"{R['roll'].min():.0f}..{R['roll'].max():.0f} deg; scale {R['scale'].min():.2f}..{R['scale'].max():.2f}"
          + (f"; cam distance {R['cam_dist'].min():.1f}..{R['cam_dist'].max():.1f} subject heights"
             if "cam_dist" in R else "") + f"; {low} low-confidence frames)")
    if low > len(files) * 0.2:
        print("note: many low-confidence frames - check the plot and consider --hint", file=sys.stderr)
    return 0


def write_landmarks(path, frames, LM, size):
    """frames (N,), lm (N, 33, 3) x, y px + visibility (NaN = no person), size (W, H). Writes the
    .npz (glitch.py) and the same data as a .csv next to it (the native engine / Fusion node,
    which can't read numpy files)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as fh:
        np.savez_compressed(fh, frames=np.asarray(frames, np.int64), lm=np.asarray(LM, np.float32),
                            size=np.asarray(size, np.int64))
    os.replace(tmp, path)
    write_landmarks_csv(path.with_suffix(".csv"), frames, LM, size)


def write_landmarks_csv(path, frames, LM, size):
    """Landmarks as text: a `size,W,H` line, a header, then `frame, x0, y0, v0, ..., x32, y32, v32`
    per frame (empty cells = no person)."""
    path = Path(path)
    LM = np.asarray(LM, np.float64)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", newline="") as fh:
        fh.write("# glitcher landmarks: 33 MediaPipe body points per frame, x y in pixels of the "
                 "size below, v = visibility (empty = no person)\n")
        fh.write(f"size,{int(size[0])},{int(size[1])}\n")
        fh.write("frame," + ",".join(f"x{i},y{i},v{i}" for i in range(33)) + "\n")
        for f, lm in zip(frames, LM):
            if not np.isfinite(lm[:33, :2]).all():
                fh.write(f"{int(f)}" + "," * 99 + "\n")
                continue
            fh.write(f"{int(f)}," + ",".join(f"{x:.2f},{y:.2f},{v:.3f}" for x, y, v in lm[:33]) + "\n")
    os.replace(tmp, path)


if __name__ == "__main__":
    raise SystemExit(main())
