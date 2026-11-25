"""Glitcher for DaVinci Resolve (Studio): Workspace > Scripts > Glitcher.

Round trip for the clip under the playhead:

  1. Source: the clip's own RGBA image sequence, or the clip rendered from the timeline with
     its alpha (so a key / Magic Mask / Fusion comp made in Resolve is included).
  2. Glitcher renders it (headless, from the job's preset) or you tune it in the full editor.
  3. Import: the glitch and its glitch pass land in a "Glitcher" bin, the glitch goes on a new
     track above the clip, and its Fusion comp is wired: the pass drives a glow from the glitch
     only (GlitchGlow) and a grade of the glitch only (GlitchGrade).

Each clip gets a job folder (job.json + preset.json + versioned renders v001, v002, ...). The
panel and the Glitcher editor share the job's preset, so whatever you set in the editor is what
the panel renders. This script only uses the Python standard library; Glitcher itself runs as a
separate program (found through ~/.glitcher_resolve.json, written by `Glitcher
--install-resolve`).
"""
import glob as _glob
import json
import os
import re
import struct
import subprocess
import sys
import tempfile
import time
import zlib

PLUGIN_VERSION = "1.7.0"
CONFIG_PATH = os.path.join(os.path.expanduser("~"), ".glitcher_resolve.json")
IMG_EXTS = (".exr", ".png", ".tif", ".tiff")
TRACK_NAME = "Glitcher"
BIN_NAME = "Glitcher"
PASS_LOADER = "GlitchPass"
RESTORE_PRESET = "__glitcher_restore__"

class GlitcherError(Exception):
    """A problem to show the user (not a bug)."""


# =========================================================================== config
def load_config(path=CONFIG_PATH):
    try:
        with open(path, encoding="utf-8") as f:
            cfg = json.load(f)
        return cfg if isinstance(cfg, dict) else {}
    except (OSError, ValueError):
        return {}


def glitcher_command(cfg):
    """[program, args...] that starts Glitcher, or None. GLITCHER_CMD (a path) overrides."""
    env = os.environ.get("GLITCHER_CMD")
    cmd = [env] if env else cfg.get("glitcher")
    if isinstance(cmd, str):
        cmd = [cmd]
    if not cmd or not all(isinstance(c, str) for c in cmd):
        return None
    if not os.path.exists(cmd[0]) or (len(cmd) > 1 and cmd[1].endswith(".py") and not os.path.exists(cmd[1])):
        return None
    return list(cmd)


def cache_root(cfg):
    return cfg.get("cache") or os.path.join(os.path.expanduser("~"), "Documents", "Glitcher Resolve")


# =========================================================================== image sequences
_SEQ = re.compile(r"^(.*?)(\d+)(\.[A-Za-z0-9]+)$")
_BRACKET = re.compile(r"^(.*?)\[(\d+)-(\d+)\](\.[A-Za-z0-9]+)$")


def scan_sequence(folder):
    """The numbered image sequence in a folder (the longest one if there are several):
    dict(prefix, digits, ext, numbers, files) or None."""
    groups = {}
    try:
        names = os.listdir(folder)
    except OSError:
        return None
    for n in names:
        m = _SEQ.match(n)
        if not m or m.group(3).lower() not in IMG_EXTS or ".tmp" in n:
            continue
        key = (m.group(1), len(m.group(2)), m.group(3))
        groups.setdefault(key, []).append((int(m.group(2)), n))
    if not groups:
        return None
    (prefix, digits, ext), items = max(groups.items(), key=lambda kv: len(kv[1]))
    items.sort()
    return {"prefix": prefix, "digits": digits, "ext": ext, "folder": folder,
            "numbers": [i for i, _ in items], "files": [os.path.join(folder, n) for _, n in items]}


def renumber(seq, prefix):
    """Rename a sequence to prefix0001.ext, prefix0002.ext, ... Resolve names its renders by
    the timeline's timecode (a timeline starting at 01:00:00:00 at 25 fps begins at frame
    90000); small numbers from 1 keep every later step free of Resolve's numbering."""
    folder, ext = seq["folder"], seq["ext"]
    digits = max(4, len(str(len(seq["files"]))))
    tmp = []
    for i, f in enumerate(seq["files"]):                        # two steps: names may overlap
        t = os.path.join(folder, ".renum_%06d%s" % (i, ext))
        os.replace(f, t)
        tmp.append(t)
    for i, t in enumerate(tmp, 1):
        os.replace(t, os.path.join(folder, "%s%0*d%s" % (prefix, digits, i, ext)))
    return scan_sequence(folder)


def printf_pattern(seq):
    """Resolve's ImportMedia wants 'name_%04d.exr' + StartIndex / EndIndex."""
    return os.path.join(seq["folder"], seq["prefix"].replace("%", "%%") + "%0" + str(seq["digits"]) + "d" + seq["ext"])


def media_sequence(props):
    """A Media Pool clip's own image sequence on disk, from its clip properties: dict(glob,
    folder, files, numbers, alpha) or None. Resolve shows sequences as 'name_[0001-0144].exr'."""
    path = (props or {}).get("File Path") or ""
    m = _BRACKET.match(os.path.basename(path))
    if not m or m.group(4).lower() not in IMG_EXTS:
        return None
    folder, prefix, first, last, ext = os.path.dirname(path), m.group(1), m.group(2), m.group(3), m.group(4)
    digits = len(first)
    pat = _glob.escape(os.path.join(folder, prefix)) + "[0-9]" * digits + ext
    every = sorted(_glob.glob(pat))                 # what Glitcher will see for this glob, in order
    num = lambda f: int(os.path.basename(f)[len(prefix):len(prefix) + digits])  # noqa: E731
    inside = [i for i, f in enumerate(every) if int(first) <= num(f) <= int(last)]
    if not inside:
        return None
    alpha = str((props or {}).get("Alpha mode") or "").strip().lower()
    return {"glob": pat, "folder": folder, "files": every, "offset": inside[0], "count": len(inside),
            "first": int(first), "last": int(last), "numbers": [num(f) for f in every],
            "alpha": alpha not in ("", "none")}


# =========================================================================== jobs
def safe_name(s, n=40):
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", str(s or "clip")).strip("._") or "clip"
    return s[:n]


def job_folder(root, project_name, item_name, item_uid):
    return os.path.join(root, safe_name(project_name), safe_name(item_name) + "_" + safe_name(item_uid, 8))


def read_job(folder):
    try:
        with open(os.path.join(folder, "job.json"), encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def write_job(folder, job):
    os.makedirs(folder, exist_ok=True)
    tmp = os.path.join(folder, "job.json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(job, f, indent=2)
    os.replace(tmp, os.path.join(folder, "job.json"))


def versions(folder):
    """Existing render versions in a job, oldest first: ['v001', 'v002', ...]."""
    out = os.path.join(folder, "out")
    try:
        return sorted(d for d in os.listdir(out) if re.fullmatch(r"v\d{3}", d)
                      and os.path.isdir(os.path.join(out, d)))
    except OSError:
        return []


def next_version(folder):
    vs = versions(folder)
    return "v%03d" % (int(vs[-1][1:]) + 1 if vs else 1)


def rendered(folder, version, expected):
    """(glitch files, pass files) of a version if it holds all `expected` frame names, else None."""
    out = os.path.join(folder, "out", version)
    got = [os.path.join(out, n) for n in expected]
    if not expected or not all(os.path.isfile(f) for f in got):
        return None
    pas = [os.path.join(out, "glitch_pass", n) for n in expected]
    return got, (pas if all(os.path.isfile(f) for f in pas) else [])


def latest_complete(folder, expected):
    for v in reversed(versions(folder)):
        r = rendered(folder, v, expected)
        if r:
            return v, r
    return None


def ensure_preset(folder, seed=None, intensity=None):
    """The job's preset (the look). Created with the editor's defaults; seed / intensity from the
    panel are written into it, so the panel and the editor never disagree."""
    path = os.path.join(folder, "preset.json")
    try:
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        if not isinstance(d, dict):
            d = {}
    except (OSError, ValueError):
        d = {}
    if seed is not None:
        d["seed"] = int(seed)
    if intensity is not None:
        d["intensity"] = float(intensity)
    d.setdefault("seed", 7)
    d.setdefault("intensity", 0.5)
    os.makedirs(folder, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(d, f, indent=2)
    os.replace(tmp, path)
    return path, d


def copy_preset(src, folder):
    """Use a preset made in the editor as this job's look (its shot paths are dropped: the job
    decides input, output and plate)."""
    with open(src, encoding="utf-8") as f:
        d = json.load(f)
    if not isinstance(d, dict):
        raise GlitcherError("%s is not a Glitcher preset" % src)
    for k in ("input", "output", "plate"):
        d.pop(k, None)
    path = os.path.join(folder, "preset.json")
    os.makedirs(folder, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(d, f, indent=2)
    return path


def out_name(src):
    """Glitcher keeps each frame's name; the job renders EXR, so f_0003.png -> f_0003.exr."""
    return os.path.splitext(os.path.basename(src))[0] + ".exr"


def parse_fps(value, default=25.0):
    m = re.match(r"\s*(\d+(?:\.\d+)?)", str(value or ""))
    return float(m.group(1)) if m else default


# =========================================================================== Resolve side
def clip_offset(info, seq):
    """Index (in the glob listing) of the first frame the timeline clip shows. Resolve reports
    the source start either as a frame number of the sequence or as an offset into the clip;
    the left trim is the fallback."""
    ss = info.get("source_start")
    if ss is not None and seq["first"] <= ss <= seq["last"] and ss in seq["numbers"]:
        return seq["numbers"].index(ss)
    if ss is not None and 0 <= ss < seq["count"]:
        return seq["offset"] + ss
    return seq["offset"] + int(info.get("left_offset") or 0)


def _items(d):
    """Resolve / Fusion return Lua tables as {1.0: x, 2.0: y}; give a plain list."""
    if d is None:
        return []
    if isinstance(d, dict):
        return [d[k] for k in sorted(d)]
    return list(d)


def _unique(values):
    out = []
    for v in values:
        if v is not None and v not in out:
            out.append(v)
    return out


def clip_frames(clip):
    """Frames in a Media Pool clip, from its 'Frames' property (None if Resolve won't say)."""
    try:
        v = str(clip.GetClipProperty("Frames") or "").strip()
    except Exception:                                           # noqa: BLE001
        return None
    return int(v) if v.isdigit() else None


def _start_property(clip):
    """A Media Pool clip's first frame number, if its 'Start' property is a plain number."""
    try:
        v = str(clip.GetClipProperty("Start") or "").strip()
    except Exception:                                           # noqa: BLE001
        return None
    return int(v) if v.isdigit() else None


def _int_or_none(obj, method):
    try:
        v = getattr(obj, method)()
        return None if v is None else int(v)
    except Exception:                                           # noqa: BLE001 - older Resolve
        return None


class Session(object):
    """What the plugin needs from Resolve, in one place."""

    def __init__(self, resolve, log=print):
        self.resolve = resolve
        self.log = log
        self.pm = resolve.GetProjectManager()
        self.project = self.pm.GetCurrentProject()
        if not self.project:
            raise GlitcherError("open a project first")
        self.timeline = self.project.GetCurrentTimeline()
        if not self.timeline:
            raise GlitcherError("open a timeline first")
        self.media_pool = self.project.GetMediaPool()

    # ---------------------------------------------------------------- the clip
    def current_item(self):
        item = self.timeline.GetCurrentVideoItem()
        if not item:
            raise GlitcherError("put the playhead over the clip to glitch (on the Edit or Fusion page)")
        return item

    def fps(self):
        return parse_fps(self.timeline.GetSetting("timelineFrameRate")
                         or self.project.GetSetting("timelineFrameRate"))

    def describe(self, item):
        track = _items(item.GetTrackTypeAndIndex())
        mpi = item.GetMediaPoolItem()
        props = mpi.GetClipProperty() if mpi else {}
        start, dur = int(item.GetStart()), int(item.GetDuration())
        return {"name": item.GetName(), "uid": item.GetUniqueId(), "start": start, "duration": dur,
                "end": start + dur, "track": int(track[1]) if len(track) > 1 else 1,
                "left_offset": int(item.GetLeftOffset() or 0), "props": props or {},
                "source_start": _int_or_none(item, "GetSourceStartFrame"),
                "fusion_comps": int(item.GetFusionCompCount() or 0)}

    def find_item(self, uid):
        for t in range(1, int(self.timeline.GetTrackCount("video")) + 1):
            for it in _items(self.timeline.GetItemListInTrack("video", t)):
                if it.GetUniqueId() == uid:
                    return it
        return None

    # ---------------------------------------------------------------- choosing the source
    def plan_source(self, info, mode, want_plate):
        """-> ("files", seq, (first_index, last_index)) or ("render", reason)."""
        seq = media_sequence(info["props"])
        if mode == "render":
            return "render", "you chose Render from timeline"
        if want_plate:
            if mode == "files":
                raise GlitcherError("a clean plate is rendered from the timeline, so the subject must be "
                                    "too (same size): choose Auto or Render from timeline")
            return "render", "a clean plate needs the subject at timeline size too"
        if mode == "auto" and info["fusion_comps"]:
            return "render", "the clip has a Fusion comp (its key / roto must be rendered in)"
        if seq is None:
            if mode == "files":
                raise GlitcherError("the clip is not an image sequence (EXR / PNG / TIFF) on disk")
            return "render", "the clip is not an image sequence on disk"
        if not seq["alpha"] and mode == "auto":
            return "render", "the clip's media has no alpha (Resolve says Alpha mode: none)"
        a = clip_offset(info, seq)
        b = a + info["duration"] - 1
        if a < 0 or b >= seq["offset"] + seq["count"]:
            if mode == "files":
                raise GlitcherError("the clip uses frames beyond its media (retimed?): use Render from timeline")
            return "render", "the clip seems retimed (uses more frames than its media has)"
        return "files", seq, (a, b)

    # ---------------------------------------------------------------- rendering from the timeline
    def _exr_format(self):
        fmts = self.project.GetRenderFormats() or {}
        fmt = next((v for k, v in fmts.items() if "exr" in (str(k) + str(v)).lower()), None)
        if not fmt:
            raise GlitcherError("this Resolve offers no EXR render format")
        codecs = self.project.GetRenderCodecs(fmt) or {}

        def score(desc):
            d = str(desc).lower()
            return ("half" in d) * 4 + ("zip" in d or "piz" in d) * 2 + ("no compression" not in d)
        if not codecs:
            raise GlitcherError("no EXR codecs offered by Resolve")
        desc = max(codecs, key=score)
        return fmt, codecs[desc], desc

    def render_range(self, info, target, name, what):
        """Render the clip's time range from the timeline to an EXR sequence. what = "subject"
        (only the clip's track, with alpha) or "plate" (only the tracks below it)."""
        p, tl = self.project, self.timeline
        os.makedirs(target, exist_ok=True)
        for f in _glob.glob(os.path.join(_glob.escape(target), "*")):
            if os.path.isfile(f):
                os.remove(f)                                    # a fresh render, not a mix of two
        n_tracks = int(tl.GetTrackCount("video"))
        enabled = {t: bool(tl.GetIsTrackEnabled("video", t)) for t in range(1, n_tracks + 1)}
        keep = ({info["track"]} if what == "subject"
                else {t for t in range(1, info["track"])})
        saved = False
        try:
            saved = bool(p.SaveAsNewRenderPreset(RESTORE_PRESET))
        except Exception:                                       # noqa: BLE001 - older Resolve
            saved = False
        mode = p.GetCurrentRenderMode() if hasattr(p, "GetCurrentRenderMode") else None
        job = None
        try:
            for t, on in enabled.items():
                tl.SetTrackEnable("video", t, bool(on and t in keep))
            fmt, codec, desc = self._exr_format()
            p.SetCurrentRenderMode(1)
            if not p.SetCurrentRenderFormatAndCodec(fmt, codec):
                raise GlitcherError("Resolve refused EXR / %s" % desc)
            ok = p.SetRenderSettings({
                "SelectAllFrames": False, "MarkIn": info["start"], "MarkOut": info["end"] - 1,
                "TargetDir": target, "CustomName": name, "ExportVideo": True, "ExportAudio": False,
                "ExportAlpha": what == "subject", "AlphaMode": 0})
            if not ok:
                raise GlitcherError("Resolve refused the render settings")
            job = p.AddRenderJob()
            if not job:
                raise GlitcherError("Resolve could not add the render job")
            self.log("Resolve renders the %s: %d frames, EXR %s" % (what, info["duration"], desc))
            if not p.StartRendering([job], False):
                raise GlitcherError("Resolve did not start rendering")
            while p.IsRenderingInProgress():
                time.sleep(0.5)
            st = p.GetRenderJobStatus(job) or {}
            if str(st.get("JobStatus", "")).lower() not in ("complete", "completed", ""):
                raise GlitcherError("Resolve render %s: %s" % (st.get("JobStatus"), st.get("Error", "")))
        finally:
            for t, on in enabled.items():
                tl.SetTrackEnable("video", t, on)
            if job:
                try:
                    p.DeleteRenderJob(job)
                except Exception:                               # noqa: BLE001
                    pass
            if saved:
                p.LoadRenderPreset(RESTORE_PRESET)
                try:
                    p.DeleteRenderPreset(RESTORE_PRESET)
                except Exception:                               # noqa: BLE001
                    pass
            if mode is not None:
                p.SetCurrentRenderMode(mode)
        seq = scan_sequence(target)
        if not seq:
            raise GlitcherError("Resolve's render produced no EXR frames in %s" % target)
        n = len(seq["files"])
        if n == info["duration"] + 1:                           # MarkOut taken as exclusive
            os.remove(seq["files"][-1])
            seq = scan_sequence(target)
        elif n != info["duration"]:
            raise GlitcherError("Resolve rendered %d frames, the clip has %d" % (n, info["duration"]))
        self._last_raw_names = [os.path.basename(f) for f in seq["files"]]   # for the probe
        return renumber(seq, name)

    # ---------------------------------------------------------------- import
    def _bin(self):
        root = self.media_pool.GetRootFolder()
        for f in _items(root.GetSubFolderList()):
            if f.GetName() == BIN_NAME:
                return f
        return self.media_pool.AddSubFolder(root, BIN_NAME)

    def import_sequence(self, files, clip_name):
        """Import frames as ONE clip of len(files) frames, checked: first as Resolve's sequence
        pattern (name_%04d.exr + StartIndex / EndIndex), else as the list of files. A wrong
        result (e.g. a single still) is removed from the bin before the next try."""
        seq = scan_sequence(os.path.dirname(files[0]))
        names = set(os.path.basename(f) for f in files)
        nums = [n for n, f in zip(seq["numbers"], seq["files"]) if os.path.basename(f) in names]
        n = len(files)
        attempts = [("pattern %s [%d-%d]" % (os.path.basename(printf_pattern(seq)), nums[0], nums[-1]),
                     [{"FilePath": printf_pattern(seq), "StartIndex": nums[0], "EndIndex": nums[-1]}]),
                    ("list of %d files" % n, list(files))]
        tried = []
        for label, arg in attempts:
            try:
                clips = _items(self.media_pool.ImportMedia(arg))
            except Exception as ex:                             # noqa: BLE001
                tried.append("%s: %s" % (label, ex))
                continue
            got = [clip_frames(c) for c in clips]
            self.log("import as %s -> %d clip(s), frames %s" % (label, len(clips), got))
            if len(clips) == 1 and got[0] in (n, None):
                clip = clips[0]
                for setter in (lambda: clip.SetName(clip_name),
                               lambda: clip.SetClipProperty("Clip Name", clip_name)):
                    try:
                        if setter():
                            break
                    except Exception:                           # noqa: BLE001 - older Resolve
                        continue
                return clip, nums[0]
            tried.append("%s gave %d clip(s) of %s frames" % (label, len(clips), got))
            try:
                self.media_pool.DeleteClips(clips)
            except Exception:                                   # noqa: BLE001
                pass
        raise GlitcherError("Resolve did not import the %d frames as one clip (%s). In the Media Pool, "
                            "check that 'Show Individual Frames' is off, then Import latest again."
                            % (n, "; ".join(tried)))

    def place(self, clip, info, n, first_number=None):
        """Put the glitch on a new track above the clip, at the clip's position, checked.

        The imported clip holds exactly the n frames, so it goes on WHOLE (no start / end frame:
        nothing depends on how Resolve numbers a clip's frames). Only if that doesn't come out as
        n frames at the right spot are explicit ranges tried (0-based, then the clip's own
        numbers). Every wrong placement is removed and the playhead put back."""
        tl = self.timeline
        playhead = None
        try:
            playhead = tl.GetCurrentTimecode()
        except Exception:                                       # noqa: BLE001
            pass
        if not tl.AddTrack("video"):
            raise GlitcherError("could not add a video track")
        idx = int(tl.GetTrackCount("video"))
        try:
            tl.SetTrackName("video", idx, TRACK_NAME)
        except Exception:                                       # noqa: BLE001
            pass
        base = {"mediaPoolItem": clip, "trackIndex": idx, "recordFrame": info["start"], "mediaType": 1}
        tries = [("whole clip", dict(base))]
        for s0 in _unique([0, first_number, _start_property(clip)]):
            tries.append(("frames %d-%d" % (s0, s0 + n - 1), dict(base, startFrame=s0, endFrame=s0 + n - 1)))
        tried = []
        try:
            for label, ci in tries:
                try:
                    items = _items(self.media_pool.AppendToTimeline([ci]))
                except Exception as ex:                         # noqa: BLE001
                    tried.append("%s: %s" % (label, ex))
                    continue
                new = items[0] if items else None
                got = "nothing" if new is None else "%s frame(s) at %s" % (new.GetDuration(), new.GetStart())
                self.log("place as %s -> %s (want %d at %d)" % (label, got, n, info["start"]))
                if new is not None and int(new.GetDuration()) == n and int(new.GetStart()) == info["start"]:
                    return new, idx
                tried.append("%s gave %s" % (label, got))
                if new is not None:
                    try:
                        if not tl.DeleteClips([new], False):
                            new.SetClipEnabled(False)
                    except Exception:                           # noqa: BLE001
                        new.SetClipEnabled(False)
        finally:
            if playhead:
                try:
                    tl.SetCurrentTimecode(playhead)
                except Exception:                               # noqa: BLE001
                    pass
        raise GlitcherError("Resolve would not place the glitch as %d frames at frame %d (%s). The clip "
                            "is in the Glitcher bin; drag it onto track V%d by hand, and please send "
                            "the log." % (n, info["start"], "; ".join(tried), idx))

    def copy_transform(self, src, dst):
        keys = ("Pan", "Tilt", "ZoomX", "ZoomY", "ZoomGang", "RotationAngle", "AnchorPointX", "AnchorPointY",
                "Pitch", "Yaw", "FlipX", "FlipY", "CropLeft", "CropRight", "CropTop", "CropBottom",
                "CropSoftness", "CropRetain", "Opacity")
        for k in keys:
            try:
                v = src.GetProperty(k)
                if v is not None:
                    dst.SetProperty(k, v)
            except Exception:                                   # noqa: BLE001
                continue


# =========================================================================== Fusion comp
def find_input(tool, *names):
    """An input's ID by ID or by its name in the Inspector (IDs differ between versions)."""
    want = set(names)
    for inp in _items(tool.GetInputList()):
        a = inp.GetAttrs() or {}
        if a.get("INPS_ID") in want or a.get("INPS_Name") in want:
            return a.get("INPS_ID")
    return None


def first_tool(comp, kind):
    t = _items(comp.GetToolList(False, kind))
    return t[0] if t else None


def set_loader(loader, first_file, start, n, log):
    """Point the pass Loader at a version and line its first frame up with the comp's first."""
    loader.SetInput("Clip", first_file)
    for key, val in (("GlobalOut", start + n - 1), ("GlobalIn", start)):
        if find_input(loader, key, key.replace("Global", "Global ")):
            loader.SetInput(key, val)
    log("%s reads %s from comp frame %d" % (PASS_LOADER, os.path.basename(first_file), start))


def build_comp(comp, pass_first, n, log):
    """MediaIn (the glitch) -> GlitchGlow (glow mask = the pass) -> GlitchGrade (effect mask =
    the pass) -> MediaOut. The pass comes in through a Loader, made into a mask by PassMatte.
    Built from standard nodes so you can see and change everything."""
    media_in, media_out = first_tool(comp, "MediaIn"), first_tool(comp, "MediaOut")
    if media_in is None or media_out is None:
        raise GlitcherError("the new Fusion comp has no MediaIn / MediaOut")
    start = int((comp.GetAttrs() or {}).get("COMPN_RenderStart") or 0)
    flow = None
    try:
        flow = comp.CurrentFrame.FlowView
        x0, y0 = [float(v) for v in _items(flow.GetPosTable(media_in))][:2]
    except Exception:                                           # noqa: BLE001 - positions are cosmetic
        flow, x0, y0 = None, 0.0, 0.0

    def add(kind, name, dx, dy):
        t = comp.AddTool(kind, -32768, -32768)
        if t is None:
            raise GlitcherError("Fusion could not add a %s node" % kind)
        t.SetAttrs({"TOOLS_Name": name})
        if flow is not None:
            try:
                flow.SetPos(t, x0 + dx, y0 + dy)
            except Exception:                                   # noqa: BLE001
                pass
        return t
    comp.Lock()
    try:
        loader = add("Loader", PASS_LOADER, 0, -2)
        set_loader(loader, pass_first, start, n, log)
        matte = add("BitmapMask", "PassMatte", 1, -2)
        matte.ConnectInput("Image", loader)
        glow = add("Glow", "GlitchGlow", 2, 0)
        glow.ConnectInput("Input", media_in)
        gm = find_input(glow, "GlowMask", "Glow Mask")
        if gm:
            glow.ConnectInput(gm, matte)        # the glitch emits light; the halo may leave it
        else:
            glow.ConnectInput("EffectMask", matte)
            log("note: this Fusion has no Glow Mask input; the pass is used as the glow's effect mask")
        grade = add("ColorCorrector", "GlitchGrade", 3, 0)
        grade.ConnectInput("Input", glow)
        grade.ConnectInput("EffectMask", matte)
        media_out.ConnectInput("Input", grade)
        for tool, text in ((loader, "Glitcher: where the glitch is (white). Select and press 1 to view."),
                           (matte, "Glitcher: raise Low to keep only the hard glitches."),
                           (glow, "Glitcher: light from the glitch only (the pass is its Glow Mask)."),
                           (grade, "Glitcher: colour of the glitch only.")):
            try:
                tool.SetInput("Comments", text)                 # shown when you hover the node
            except Exception:                                   # noqa: BLE001 - cosmetic
                pass
    finally:
        comp.Unlock()
    return {"loader": loader, "matte": matte, "glow": glow, "grade": grade}


def update_comp(comp, pass_first, n, log):
    """A comp copied from the previous version: point its pass Loader at the new version."""
    loader = comp.FindTool(PASS_LOADER) if hasattr(comp, "FindTool") else None
    if loader is None:
        return False
    start = int((comp.GetAttrs() or {}).get("COMPN_RenderStart") or 0)
    set_loader(loader, pass_first, start, n, log)
    return True


# =========================================================================== probe (contract)
PROBE_W, PROBE_H, PROBE_N = 64, 36, 6


def write_png(path, w, h, rgba):
    """A tiny RGBA PNG with the standard library only (no NumPy / Pillow inside Resolve)."""
    raw = b"".join(b"\x00" + bytes(rgba) * w for _ in range(h))

    def chunk(kind, data):
        c = kind + data
        return struct.pack(">I", len(data)) + c + struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)
    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 6, 0, 0, 0))
                + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))


def _safe(fn, *a):
    try:
        return fn(*a)
    except Exception as ex:                                     # noqa: BLE001 - record, go on
        return "ERROR %s: %s" % (type(ex).__name__, ex)


def _call_items(fn, *a):
    """(list of results, None) or ([], "ERROR ...") for an API call returning a Lua table."""
    r = _safe(fn, *a)
    if isinstance(r, str):
        return [], r
    return _items(r), None


def _item_facts(it):
    if it is None or isinstance(it, str):
        return it
    return {"start": _safe(it.GetStart), "duration": _safe(it.GetDuration), "end": _safe(it.GetEnd),
            "left_offset": _safe(it.GetLeftOffset), "source_start": _safe(it.GetSourceStartFrame)}


def probe(resolve, log=print, render=True, folder=None):
    """Ask this Resolve, on a throwaway timeline, every question the plugin depends on, and
    return the answers. Everything it creates is removed again. The report is the contract the
    tests hold the fake Resolve to (tests/resolve_contract/)."""
    pm = resolve.GetProjectManager()
    project = pm.GetCurrentProject()
    if not project:
        raise GlitcherError("open a project first")
    mp = project.GetMediaPool()
    folder = folder or tempfile.mkdtemp(prefix="glitcher_probe_")
    rep = {"probe": 1, "plugin": PLUGIN_VERSION, "python": sys.version.split()[0], "platform": sys.platform,
           "resolve": {"version": _safe(lambda: resolve.GetVersionString()),
                       "product": _safe(lambda: resolve.GetProductName())}}
    seqs = {"small": ("probe_", 101, 4), "timecode": ("tc_", 90000, 8)}
    for key, (prefix, first, digits) in seqs.items():
        d = os.path.join(folder, key)
        os.makedirs(d, exist_ok=True)
        for i in range(PROBE_N):
            write_png(os.path.join(d, "%s%0*d.png" % (prefix, digits, first + i)), PROBE_W, PROBE_H,
                      (40 * i, 120, 200, 255 if i % 2 else 128))
    old_tl = project.GetCurrentTimeline()
    tl, made_clips, root = None, [], mp.GetRootFolder()
    prev_folder = mp.GetCurrentFolder()
    try:
        bin_ = mp.AddSubFolder(root, "Glitcher probe")
        mp.SetCurrentFolder(bin_)
        imports = {}
        clips = {}
        for key, (prefix, first, digits) in seqs.items():
            d = os.path.join(folder, key)
            pat = os.path.join(d, prefix + "%0" + str(digits) + "d.png")
            files = [os.path.join(d, "%s%0*d.png" % (prefix, digits, first + i)) for i in range(PROBE_N)]
            for how, arg in (("pattern", [{"FilePath": pat, "StartIndex": first, "EndIndex": first + PROBE_N - 1}]),
                             ("file_list", files)):
                got, err = _call_items(mp.ImportMedia, arg)
                if err:
                    imports["%s/%s" % (key, how)] = err
                    continue
                made_clips += got
                imports["%s/%s" % (key, how)] = {
                    "clips": len(got),
                    "props": {k: (got[0].GetClipProperty(k) if got else None)
                              for k in ("Frames", "Start", "End", "Start TC", "File Path", "Type", "Format",
                                        "Alpha mode", "Clip Name")}}
                if got and how == "pattern":
                    clips[key] = got[0]
        rep["import"] = imports
        log("probe: imports %s" % {k: (v["clips"], v["props"]["Frames"]) if isinstance(v, dict) else v
                                   for k, v in imports.items()})
        tl = mp.CreateEmptyTimeline("Glitcher probe %s" % time.strftime("%H%M%S"))
        if not tl:
            raise GlitcherError("could not create a probe timeline")
        project.SetCurrentTimeline(tl)
        t0 = int(_safe(tl.GetStartFrame) or 0)
        rep["timeline_start"] = t0
        places = {}
        for key, clip in clips.items():
            first = seqs[key][1]
            variants = [("whole", {}), ("zero_based", {"startFrame": 0, "endFrame": PROBE_N - 1}),
                        ("file_numbers", {"startFrame": first, "endFrame": first + PROBE_N - 1}),
                        ("trim2_zero_based", {"startFrame": 2, "endFrame": PROBE_N - 1}),
                        ("trim2_file_numbers", {"startFrame": first + 2, "endFrame": first + PROBE_N - 1})]
            for name, extra in variants:
                ci = dict({"mediaPoolItem": clip, "trackIndex": 1, "recordFrame": t0 + 10, "mediaType": 1}, **extra)
                its, err = _call_items(mp.AppendToTimeline, [ci])
                places["%s/%s" % (key, name)] = _item_facts(its[0]) if its else err
                if its:
                    _safe(tl.DeleteClips, its, False)
        rep["place"] = places
        log("probe: placements %s" % {k: (v or {}).get("duration") if isinstance(v, dict) else v
                                      for k, v in places.items()})
        # Fusion: what a new comp looks like and which inputs the nodes have
        fus = {}
        its, _err = _call_items(mp.AppendToTimeline, [{"mediaPoolItem": clips.get("small"), "trackIndex": 1,
                                                       "recordFrame": t0, "mediaType": 1}])
        item = its[0] if its else None
        if item is not None:
            comp = _safe(item.AddFusionComp)
            if comp is not None and not isinstance(comp, str):
                a = _safe(comp.GetAttrs) or {}
                fus["render_start"] = a.get("COMPN_RenderStart") if isinstance(a, dict) else a
                fus["render_end"] = a.get("COMPN_RenderEnd") if isinstance(a, dict) else a
                fus["tools"] = sorted(str((t.GetAttrs() or {}).get("TOOLS_RegID"))
                                      for t in _call_items(comp.GetToolList, False)[0])
                fus["flowview"] = not isinstance(_safe(lambda: comp.CurrentFrame.FlowView.GetPosTable), str)
                for kind in ("Loader", "BitmapMask", "Glow", "ColorCorrector"):
                    t = _safe(comp.AddTool, kind, -32768, -32768)
                    fus[kind] = (sorted(str((i.GetAttrs() or {}).get("INPS_ID"))
                                        for i in _call_items(t.GetInputList)[0])
                                 if t is not None and not isinstance(t, str) else t)
                path = os.path.join(folder, "probe.comp")
                fus["export_comp"] = _safe(item.ExportFusionComp, path, 1)
                fus["export_file"] = os.path.isfile(path)
            _safe(tl.DeleteClips, [item], False)
        rep["fusion"] = fus
        # tracks
        rep["tracks"] = {"enabled_query": _safe(tl.GetIsTrackEnabled, "video", 1),
                         "set_enable": _safe(tl.SetTrackEnable, "video", 1, True)}
        # render: naming and range of an EXR render of 3 frames
        if render and clips.get("small") is not None:
            its, _err = _call_items(mp.AppendToTimeline, [{"mediaPoolItem": clips["small"], "trackIndex": 1,
                                                           "recordFrame": t0 + 100, "mediaType": 1}])
            if its:
                sess = Session.__new__(Session)
                sess.resolve, sess.log, sess.project, sess.timeline, sess.media_pool = resolve, log, project, tl, mp
                target = os.path.join(folder, "render")
                info = {"start": t0 + 100, "end": t0 + 103, "duration": 3, "track": 1}
                os.makedirs(target, exist_ok=True)
                try:
                    sess.render_range(info, target, "r_", "subject")
                    rep["render"] = {"ok": True}
                except GlitcherError as ex:
                    rep["render"] = {"ok": False, "error": str(ex)}
                rep["render"]["files_written"] = sorted(os.listdir(target))
                rep["render"]["raw_names"] = getattr(sess, "_last_raw_names", None)
        log("probe: render %s" % rep.get("render"))
    finally:
        if tl is not None:
            if old_tl:
                _safe(project.SetCurrentTimeline, old_tl)
            _safe(mp.DeleteTimelines, [tl])
        if made_clips:
            _safe(mp.DeleteClips, made_clips)
        try:
            for f in _items(root.GetSubFolderList()):
                if f.GetName() == "Glitcher probe":
                    _safe(mp.DeleteFolders, [f])
        except Exception:                                       # noqa: BLE001
            pass
        if prev_folder:
            _safe(mp.SetCurrentFolder, prev_folder)
    return rep


def contract(rep):
    """The facts from a probe report that the plugin depends on (what the tests compare)."""
    imp, pl, fu = rep.get("import", {}), rep.get("place", {}), rep.get("fusion", {})

    def frames(k):
        v = imp.get(k)
        return (v.get("clips"), str((v.get("props") or {}).get("Frames"))) if isinstance(v, dict) else v

    def dur(k):
        v = pl.get(k)
        return (v.get("duration"), v.get("left_offset"), v.get("source_start")) if isinstance(v, dict) else v
    return {"import": {k: frames(k) for k in sorted(imp)},
            "place": {k: dur(k) for k in sorted(pl)},
            "start_property": {k: (v.get("props") or {}).get("Start") for k, v in sorted(imp.items())
                               if isinstance(v, dict)},
            "fusion_inputs": {k: sorted(set(fu.get(k) or []) & set(NEEDED_INPUTS.get(k, [])))
                              for k in NEEDED_INPUTS},
            "render": _render_contract(rep)}


def _render_contract(rep):
    """How Resolve names and counts an EXR sequence render of 3 frames from MarkIn."""
    r = rep.get("render") or {}
    names = r.get("raw_names") or []
    if not names:
        return r.get("error") or None
    m = _SEQ.match(names[0])
    markin = int(rep.get("timeline_start") or 0) + 100
    return {"count": len(names), "digits": len(m.group(2)) if m else None,
            "first_minus_markin": int(m.group(2)) - markin if m else None}


NEEDED_INPUTS = {"Loader": ["Clip", "GlobalIn", "GlobalOut"], "BitmapMask": ["Image"],
                 "Glow": ["Input", "EffectMask", "GlowMask"], "ColorCorrector": ["Input", "EffectMask"]}

# =========================================================================== the round trip
class Plugin(object):
    """Everything the panel's buttons do, without the panel (testable outside Resolve)."""

    def __init__(self, resolve, cfg=None, log=print):
        self.cfg = load_config() if cfg is None else cfg
        self.log = log
        self.s = Session(resolve, log)
        self.proc = None
        self.proc_job = None

    # ---------------------------------------------------------------- job for the current clip
    def job_for(self, item):
        info = self.s.describe(item)
        folder = job_folder(cache_root(self.cfg), self.s.project.GetName(), info["name"], info["uid"])
        return info, folder, read_job(folder)

    def prepare(self, item, mode="auto", want_plate=False, seed=None, intensity=None,
                exr_space="auto", refresh_source=True):
        """Make / refresh the job: sources ready on disk, job.json written. -> (folder, job).
        exr_space: how EXR values are encoded. Resolve's own EXR renders carry what you see
        (display) unless the project is colour managed to a linear output; EXR files from other
        apps are normally scene-linear. Glitcher writes the result back the same way."""
        info, folder, old = self.job_for(item)
        plan = self.s.plan_source(info, mode, want_plate)
        self.log("source: %s" % ("the clip's media files" if plan[0] == "files" else
                                 "render from the timeline (%s)" % plan[1]))
        preset, _ = ensure_preset(folder, seed, intensity)
        if plan[0] == "files":
            seq, (a, b) = plan[1], plan[2]
            src, rng = seq["glob"], [a, b]
            names = [out_name(f) for f in seq["files"][a:b + 1]]
        else:
            src, rng = os.path.join(folder, "src"), None
            seq = scan_sequence(src)
            if refresh_source or not seq or len(seq["files"]) != info["duration"]:
                seq = self.s.render_range(info, src, "src_", "subject")
            else:
                self.log("reusing the source frames rendered earlier (tick Refresh source to re-render)")
            names = [out_name(f) for f in seq["files"]]
        plate = ""
        if want_plate:
            plate = os.path.join(folder, "plate")
            pseq = scan_sequence(plate)
            if refresh_source or not pseq or len(pseq["files"]) != info["duration"]:
                self.s.render_range(info, plate, "plate_", "plate")
        if exr_space not in ("linear", "display"):
            exr_space = "linear" if plan[0] == "files" else "display"
        job = dict(old or {})
        job.update({
            "glitcher_job": 1, "created_by": "Glitcher Resolve plugin %s" % PLUGIN_VERSION,
            "name": "%s (V%d @ %d)" % (info["name"], info["track"], info["start"]),
            "input": src, "range": rng, "plate": plate, "fps": self.s.fps(), "format": "exr",
            "exr_space": exr_space,
            "preset": preset, "output": os.path.join(folder, "out", next_version(folder)),
            "frames": names, "source": plan[0],
            "resolve": {"project": self.s.project.GetName(), "timeline": self.s.timeline.GetName(),
                        "item_uid": info["uid"], "item_name": info["name"], "start": info["start"],
                        "duration": info["duration"], "track": info["track"]}})
        write_job(folder, job)
        return folder, job

    # ---------------------------------------------------------------- Glitcher
    def _command(self):
        cmd = glitcher_command(self.cfg)
        if not cmd:
            raise GlitcherError("Glitcher not found. Open the Glitcher app and choose "
                                "Install DaVinci Resolve plugin (or run: Glitcher --install-resolve).")
        return cmd

    def _popen(self, args, log_path):
        kw = {}
        if sys.platform == "win32":
            kw["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        logf = open(log_path, "w", encoding="utf-8")
        try:
            return subprocess.Popen(args, stdin=subprocess.DEVNULL, stdout=logf, stderr=subprocess.STDOUT, **kw)
        finally:
            logf.close()                                        # the child keeps its own handle

    def render(self, folder, job):
        """Start a headless render of the job (returns at once; poll() follows it)."""
        if self.busy():
            raise GlitcherError("Glitcher is already rendering")
        cpus = max(1, (os.cpu_count() or 2) - 1)
        args = self._command() + ["--render-job", os.path.join(folder, "job.json"), "-j", str(cpus)]
        self.proc = self._popen(args, os.path.join(folder, "render.log"))
        self.proc_job = (folder, job, time.time())
        self.log("Glitcher renders %d frames into %s" % (len(job["frames"]), job["output"]))

    def open_editor(self, folder):
        args = self._command() + ["--job", os.path.join(folder, "job.json")]
        self._popen(args, os.path.join(folder, "editor.log"))
        self.log("opened the Glitcher editor; its changes go into this clip's preset. Render there "
                 "or press Render here, then Import.")

    def busy(self):
        return self.proc is not None and self.proc.poll() is None

    def poll(self):
        """-> None (nothing running), ("running", done, total) or ("done", code, seconds)."""
        if self.proc is None:
            return None
        folder, job, t0 = self.proc_job
        code = self.proc.poll()
        if code is None:
            done = sum(os.path.isfile(os.path.join(job["output"], n)) for n in job["frames"])
            return "running", done, len(job["frames"])
        self.proc = None
        if code != 0:
            tail = ""
            try:
                with open(os.path.join(folder, "render.log"), encoding="utf-8", errors="replace") as f:
                    tail = "".join(f.readlines()[-6:])
            except OSError:
                pass
            self.log("Glitcher stopped with exit code %d:\n%s" % (code, tail.strip()))
        return "done", code, time.time() - t0

    def cancel(self):
        if self.busy():
            self.proc.terminate()
            self.log("render cancelled")

    # ---------------------------------------------------------------- import
    def import_latest(self, folder, job, disable_original=True):
        names = job.get("frames") or []
        found = latest_complete(folder, names)
        if not found:
            raise GlitcherError("no complete render of this clip yet (render it first)")
        version, (glitch_files, pass_files) = found
        info = job["resolve"]
        base = safe_name(info.get("item_name"), 30)
        mp = self.s.media_pool
        prev_folder = mp.GetCurrentFolder()
        mp.SetCurrentFolder(self.s._bin())
        try:
            clip, first_number = self.s.import_sequence(glitch_files, "%s_glitch_%s" % (base, version))
            if pass_files:
                self.s.import_sequence(pass_files, "%s_glitchpass_%s" % (base, version))
        finally:
            if prev_folder:
                mp.SetCurrentFolder(prev_folder)
        original = self.s.find_item(info["item_uid"])
        new, track = self.s.place(clip, info, len(names), first_number)
        if job.get("source") == "files" and original is not None:
            self.s.copy_transform(original, new)               # media at source size: same framing
        # the comp: carry over the last version's (with your tweaks), else build one
        prev = (job.get("imported") or [None])[-1]
        old_item = self.s.find_item(prev["item_uid"]) if prev else None
        comp_note = "no glitch pass, so no comp"
        if pass_files:
            comp = None
            healthy = old_item is not None and int(old_item.GetDuration() or 0) == len(names)
            if healthy and int(old_item.GetFusionCompCount() or 0):     # never copy a broken one
                path = os.path.join(folder, "comp_%s.comp" % prev["version"])
                try:
                    if old_item.ExportFusionComp(path, 1):
                        comp = new.ImportFusionComp(path)
                except Exception:                               # noqa: BLE001
                    comp = None
                if comp is not None and update_comp(comp, pass_files[0], len(names), self.log):
                    comp_note = "comp carried over from %s" % prev["version"]
                else:
                    comp = None
            if comp is None:
                comp = new.AddFusionComp()
                if comp is None:
                    raise GlitcherError("Resolve could not add a Fusion comp to the glitch clip")
                build_comp(comp, pass_files[0], len(names), self.log)
                comp_note = "new comp: GlitchGlow + GlitchGrade masked by the pass"
        if old_item is not None:
            old_item.SetClipEnabled(False)
        if disable_original and original is not None:
            original.SetClipEnabled(False)
        job.setdefault("imported", []).append({"version": version, "item_uid": new.GetUniqueId(),
                                               "track": track, "time": time.strftime("%Y-%m-%d %H:%M")})
        write_job(folder, job)
        self.log("imported %s on track V%d (%s)" % (version, track, comp_note))
        return new


# =========================================================================== the panel
def _app_objects():
    """resolve / fusion / bmd, however this Resolve hands them to scripts."""
    g = globals()
    resolve = g.get("resolve")
    fu = g.get("fusion") or g.get("fu")
    bmd = g.get("bmd")
    app = g.get("app")
    if resolve is None and app is not None:
        resolve = app.GetResolve()
    if resolve is None and bmd is not None:
        resolve = bmd.scriptapp("Resolve")
    if resolve is None:
        try:
            import DaVinciResolveScript as dvr                  # noqa: F401 - external Python
            resolve = dvr.scriptapp("Resolve")
        except ImportError:
            pass
    if fu is None and resolve is not None:
        fu = resolve.Fusion()
    return resolve, fu, bmd


class Panel(object):
    MODES = (("Auto", "auto"), ("Clip's media files", "files"), ("Render from timeline", "render"))
    SPACES = (("Auto (Resolve renders: display, media files: linear)", "auto"),
              ("Display-encoded (as you see it)", "display"), ("Scene-linear", "linear"))

    def __init__(self, resolve, fu, bmd):
        self.ui = fu.UIManager
        self.disp = bmd.UIDispatcher(self.ui)
        self.fu = fu
        ui = self.ui
        self.win = self.disp.AddWindow(
            {"ID": "GlitcherWin", "WindowTitle": "Glitcher %s" % PLUGIN_VERSION, "Geometry": [120, 120, 560, 620]},
            ui.VGroup({"Spacing": 6}, [
                ui.HGroup({"Weight": 0}, [ui.Label({"ID": "Clip", "Text": "", "WordWrap": True}),
                                          ui.Button({"ID": "Refresh", "Text": "Use clip under playhead", "Weight": 0})]),
                ui.HGroup({"Weight": 0}, [ui.Label({"Text": "Source", "Weight": 0, "MinimumSize": [90, 0]}),
                                          ui.ComboBox({"ID": "Mode"})]),
                ui.CheckBox({"ID": "Plate", "Text": "Clean plate from the tracks below (the glitch spills into it)",
                             "Weight": 0}),
                ui.CheckBox({"ID": "Disable", "Text": "Disable the original clip after importing", "Checked": True,
                             "Weight": 0}),
                ui.CheckBox({"ID": "Fresh", "Text": "Refresh source from the timeline before each render",
                             "Checked": True, "Weight": 0}),
                ui.HGroup({"Weight": 0}, [ui.Label({"Text": "EXR colour", "Weight": 0, "MinimumSize": [90, 0]}),
                                          ui.ComboBox({"ID": "Space"})]),
                ui.HGroup({"Weight": 0}, [ui.Label({"Text": "Seed", "Weight": 0, "MinimumSize": [90, 0]}),
                                          ui.SpinBox({"ID": "Seed", "Minimum": 0, "Maximum": 999999, "Value": 7}),
                                          ui.Label({"Text": "Intensity", "Weight": 0}),
                                          ui.DoubleSpinBox({"ID": "Intensity", "Minimum": 0.0, "Maximum": 1.0,
                                                            "SingleStep": 0.05, "Value": 0.5, "Decimals": 2})]),
                ui.HGroup({"Weight": 0}, [ui.Label({"ID": "Preset", "Text": "Look: job preset"}),
                                          ui.Button({"ID": "LoadPreset", "Text": "Use a preset...", "Weight": 0})]),
                ui.HGroup({"Weight": 0}, [ui.Button({"ID": "Editor", "Text": "Open in Glitcher editor"}),
                                          ui.Button({"ID": "Render", "Text": "Render"}),
                                          ui.Button({"ID": "Import", "Text": "Import latest"})]),
                ui.HGroup({"Weight": 0}, [ui.Button({"ID": "Probe", "Text": "Check this Resolve (for bug reports)",
                                                     "Weight": 0})]),
                ui.Label({"ID": "Status", "Text": "", "Weight": 0, "WordWrap": True}),
                ui.TextEdit({"ID": "Log", "ReadOnly": True}),
            ]))
        self.it = self.win.GetItems()
        for label, _m in self.MODES:
            self.it["Mode"].AddItem(label)
        for label, _m in self.SPACES:
            self.it["Space"].AddItem(label)
        self.plugin = None
        self.item = None
        try:
            self.plugin = Plugin(resolve, log=self.log)
        except GlitcherError as ex:
            self.status("⚠ " + str(ex))
        on = self.win.On
        on.GlitcherWin.Close = lambda ev: self.close()
        on.Refresh.Clicked = lambda ev: self.guard(self.refresh)
        on.Editor.Clicked = lambda ev: self.guard(self.editor)
        on.Render.Clicked = lambda ev: self.guard(self.render)
        on.Import.Clicked = lambda ev: self.guard(self.do_import)
        on.LoadPreset.Clicked = lambda ev: self.guard(self.load_preset)
        on.Probe.Clicked = lambda ev: self.guard(self.run_probe)
        self.timer = None
        try:                                                    # polls Glitcher while it renders
            self.timer = ui.Timer({"ID": "GlitcherPoll", "Interval": 500})
            self.disp.On.GlitcherPoll.Timeout = lambda ev: self.guard(self.tick)
        except Exception:                                       # noqa: BLE001
            self.timer = None
        if self.plugin:
            self.guard(self.refresh)
        if self.plugin and glitcher_command(self.plugin.cfg) is None:
            self.status("⚠ Glitcher not found: open the Glitcher app and choose Install DaVinci Resolve "
                        "plugin (or run Glitcher --install-resolve), then reopen this panel.")

    # ---------------------------------------------------------------- helpers
    def log(self, msg):
        try:
            self.it["Log"].Append(str(msg))
        except Exception:                                       # noqa: BLE001
            print(msg)
        try:                                                    # a file you can send when stuck
            path = os.path.join(cache_root(self.plugin.cfg if getattr(self, "plugin", None) else load_config()), "panel.log")
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "a", encoding="utf-8") as f:
                f.write(time.strftime("%Y-%m-%d %H:%M:%S ") + str(msg) + "\n")
        except Exception:                                       # noqa: BLE001
            pass

    def status(self, msg):
        self.it["Status"].Text = str(msg)

    def guard(self, fn):
        try:
            fn()
        except GlitcherError as ex:
            self.status("⚠ " + str(ex))
            self.log("⚠ " + str(ex))
        except Exception as ex:                                 # noqa: BLE001 - never kill the panel
            self.status("⚠ unexpected error: %s" % ex)
            self.log("⚠ %s: %s" % (type(ex).__name__, ex))

    def mode(self):
        return self.MODES[int(self.it["Mode"].CurrentIndex or 0)][1]

    # ---------------------------------------------------------------- actions
    def refresh(self):
        p = self.plugin
        self.item = p.s.current_item()
        info, folder, job = p.job_for(self.item)
        _path, look = ensure_preset(folder)
        self.it["Seed"].Value = int(look.get("seed", 7))
        self.it["Intensity"].Value = float(look.get("intensity", 0.5))
        vs = versions(folder)
        self.it["Clip"].Text = "%s  ·  V%d  ·  %d frames" % (info["name"], info["track"], info["duration"])
        self.status("renders: %s" % (", ".join(vs) if vs else "none yet") +
                    ("  ·  last import: %s" % job["imported"][-1]["version"] if job and job.get("imported") else ""))

    def _prepare(self):
        if self.item is None:
            self.refresh()
        space = self.SPACES[int(self.it["Space"].CurrentIndex or 0)][1]
        return self.plugin.prepare(self.item, self.mode(), bool(self.it["Plate"].Checked),
                                   int(self.it["Seed"].Value), float(self.it["Intensity"].Value),
                                   space, bool(self.it["Fresh"].Checked))

    def editor(self):
        folder, _job = self._prepare()
        self.plugin.open_editor(folder)
        self.status("Glitcher editor open. When it looks right: render there or here, then Import latest.")

    def render(self):
        if self.plugin.busy():
            self.plugin.cancel()
            self.it["Render"].Text = "Render"
            return
        folder, job = self._prepare()
        self.plugin.render(folder, job)
        self.it["Render"].Text = "Cancel"
        self.status("rendering...")
        if self.timer is not None:
            self.timer.Start()
        else:
            self.status("rendering... (press Import latest when it's done)")

    def tick(self):
        r = self.plugin.poll()
        if r is None:
            return
        if r[0] == "running":
            self.status("rendering... %d / %d frames" % (r[1], r[2]))
            return
        if self.timer is not None:
            self.timer.Stop()
        self.it["Render"].Text = "Render"
        if r[1] == 0:
            self.status("rendered in %.0f s, importing..." % r[2])
            self.do_import()
        else:
            self.status("⚠ the render failed (see the log)")

    def do_import(self):
        if self.item is None:
            self.refresh()
        info, folder, job = self.plugin.job_for(self.item)
        if not job:
            raise GlitcherError("nothing rendered for this clip yet")
        self.plugin.import_latest(folder, job, bool(self.it["Disable"].Checked))
        self.status("imported: the glitch is on the Glitcher track, its comp has GlitchGlow and GlitchGrade")

    def load_preset(self):
        path = None
        try:
            path = self.fu.RequestFile(os.path.expanduser("~"), "", {"FReqS_Filter": "Glitcher preset (*.json)|*.json"})
        except Exception:                                       # noqa: BLE001
            path = None
        if not path:
            return
        if self.item is None:
            self.refresh()
        _info, folder, _job = self.plugin.job_for(self.item)
        copy_preset(path, folder)
        self.it["Preset"].Text = "Look: " + os.path.basename(path)
        self.refresh()

    def run_probe(self):
        """Ask Resolve the questions the plugin relies on, on a throwaway timeline, and save the
        answers: resolve_probe.json in the Glitcher Resolve folder. Send that file with a bug."""
        self.status("checking Resolve (a throwaway timeline, a few seconds)...")
        rep = probe(self.plugin.s.resolve, log=self.log)
        rep["contract"] = contract(rep)
        path = os.path.join(cache_root(self.plugin.cfg), "resolve_probe.json")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(rep, f, indent=2, default=str)
        self.status("saved %s - send it along with a bug report" % path)
        self.log("probe saved: %s" % path)

    def close(self):
        if self.plugin and self.plugin.busy():
            self.log("Glitcher keeps rendering in the background; reopen the panel and press Import latest.")
        if self.timer is not None:
            try:
                self.timer.Stop()
            except Exception:                                   # noqa: BLE001
                pass
        self.disp.ExitLoop()

    def run(self):
        self.win.Show()
        self.disp.RunLoop()
        self.win.Hide()


def main():
    resolve, fu, bmd = _app_objects()
    if resolve is None or fu is None or bmd is None:
        print("Glitcher: run this from DaVinci Resolve (Workspace > Scripts > Glitcher).")
        return 1
    Panel(resolve, fu, bmd).run()
    return 0


if __name__ == "__main__" or any(k in globals() for k in ("resolve", "fusion", "fu", "app", "bmd")):
    main()
