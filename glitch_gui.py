#!/usr/bin/env python3
"""
glitch_gui.py - Glitcher, the desktop editor for glitch.py: tweak the effect, preview a frame live, play the
sequence at preview resolution, render the full sequence.

Needs glitch.py in the same folder, plus numpy and Pillow (miniaudio for MP3, optional). Tkinter ships with the standard Python
installer on Windows and macOS (on Linux: the python3-tk package).

    python glitch_gui.py                 # then pick an input folder
    python glitch_gui.py frames/         # open straight away

Design notes
------------
* The GUI never re-implements rendering. Every setting becomes a glitch.py command line, parsed
  by glitch.py's own parser and planned by glitch.plan_render() - the same code path as the CLI.
  "Copy CLI command" therefore reproduces exactly what you see, and a full render from the GUI is
  bit-identical to running that command.
* Previews render in worker processes (rendering is CPU-heavy numpy; threads would freeze the
  window). A live preview is one frame at the chosen preview scale; Play renders the whole range
  at that scale in the background and loops it while it fills.
* Timeline (under the preview): the shaded area is how busy the glitch is over the shot, the
  line is an intensity multiplier you keyframe (x0..x2), orange markers are hits (forced bursts).
  It is written as a --curve CSV + --hits, so the CLI and a full render see exactly the same.
* Pose tab: runs estimate_pose.py in a separate process (MediaPipe stays out of the GUI), keeps
  its landmark cache next to the pose file, and shows yaw / pitch / roll on the timeline. Hints
  you key there are passed as --hint / --pitch-hint / --roll-hint and re-solved from the cache
  (well under a second), so the curve follows your keys almost immediately.
* Audio tab: a WAV / MP3 drives the glitch (glitch.py --audio). Drag on the waveform to pick the
  passage; its start lines up with the first frame. Loudness scales the glitch, beats (spectral-
  flux onsets) force short bursts. With "play audio along" on, ▶ Play follows the audio clock,
  so picture and sound stay in sync even while frames are still rendering. MP3 needs the
  miniaudio package (pip install miniaudio) or ffmpeg; WAV needs nothing extra.
* Every section folds (click its header; remembered), Ctrl+B hides the whole side panel.
  Shortcuts: Ctrl+S save preset (then autosaves every 2 min), Ctrl+O load preset, Ctrl+B panel,
  Ctrl+Z undo, Ctrl+Y / Ctrl+Shift+Z redo.
* Preview scale shrinks the input before glitching. Effects are laid out relative to the person,
  so the look is the same; only pixel-sized details (8 px JPEG blocks, 1 px streaks) differ
  slightly from the full-resolution render.
"""
from __future__ import annotations

import json
import math
import os
import queue
import random
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path

import numpy as np
import tkinter as tk
import tkinter.font as tkfont
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageTk

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import glitch  # noqa: E402

GLOBALS = [  # key, label, min, max, step, default, cli flag
    ("intensity", "Intensity", 0.0, 1.0, 0.01, 0.5, "-i"),
    ("tempo", "Tempo (event speed)", 0.25, 4.0, 0.05, 1.0, "--tempo"),
    ("density", "Density (event count)", 0.0, 3.0, 0.05, 1.0, "--density"),
    ("bursts", "Bursts (calm vs busy)", 0.0, 1.0, 0.01, 0.5, "--bursts"),
    ("motion", "Motion → glitch", 0.0, 1.0, 0.05, 0.0, "--motion"),
    ("spill", "Spill into plate", 0.0, 1.0, 0.05, 0.05, "--spill"),
    ("tint", "Tint (purple/cyan)", 0.0, 1.0, 0.01, 0.0, "--tint"),
    ("follow", "Follow subject (s, 0 = fixed)", 0.0, 2.0, 0.05, 0.0, "--follow"),
    ("pad", "Canvas padding", 0.0, 0.4, 0.01, 0.0, "--pad"),
]
EFFECT_LABELS = {
    "jpeg": "JPEG corruption", "blocks": "Block copies", "slices": "Horizontal tears",
    "wave": "Wave warp", "sort": "Pixel sort", "smear": "Vertical smears",
    "split": "RGB split", "tint": "Tint", "scanlines": "Scanlines", "noise": "Grain",
    "echo": "Echo trails (time)", "stutter": "Frame hold (time)", "mosh": "Datamosh (time)",
}
POSE_CH = {"yaw": ("Yaw", "--hint", "#e8596f"), "pitch": ("Pitch", "--pitch-hint", "#4fd1c5"),
           "roll": ("Roll", "--roll-hint", "#7aa2f7")}
# UI palettes. The preview and the timeline stay dark in both (pictures read better on dark).
THEMES = {
    "dark": dict(bg="#1e1f22", panel="#26282c", field="#2b2d31", btn="#33363b", hover="#3e4147",
                 press="#2a2c30", border="#3a3d43", fg="#e4e5e7", muted="#9a9fa6", accent="#7c5cff",
                 accent_hover="#9077ff", accent_fg="#ffffff", trough="#2f3136", stage="#141416",
                 err="#ff6b6b"),
    "light": dict(bg="#f2f2f5", panel="#e7e7ec", field="#ffffff", btn="#e3e3e9", hover="#d7d7df",
                  press="#cbcbd4", border="#c6c6cf", fg="#1d1d21", muted="#6a6a75", accent="#6a4cf0",
                  accent_hover="#5938e6", accent_fg="#ffffff", trough="#d9d9e0", stage="#26262a",
                  err="#c0392b"),
}
PREFS_FILE = Path.home() / ".glitcher.json"
_OLD_PREFS = Path.home() / ".glitch_gui.json"          # before 1.2: picked up once
TITLE = f"{glitch.APP_NAME} {glitch.__version__.rsplit('.', 1)[0] if glitch.__version__.endswith('.0') else glitch.__version__}"


def _load_prefs() -> dict:
    try:
        src = PREFS_FILE if PREFS_FILE.exists() or not _OLD_PREFS.exists() else _OLD_PREFS
        return json.loads(src.read_text())
    except (OSError, ValueError):
        return {}


def _save_prefs(**kw):
    try:
        d = _load_prefs()
        d.update(kw)
        tmp = PREFS_FILE.with_name(PREFS_FILE.name + ".tmp")
        tmp.write_text(json.dumps(d))
        os.replace(tmp, PREFS_FILE)                      # never a half-written prefs file
    except OSError:
        pass


TARGET_CHOICES = ["all", "head", "face", "torso", "arms", "hands", "legs", "feet", "upper", "lower"]
PATH_KEYS = ("input", "output", "pose", "plate", "audio")


def native_path(p) -> str:
    """A path with this system's separators. Tk's dialogs hand back 'E:/a/b' on Windows while
    Python builds 'E:\\a\\b'; presets may come from another OS. '' stays ''."""
    p = str(p or "").strip()
    if not p:
        return ""
    if os.name == "nt":
        return os.path.normpath(p)
    return os.path.normpath(p.replace("\\", "/"))          # a preset written on Windows


def portable_path(p) -> str:
    """How paths are stored in presets: forward slashes, readable on any OS."""
    return str(p or "").replace("\\", "/")


class AudioPlayer:
    """Plays a stretch of decoded mono audio. miniaudio when installed (any OS, in-process);
    otherwise the system's player on a temp WAV (winsound on Windows, afplay on macOS,
    paplay / aplay / ffplay on Linux). position() is the audio time now, None when stopped."""

    def __init__(self, tmp_dir: Path):
        self.tmp_dir, self.dev, self.proc, self.t0 = tmp_dir, None, None, None
        self.offset = self.until = 0.0
        self.backend = None

    def play(self, x, sr, start: float, stop: float) -> str | None:
        """Start playing [start, stop) seconds; returns an error text or None."""
        self.stop()
        a, b = max(0, int(start * sr)), min(len(x), int(stop * sr))
        if b <= a:
            return "nothing to play there"
        pcm = (np.clip(x[a:b], -1, 1) * 32767).astype("<i2")
        err = None
        try:
            import miniaudio

            def feed():
                need = yield b""
                pos = 0
                while True:
                    chunk = pcm[pos:pos + need]
                    pos += need
                    if len(chunk) < need:                # past the end: silence until stopped
                        chunk = np.concatenate([chunk, np.zeros(need - len(chunk), "<i2")])
                    need = yield chunk.tobytes()
            dev = miniaudio.PlaybackDevice(output_format=miniaudio.SampleFormat.SIGNED16,
                                           nchannels=1, sample_rate=int(sr))
            gen = feed()
            next(gen)
            dev.start(gen)
            self.dev, self.backend = dev, "miniaudio"
        except Exception as ex:                       # noqa: BLE001 - no miniaudio / no device
            err = self._play_external(pcm, sr) or None
            if err:
                return f"cannot play audio here ({ex if 'miniaudio' in str(type(ex)) else err})"
        self.t0, self.offset, self.until = time.monotonic(), a / sr, b / sr
        return None

    def _play_external(self, pcm, sr) -> str:
        import wave
        f = self.tmp_dir / "preview.wav"
        with wave.open(str(f), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(int(sr))
            w.writeframes(pcm.tobytes())
        if sys.platform.startswith("win"):
            import winsound
            winsound.PlaySound(str(f), winsound.SND_FILENAME | winsound.SND_ASYNC)
            self.backend = "winsound"
            return ""
        for cmd in (["afplay"], ["paplay"], ["aplay", "-q"], ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet"]):
            exe = shutil.which(cmd[0])
            if exe:
                self.proc = subprocess.Popen([exe, *cmd[1:], str(f)], stdin=subprocess.DEVNULL,
                                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                self.backend = cmd[0]
                return ""
        return "no audio player found (pip install miniaudio)"

    def position(self):
        if self.t0 is None:
            return None
        t = self.offset + (time.monotonic() - self.t0)
        if t >= self.until or (self.proc is not None and self.proc.poll() is not None):
            self.stop()
            return None
        return t

    def stop(self):
        if self.dev is not None:
            try:
                self.dev.close()
            except Exception:                         # noqa: BLE001
                pass
            self.dev = None
        if self.proc is not None:
            try:
                self.proc.terminate()
            except OSError:
                pass
            self.proc = None
        if self.backend == "winsound":
            import winsound
            winsound.PlaySound(None, 0)
        self.t0, self.backend = None, None


AUTOSAVE_MS = 120_000
UNDO_SETTLE_MS = 400                  # edits closer together than this are one undo step
UNDO_DEPTH = 200
VIEW_KEYS = {"pscale", "compare", "bg", "audio_sync"}   # how you look at it, not what it renders                # preset autosave interval, once a preset has been saved
BACKGROUNDS = {"dark": (34, 34, 34), "black": (0, 0, 0), "white": (255, 255, 255),
               "green": (0, 177, 64), "checker": None}
CACHE_BYTES = 1_500_000_000          # preview cache budget (oldest frames are evicted beyond it)
MAX_WORKERS = 61                     # ProcessPoolExecutor limit on Windows


def _cpus() -> int:
    return max(1, min(MAX_WORKERS, os.cpu_count() or 1))


# --------------------------------------------------------------------------- worker side
def preview_job(job, scale: float):
    """Render one planned job at a reduced input size, return an 8-bit RGBA image for display.
    Runs in a worker process. EXR frames are glitched in float like the full render."""
    src, dst, frame, bbox, kw, ang, camd, io, fo, prev = job
    fo = dict(fo)
    plate_path, prev_plates = fo.pop("plate", None), fo.pop("prev_plates", None)
    prev_frames = fo.pop("prev_frames", None)
    yaw, pitch, roll = ang if ang is not None else (None, None, None)
    img, _raw = glitch.load_frame(src, io)
    size = None
    if scale < 0.999:
        w, h = (img.shape[1], img.shape[0]) if hasattr(img, "shape") else img.size
        size = (max(8, round(w * scale)), max(8, round(h * scale)))
        img = glitch.resize_work(img, size)
        bbox = None if bbox is None else tuple(v * scale for v in bbox)
        if fo.get("landmarks") is not None:               # body-part landmarks follow the resize
            lm = glitch.np.array(fo["landmarks"], glitch.np.float32)
            lm[:, 0] *= size[0] / w
            lm[:, 1] *= size[1] / h
            fo["landmarks"] = lm
    hist = glitch.SeqHistory(prev, io, size, plates=prev_plates, frames=prev_frames) if prev else None
    plate = None
    if plate_path:                                          # comp mode: the plate at preview size
        cur = (img.shape[1], img.shape[0]) if hasattr(img, "shape") else img.size
        plate = glitch.load_plate(plate_path, io, img, cur)
    bleed = kw.get("bleed")
    kw2 = dict(kw, **fo)
    kw2.update(bleed=None if bleed is None else max(0, int(round(bleed * scale))), trim=False)
    out = glitch.glitch(img, t=frame / kw["fps"], frame=frame, bbox=bbox, yaw=yaw,
                        pitch=pitch, roll=roll, cam_dist=camd, history=hist,
                        history_key=(str(src), size, str(plate_path) if plate_path else None),
                        plate=plate, **kw2)
    return glitch.to_display8(out)


def _checker(size, cell=16):
    w, h = size
    im = Image.new("RGB", (w, h), (90, 90, 90))
    px = Image.new("RGB", (cell, cell), (60, 60, 60))
    for y in range(0, h, cell):
        for x in range((y // cell) % 2 * cell, w, cell * 2):
            im.paste(px, (x, y))
    return im


# --------------------------------------------------------------------------- the app
class App:
    def __init__(self, root: tk.Tk, initial: str | None = None):
        self.root = root
        root.title(TITLE)
        self.preset_path = None        # set by the first save; Ctrl+S and autosave write here
        self._preset_saved = None      # what was last written (autosave skips unchanged settings)
        self._preset_loaded = None     # last loaded preset: suggested name for the first save
        self._autosave_id = None
        root.geometry("1280x820")
        root.minsize(980, 640)
        self.q: queue.Queue = queue.Queue()
        self.scan_cache: dict = {}
        self.plan = None
        self.plan_key = None
        self.scan_done = False
        self.cache: OrderedDict = OrderedDict()   # (settings key, scale, idx) -> RGBA image | None
        self.cache_bytes = 0
        self.pending: dict = {}                   # cache key -> future (Play renders in flight)
        self.playing = False
        self.play_idx = 0
        self._tick_id = None
        self.live_token = None
        self.live_future = None
        self.render_cancel = None
        self._debounce = None
        self._frame_debounce = None
        self._photo = None
        self._bg_cache = {}
        self._last_paths = {"input": None, "pose": None, "plate": None, "audio": None}
        self._auto_output = None
        self.tl_keys: list = []        # [(frame, intensity multiplier 0..2)]
        self.tl_hits: list = []        # [(frame, strength 0..1, seconds | None = default length)]
        self.tl_trans: list = []       # [(frame, 'in' | 'out', seconds | None = default length)]
        self.tl_extra: dict = {}       # other curve columns from a loaded file (kept, not drawn)
        self._tl_drag = None
        self._tl_act = (None, None)    # (cache key, activity array)
        # a private temp folder (only this user can open it): no guessable shared /tmp names
        self._tmp_dir = Path(tempfile.mkdtemp(prefix="glitcher_"))
        self._tmp_curve = self._tmp_dir / "timing.csv"
        self.player = AudioPlayer(self._tmp_dir)
        self.pvars: dict[str, tk.Variable] = {}   # Pose tab settings (not part of glitch presets)
        self.pose_hints = {ch: [] for ch in POSE_CH}   # {channel: [(frame, degrees)]}
        self.pose_curves = None        # loaded pose file: {"frame": arr, "yaw": arr, ...}
        self.pose_proc = None
        self._pose_job = None          # (pose path, tmp path, settings json, is a re-solve)
        self._pose_cancelled = False
        self._pose_debounce = None
        self._pose_frozen = None       # value range kept fixed while a hint is dragged
        self.audio = None              # loaded file: dict(path, x, sr, dur)
        self._aview = (0.0, 1.0)       # waveform view window, seconds
        self._acursor = 0.0            # play cursor, seconds
        self._adrag = None
        self._awave_key = None
        self._audio_sync = None        # (monotonic t0, start index) while Play follows the audio
        self._undo: list = []          # snapshots (JSON); _undo_i points at the current one
        self._undo_i = -1
        self._undo_id = None
        self._restoring = False
        self.live_pool = ProcessPoolExecutor(1)
        self.play_pool = ProcessPoolExecutor(max(1, _cpus() - 1))

        self.vars: dict[str, tk.Variable] = {}
        prefs = _load_prefs()
        self._collapsed = set(prefs.get("collapsed", []))
        self._section_toggles = {}
        self.theme = prefs.get("theme", "dark")
        if self.theme not in THEMES:
            self.theme = "dark"
        self.pal = THEMES[self.theme]
        self._style_ttk()
        self._build()
        self._apply_theme(self.theme, save=False)
        if initial:
            self.vars["input"].set(initial)
            self._input_changed()
        root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.after(600, self._undo_commit)             # the starting point is the first step
        for keys, fn in ((("<Control-z>", "<Command-z>"), lambda: self.undo()),
                         (("<Control-y>", "<Control-Y>", "<Control-Z>", "<Control-Shift-z>", "<Control-Shift-Z>",
                           "<Command-Z>", "<Command-Shift-z>"), lambda: self.redo()),
                         (("<Control-b>", "<Control-B>", "<Command-b>"), lambda: self.toggle_panel()),
                         (("<Control-s>", "<Control-S>", "<Command-s>"), lambda: self.save_preset()),
                         (("<Control-o>", "<Control-O>", "<Command-o>"), lambda: self.load_preset())):
            for seq in keys:                                   # Cmd+S / Cmd+O on macOS
                try:
                    root.bind_all(seq, lambda _e, f=fn: (f(), "break")[1])
                except tk.TclError:                            # <Command-…> only exists on macOS
                    pass
        root.after(30, self._poll)

    # ----------------------------------------------------------------- layout
    def _build(self):
        r = self.root
        top = ttk.Frame(r, padding=(8, 6))
        top.pack(fill="x")
        for row, (key, label, kind) in enumerate((("input", "Input frames", "in"),
                                                  ("output", "Output folder", "out"),
                                                  ("pose", "Pose file (optional, 3D)", "pose"),
                                                  ("plate", "Background plate (optional)", "plate"))):
            ttk.Label(top, text=label, width=22).grid(row=row, column=0, sticky="w")
            v = self.vars[key] = tk.StringVar()
            e = ttk.Entry(top, textvariable=v)
            e.grid(row=row, column=1, sticky="ew", padx=4, pady=2)
            e.bind("<Return>", lambda _e, k=key: self._path_changed(k))
            e.bind("<FocusOut>", lambda _e, k=key: self._path_changed(k))
            if kind == "in":
                ttk.Button(top, text="Folder…", command=self._pick_input_dir).grid(row=row, column=2)
                ttk.Button(top, text="File…", command=self._pick_input_file).grid(row=row, column=3)
            elif kind == "out":
                ttk.Button(top, text="Folder…", command=self._pick_output).grid(row=row, column=2)
            elif kind == "pose":
                ttk.Button(top, text="File…", command=self._pick_pose).grid(row=row, column=2)
                ttk.Button(top, text="Clear", command=self._pose_clear_file).grid(row=row, column=3)
            else:
                pb = ttk.Frame(top)
                pb.grid(row=row, column=2, sticky="ew")
                ttk.Button(pb, text="Folder…", width=7, command=lambda: self._pick_plate(True)).pack(side="left", fill="x", expand=True)
                ttk.Button(pb, text="File…", width=5, command=lambda: self._pick_plate(False)).pack(side="left", fill="x", expand=True)
                ttk.Button(top, text="Clear", command=lambda: (self.vars["plate"].set(""), self._path_changed("plate"))
                           ).grid(row=row, column=3)
        self.theme_btn = ttk.Button(top, text="", width=8, style="Ghost.TButton", command=self.toggle_theme)
        self.theme_btn.grid(row=0, column=4, padx=(8, 0))
        self.panel_btn = ttk.Button(top, text="◀ Panel", width=8, style="Ghost.TButton", command=self.toggle_panel)
        self.panel_btn.grid(row=1, column=4, padx=(8, 0))
        self.undo_btn = ttk.Button(top, text="↶ Undo", width=8, style="Ghost.TButton", command=self.undo,
                                   state="disabled")
        self.undo_btn.grid(row=2, column=4, padx=(8, 0))
        self.redo_btn = ttk.Button(top, text="↷ Redo", width=8, style="Ghost.TButton", command=self.redo,
                                   state="disabled")
        self.redo_btn.grid(row=3, column=4, padx=(8, 0))
        top.columnconfigure(1, weight=1)

        body = self.body = ttk.PanedWindow(r, orient="horizontal")
        body.pack(fill="both", expand=True, padx=8, pady=4)

        # left: tabs - Glitch settings (scrollable) and Pose
        self.nb = ttk.Notebook(body)
        scrolls = []
        self._refits = []                                   # re-fit scroll areas after folding

        def scrollable(pad=0):
            outer = ttk.Frame(self.nb)
            canvas = tk.Canvas(outer, width=440, highlightthickness=0)
            sb = ttk.Scrollbar(outer, orient="vertical", command=canvas.yview)
            inner = ttk.Frame(canvas, padding=pad)
            win = canvas.create_window((0, 0), window=inner, anchor="nw")

            def refit(_e=None):
                """Scroll area = the content, but never shorter than the panel: when sections fold
                and everything fits, the view returns to the top (instead of keeping an old scroll
                offset and drawing the content under empty space) and the scrollbar goes away."""
                ch, h = canvas.winfo_height(), inner.winfo_reqheight()
                # the canvas never shrinks an embedded frame by itself: size it to what it needs
                if int(canvas.itemcget(win, "height") or 0) != h:
                    canvas.itemconfigure(win, height=h)
                canvas.configure(scrollregion=(0, 0, canvas.winfo_width(), max(h, ch)))
                fits = h <= ch
                if fits:
                    canvas.yview_moveto(0)
                if fits and sb.winfo_manager():
                    sb.pack_forget()
                elif not fits and not sb.winfo_manager():
                    sb.pack(side="right", fill="y", before=canvas)

            def on_canvas(e):
                canvas.itemconfigure(win, width=e.width)                  # fit width
                refit()
            inner.bind("<Configure>", refit)
            canvas.bind("<Configure>", on_canvas)
            self._refits.append(refit)
            canvas.configure(yscrollcommand=sb.set)
            sb.pack(side="right", fill="y")
            canvas.pack(side="left", fill="both", expand=True)
            scrolls.append((str(outer), canvas))
            return outer, inner

        def wheel(e):
            for prefix, canvas in scrolls:
                if str(e.widget).startswith(prefix):
                    break
            else:
                return
            if e.num in (4, 5):                                  # X11
                step = -1 if e.num == 4 else 1
            elif sys.platform == "darwin":                       # macOS: small deltas
                step = -e.delta
            else:                                                # Windows: multiples of 120
                step = int(-e.delta / 120) or (-1 if e.delta > 0 else 1)
            canvas.yview_scroll(step, "units")
        for seq in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
            self.root.bind_all(seq, wheel)
        left_outer, left = scrollable()
        self.nb.add(left_outer, text="Glitch")
        pose_outer, pose_tab = scrollable(pad=6)
        self._build_pose_tab(pose_tab)
        self.nb.add(pose_outer, text="Pose")
        audio_outer, audio_tab = scrollable(pad=6)
        self._build_audio_tab(audio_tab)
        self.nb.add(audio_outer, text="Audio")
        body.add(self.nb, weight=0)
        self.nb.bind("<<NotebookTabChanged>>", lambda _e: self._mode_changed())

        g = self._section(left, "Global", "global")
        seed_row = ttk.Frame(g)
        seed_row.pack(fill="x", pady=1)
        ttk.Label(seed_row, text="Seed", width=24).pack(side="left")
        self.vars["seed"] = tk.IntVar(value=7)
        sp = ttk.Spinbox(seed_row, from_=0, to=2**31 - 1, textvariable=self.vars["seed"], width=10,
                         command=self._settings_changed)
        sp.pack(side="left")
        sp.bind("<KeyRelease>", lambda _e: self._settings_changed())
        ttk.Button(seed_row, text="Random", width=7,
                   command=lambda: (self.vars["seed"].set(random.randrange(1, 10**6)),
                                    self._settings_changed())).pack(side="left", padx=4)
        fps_row = ttk.Frame(g)
        fps_row.pack(fill="x", pady=1)
        ttk.Label(fps_row, text="Frame rate", width=24).pack(side="left")
        self.vars["fps"] = tk.DoubleVar(value=25.0)
        f = ttk.Spinbox(fps_row, from_=1, to=120, increment=1, textvariable=self.vars["fps"], width=10,
                        command=self._settings_changed)
        f.pack(side="left")
        f.bind("<KeyRelease>", lambda _e: self._settings_changed())
        for key, label, lo, hi, step, default, _flag in GLOBALS:
            self._slider(g, key, label, lo, hi, step, default)
        jq = ttk.Frame(g)
        jq.pack(fill="x", pady=1)
        ttk.Label(jq, text="JPEG quality (0 = auto)", width=24).pack(side="left")
        self.vars["jpeg_quality"] = tk.IntVar(value=0)
        s = ttk.Spinbox(jq, from_=0, to=100, textvariable=self.vars["jpeg_quality"], width=10,
                        command=self._settings_changed)
        s.pack(side="left")
        s.bind("<KeyRelease>", lambda _e: self._settings_changed())
        fr = ttk.Frame(g)
        fr.pack(fill="x", pady=1)
        ttk.Label(fr, text="Frames (single still input)", width=24).pack(side="left")
        self.vars["frames"] = tk.IntVar(value=48)
        s2 = ttk.Spinbox(fr, from_=1, to=10000, textvariable=self.vars["frames"], width=10,
                         command=self._settings_changed)
        s2.pack(side="left")
        s2.bind("<KeyRelease>", lambda _e: self._settings_changed())

        e = self._section(left, "Effects (strength: 1 = default, 0 = off, up to 3)", "effects")
        for fx in glitch.EFFECT_ORDER:
            row = ttk.Frame(e)
            row.pack(fill="x")
            on = self.vars[f"on_{fx}"] = tk.BooleanVar(value=True)
            ttk.Checkbutton(row, text=EFFECT_LABELS[fx], variable=on, width=16,
                            command=self._settings_changed).pack(side="left")
            tv = self.vars[f"target_{fx}"] = tk.StringVar(value="all")
            cb = ttk.Combobox(row, textvariable=tv, values=TARGET_CHOICES, width=6, state="readonly")
            cb.pack(side="left", padx=(0, 4))
            cb.bind("<<ComboboxSelected>>", lambda _e: self._settings_changed())
            gv = self.vars[f"gain_{fx}"] = tk.DoubleVar(value=1.0)
            self._scale(row, gv, 0.0, 3.0, 0.05, self._settings_changed
                     ).pack(side="left", fill="x", expand=True)

        bp = self._section(left, "Body parts", "parts")
        for mode, label in (("only", "Glitch only"), ("protect", "Keep clean")):
            ttk.Label(bp, text=label).pack(anchor="w")
            grid = ttk.Frame(bp)
            grid.pack(fill="x", pady=(0, 4))
            for i, part in enumerate(glitch.PARTS):
                v = self.vars[f"{mode}_{part}"] = tk.BooleanVar(value=False)
                ttk.Checkbutton(grid, text=part, variable=v, command=self._settings_changed
                                ).grid(row=i // 4, column=i % 4, sticky="w", padx=(0, 10))
        self._slider(bp, "part_feather", "Edge softness", 0.2, 3.0, 0.1, 1.0)
        self.parts_lbl = ttk.Label(bp, style="Muted.TLabel", wraplength=400, justify="left")
        self.parts_lbl.pack(fill="x", pady=(4, 0))
        self._parts_status()

        o = self._section(left, "Output format", "output")
        for row, (key, label, values, default) in enumerate((
                ("format", "File type", ["auto", "png", "exr"], "auto"),
                ("exr_space", "EXR colour", ["linear", "display"], "linear"),
                ("exr_alpha", "EXR alpha", ["premultiplied", "straight"], "premultiplied"),
                ("exr_depth", "EXR depth", ["half", "float"], "half"),
                ("exr_compression", "EXR compression", sorted(glitch.EXR_COMPRESSIONS), "zip"))):
            ttk.Label(o, text=label, width=24).grid(row=row, column=0, sticky="w")
            v = self.vars[key] = tk.StringVar(value=default)
            cb = ttk.Combobox(o, textvariable=v, values=values, width=14, state="readonly")
            cb.grid(row=row, column=1, sticky="w", pady=1)
            cb.bind("<<ComboboxSelected>>", lambda _e: self._settings_changed())

        a = self._section(left, "Presets & output", "presets", pady=(0, 0))
        ttk.Button(a, text="Save preset…  Ctrl+S", command=lambda: self.save_preset(ask=True)
                   ).grid(row=0, column=0, sticky="ew", padx=2, pady=2)
        ttk.Button(a, text="Load preset…  Ctrl+O", command=self.load_preset).grid(row=0, column=1, sticky="ew", padx=2, pady=2)
        ttk.Button(a, text="Copy CLI command", command=self.copy_cli).grid(row=1, column=0, sticky="ew", padx=2, pady=2)
        ttk.Button(a, text="Reset to defaults", command=self.reset).grid(row=1, column=1, sticky="ew", padx=2, pady=2)
        self.render_btn = ttk.Button(a, text="Render sequence", style="Accent.TButton", command=self.render_all)
        self.render_btn.grid(row=2, column=0, columnspan=2, sticky="ew", padx=2, pady=(8, 2))
        self.progress = ttk.Progressbar(a, mode="determinate")
        self.progress.grid(row=3, column=0, columnspan=2, sticky="ew", padx=2, pady=2)
        a.columnconfigure(0, weight=1)
        a.columnconfigure(1, weight=1)

        # right: preview
        right = ttk.Frame(body)
        body.add(right, weight=1)
        self.view = tk.Canvas(right, bg=self.pal["stage"], highlightthickness=0)
        self.view.bind("<Configure>", lambda _e: self._redraw())
        self._build_timeline(right)
        ctl = ttk.Frame(right, padding=(0, 6, 0, 0))
        ctl.pack(side="bottom", fill="x")
        self.play_btn = ttk.Button(ctl, text="▶ Play", width=9, style="Accent.TButton", command=self.toggle_play)
        self.play_btn.pack(side="left")
        self.vars["frame"] = tk.IntVar(value=0)
        self.frame_scale = self._scale(ctl, self.vars["frame"], 0, 0, 1, self._frame_moved, show=False)
        self.frame_scale.pack(side="left", fill="x", expand=True, padx=6)
        self.frame_lbl = ttk.Label(ctl, text="frame –", width=16)
        self.frame_lbl.pack(side="left")
        ttk.Label(ctl, text="Preview").pack(side="left", padx=(8, 2))
        self.vars["pscale"] = tk.StringVar(value="50%")
        cb = ttk.Combobox(ctl, textvariable=self.vars["pscale"], values=["25%", "50%", "100%"], width=5,
                          state="readonly")
        cb.pack(side="left")
        cb.bind("<<ComboboxSelected>>", lambda _e: self._settings_changed())
        ttk.Label(ctl, text="BG").pack(side="left", padx=(8, 2))
        self.vars["bg"] = tk.StringVar(value="dark")
        cb2 = ttk.Combobox(ctl, textvariable=self.vars["bg"], values=list(BACKGROUNDS), width=7,
                           state="readonly")
        cb2.pack(side="left")
        cb2.bind("<<ComboboxSelected>>", lambda _e: self._redraw())
        self.vars["compare"] = tk.BooleanVar(value=False)
        ttk.Checkbutton(ctl, text="Original", variable=self.vars["compare"],
                        command=self._frame_moved).pack(side="left", padx=8)
        self.view.pack(fill="both", expand=True)

        self.status = ttk.Label(r, text="Pick an input folder of RGBA PNG frames (or a single PNG).",
                                anchor="w", padding=(10, 4), style="Status.TLabel")
        self.status.pack(fill="x")
        self.current_img = None

    # ----------------------------------------------------------------- timeline
    TL_L, TL_R, TL_T, TL_B = 30, 10, 8, 16          # plot margins (px)

    def _build_timeline(self, parent):
        tf = self._section(parent, "Timeline", "timeline", pady=(6, 0), side="bottom", padding=(4, 2))
        self.tl = tk.Canvas(tf, height=118, bg="#161616", highlightthickness=0, cursor="crosshair")
        self.tl.pack(fill="x")
        self.tl.bind("<Configure>", lambda _e: self._draw_timeline())
        self.tl.bind("<Button-1>", self._tl_press)
        self.tl.bind("<Shift-Button-1>", self._tl_shift)
        self.tl.bind("<B1-Motion>", self._tl_motion)
        self.tl.bind("<ButtonRelease-1>", self._tl_release)
        self.tl.bind("<Double-Button-1>", self._tl_double)
        for seq in ("<Button-3>", "<Button-2>", "<Control-Button-1>"):   # right-click on any OS
            self.tl.bind(seq, self._tl_delete)
        self.tl_row_glitch = ttk.Frame(tf)
        self.tl_row_glitch.pack(fill="x", pady=(3, 0))
        row2 = ttk.Frame(self.tl_row_glitch)               # materialise / vanish
        row2.pack(side="bottom", fill="x", pady=(3, 0))
        ttk.Button(row2, text="▲ Appear at playhead", command=lambda: self._tl_trans_here("in")).pack(side="left")
        ttk.Button(row2, text="▼ Vanish at playhead", command=lambda: self._tl_trans_here("out")
                   ).pack(side="left", padx=4)
        ttk.Label(row2, text="Length (s)").pack(side="left", padx=(8, 2))
        self.vars["trans_len"] = tk.DoubleVar(value=1.0)
        tl_sp = ttk.Spinbox(row2, from_=0.1, to=10, increment=0.1, textvariable=self.vars["trans_len"],
                            width=5, command=self._timing_changed)
        tl_sp.pack(side="left")
        tl_sp.bind("<KeyRelease>", lambda _e: self._timing_changed())
        ttk.Label(row2, style="Muted.TLabel", text="gone before ▲, after ▼"
                  ).pack(side="left", padx=8)
        row = ttk.Frame(self.tl_row_glitch)
        row.pack(fill="x")
        ttk.Button(row, text="Key at playhead", command=self._tl_key_here).pack(side="left")
        ttk.Button(row, text="Hit at playhead", command=self._tl_hit_here).pack(side="left", padx=4)
        ttk.Label(row, text="Hit length (s)").pack(side="left", padx=(8, 2))
        self.vars["hit_len"] = tk.DoubleVar(value=0.5)
        hl = ttk.Spinbox(row, from_=0.05, to=10, increment=0.05, textvariable=self.vars["hit_len"],
                         width=5, command=self._timing_changed)
        hl.pack(side="left")
        hl.bind("<KeyRelease>", lambda _e: self._timing_changed())
        ttk.Button(row, text="Clear", command=self._tl_clear).pack(side="right")
        ttk.Button(row, text="Save curve…", command=self._tl_save).pack(side="right", padx=4)
        ttk.Button(row, text="Load curve…", command=self._tl_load).pack(side="right")
        prow = self.tl_row_pose = ttk.Frame(tf)          # shown in Pose mode instead
        self.tl_pose_lbl = ttk.Label(prow, text="")
        self.tl_pose_lbl.pack(side="left")
        ttk.Button(prow, text="Clear hints", command=lambda: self._pose_clear_hints(self._pose_ch())
                   ).pack(side="right")
        ttk.Button(prow, text="Hint at playhead", command=self._pose_hint_here).pack(side="right", padx=4)
        self.tl_help = ttk.Label(tf, style="Muted.TLabel")
        self.tl_help.pack(fill="x")
        self._mode_changed(redraw=False)

    def _tl_frames(self):
        return [w[2] for w in self.plan["work"]] if self.plan else []

    def _hit_len(self) -> float:
        try:
            return max(0.01, float(self.vars["hit_len"].get()))
        except (tk.TclError, ValueError):
            return 0.5

    def _trans_len(self) -> float:
        try:
            return max(0.05, float(self.vars["trans_len"].get()))
        except (tk.TclError, ValueError):
            return 1.0

    def _timing(self) -> "glitch.Timing":
        try:
            fps = float(self.vars["fps"].get() or 25)
        except (tk.TclError, ValueError):
            fps = 24.0
        keys = dict(self.tl_extra)
        if self.tl_keys:
            keys["intensity"] = list(self.tl_keys)
        hl = self._hit_len()
        tl = self._trans_len()
        return glitch.Timing(keys, [(f, st, sec or hl) for f, st, sec in self.tl_hits], fps,
                             [(f, k, sec or tl) for f, k, sec in self.tl_trans])

    def timing_state(self) -> dict:
        return {"keys": [list(k) for k in self.tl_keys], "hits": [list(h) for h in self.tl_hits],
                "extra": {k: [list(v) for v in kv] for k, kv in self.tl_extra.items()},
                "trans": [list(x) for x in self.tl_trans]}

    def _set_timing_state(self, d: dict):
        self.tl_keys = sorted((float(f), float(v)) for f, v in d.get("keys", []))
        self.tl_hits = sorted((float(h[0]), float(h[1]), (float(h[2]) if len(h) > 2 and h[2] else None))
                              for h in d.get("hits", []))
        self.tl_extra = {k: [(float(f), float(v)) for f, v in kv] for k, kv in d.get("extra", {}).items()}
        self.tl_trans = sorted((float(x[0]), str(x[1]), (float(x[2]) if len(x) > 2 and x[2] else None))
                               for x in d.get("trans", []) if str(x[1]) in ("in", "out"))

    def _timing_changed(self):
        self._draw_timeline()
        self._settings_changed()

    # geometry
    def _tl_geom(self):
        w = max(50, self.tl.winfo_width())
        h = max(40, int(self.tl.cget("height")))
        return self.TL_L, w - self.TL_R, self.TL_T, h - self.TL_B

    def _tl_x(self, frame):
        frs = self._tl_frames()
        x0, x1, _t, _b = self._tl_geom()
        if len(frs) < 2:
            return (x0 + x1) / 2
        i = float(glitch.np.interp(frame, frs, range(len(frs))))
        return x0 + (x1 - x0) * i / (len(frs) - 1)

    def _tl_idx(self, x) -> int:
        frs = self._tl_frames()
        x0, x1, _t, _b = self._tl_geom()
        if len(frs) < 2:
            return 0
        return int(min(len(frs) - 1, max(0, round((x - x0) / (x1 - x0) * (len(frs) - 1)))))

    def _tl_val(self, y, top=2.0) -> float:
        _x0, _x1, t, b = self._tl_geom()
        return float(min(top, max(0.0, (b - y) / (b - t) * top)))

    def _tl_y(self, v, top=2.0):
        _x0, _x1, t, b = self._tl_geom()
        return b - (b - t) * v / top

    def _tl_hit_item(self, x, y, tol=8):
        """Hits are grabbed by their marker (a triangle at strength height)."""
        best = None
        for j, (f, st, _sec) in enumerate(self.tl_hits):
            d = abs(self._tl_x(f) - x) + 0.5 * abs(self._tl_y(st, 1.0) - y)
            if d < tol and (best is None or d < best[0]):
                best = (d, j)
        return None if best is None else best[1]

    def _tl_key_item(self, x, y, tol=8):
        best = None
        for j, (f, v) in enumerate(self.tl_keys):
            d = max(abs(self._tl_x(f) - x), abs(self._tl_y(v) - y))
            if d < tol and (best is None or d < best[0]):
                best = (d, j)
        return None if best is None else best[1]

    def _tl_trans_item(self, x, y, tol=8):
        """Transitions are grabbed by their marker in the strip along the bottom."""
        _x0, _x1, _t, b = self._tl_geom()
        if y < b - 18:
            return None
        best = None
        for j, (f, _k, _sec) in enumerate(self.tl_trans):
            d = abs(self._tl_x(f) - x)
            if d < tol and (best is None or d < best[0]):
                best = (d, j)
        return None if best is None else best[1]

    def _tl_activity(self):
        """Activity (0..1) per frame for the shaded graph, from the same Timing the render uses."""
        frs = self._tl_frames()
        if not frs:
            return glitch.np.zeros(0)
        kw = self.plan["work"][0][4]
        base_gains = dict(kw.get("gains") or {})
        mv = self.plan.get("motion")
        try:
            mamt = float(self.vars["motion"].get())
        except (tk.TclError, ValueError):
            mamt = 0.0
        if mv is not None and len(mv) != len(frs):
            mv = None
        au = self.plan.get("audio")
        try:
            aamt = float(self.vars["audio_strength"].get()) if self.vars["audio"].get().strip() else 0.0
        except (tk.TclError, ValueError):
            aamt = 0.0
        if au is None or len(au["loud"]) != len(frs) or aamt <= 0:
            au, aamt = None, 0.0
        key = json.dumps([self.timing_state(), self._hit_len(), self._trans_len(), mamt, mv is not None,
                          aamt, None if au is None else float(au["loud"].sum() + au["hit"].sum()),
                          kw["seed"], kw["intensity"],
                          kw["bursts"], kw["fps"], frs[0], frs[-1], len(frs)])
        if self._tl_act[0] == key:
            return self._tl_act[1]
        tm = self._timing()
        step = max(1, len(frs) // 1500)                  # very long shots: sample, then hold
        act = glitch.np.zeros(len(frs))
        for i in range(0, len(frs), step):
            fo = tm.overrides(frs[i], kw["intensity"], base_gains)
            if mv is not None and mamt > 0:
                fo = glitch.apply_motion(fo, kw["intensity"], float(mv[i]), mamt)
            if au is not None:
                fo = glitch.apply_audio(fo, kw["intensity"], base_gains, float(au["loud"][i]),
                                        float(au["hit"][i]), aamt)
            act[i:i + step] = glitch.activity_level(frs[i] / kw["fps"], **{**kw, **fo})
        self._tl_act = (key, act)
        return act

    def _draw_timeline(self):
        c = self.tl
        c.delete("all")
        x0, x1, t, b = self._tl_geom()
        frs = self._tl_frames()
        if frs and self.mode == "pose":
            self._draw_pose_timeline()
            return
        if not frs:
            c.create_text((x0 + x1) / 2, (t + b) / 2, text="timeline appears once frames are loaded",
                          fill="#666")
            return
        n = len(frs)
        c.create_rectangle(x0, t, x1, b, outline="#333")
        act = self._tl_activity()
        xs = [x0 + (x1 - x0) * i / max(1, n - 1) for i in range(n)]
        if n == 1:
            xs = [x0, x1]
            act = glitch.np.repeat(act, 2)
        pts = [x0, b] + [v for x, a in zip(xs, act) for v in (x, b - (b - t) * float(a))] + [x1, b]
        c.create_polygon(*pts, fill="#3d2b5e", outline="#8e6fd0")
        mv = self.plan.get("motion") if self.plan else None
        if mv is not None and len(mv) == n and n > 1:          # what the motion slider reacts to
            line = [q for x, v in zip(xs, mv) for q in (x, b - (b - t) * min(1.0, float(v)))]
            c.create_line(*line, fill="#9aa0a6", width=1, dash=(2, 2))
            c.create_text(x1 - 4, t + 2, text="- - motion", anchor="ne", fill="#9aa0a6",
                          font=("TkDefaultFont", 8))
        au = self.plan.get("audio") if self.plan else None
        if au is not None and len(au["hit"]) == n and n > 1:     # beats from the audio
            for i in np.flatnonzero(au["hit"] >= 0.5):
                if i == 0 or au["hit"][i - 1] < au["hit"][i]:
                    c.create_line(xs[i], t, xs[i], t + 7, fill="#ff5fa2", width=2)
        tm = self._timing()
        if tm.trans:                                         # gone stretches + transition ramps
            states = [tm.reveal(f) for f in frs]
            half = (x1 - x0) / max(1, n - 1) / 2
            for i, st in enumerate(states):
                if st == ("hidden",):
                    c.create_rectangle(max(x0, xs[i] - half), t, min(x1, xs[i] + half), b,
                                       fill="#000000", outline="", stipple="gray50")
            for f, kind, sec in self.tl_trans:
                xa = self._tl_x(f)
                xb = self._tl_x(f + (sec or self._trans_len()) * tm.fps)
                col = "#3ccf7a" if kind == "in" else "#e8596f"
                c.create_rectangle(xa, b - 9, max(xa + 2, xb), b - 3, fill=col, outline="", stipple="gray50")
                c.create_line(xa, t, xa, b, fill=col, width=1, dash=(2, 2))
                if kind == "in":
                    c.create_polygon(xa - 6, b - 2, xa + 6, b - 2, xa, b - 13, fill=col, outline="#000")
                else:
                    c.create_polygon(xa - 6, b - 13, xa + 6, b - 13, xa, b - 2, fill=col, outline="#000")
        # hits: decay band + marker
        fps = tm.fps
        hl = self._hit_len()
        for f, st, sec in self.tl_hits:
            xa = self._tl_x(f)
            xb = self._tl_x(f + (sec or hl) * fps)
            c.create_rectangle(xa, t, max(xa + 1, xb), b, fill="#5a3412", outline="", stipple="gray50")
            c.create_line(xa, t, xa, b, fill="#ff9a3c", width=2)
            y = self._tl_y(st, 1.0)
            c.create_polygon(xa - 5, y - 6, xa + 5, y - 6, xa, y + 2, fill="#ff9a3c", outline="#000")
        # intensity multiplier curve
        y1 = self._tl_y(1.0)
        c.create_line(x0, y1, x1, y1, fill="#444", dash=(3, 3))
        for v, lab in ((2.0, "×2"), (1.0, "×1"), (0.0, "×0")):
            c.create_text(x0 - 4, self._tl_y(v), text=lab, anchor="e", fill="#888", font=("TkDefaultFont", 8))
        if self.tl_keys:
            kf, kv = zip(*self.tl_keys)
            line = [v for x, f in zip(xs if n > 1 else [x0], frs)
                    for v in (x, self._tl_y(float(glitch.np.interp(f, kf, kv))))]
            if n == 1:
                line += [x1, line[1]]
            c.create_line(*line, fill="#4fd1c5", width=2)
            for f, v in self.tl_keys:
                x, y = self._tl_x(f), self._tl_y(v)
                c.create_rectangle(x - 4, y - 4, x + 4, y + 4, fill="#4fd1c5", outline="#000")
        c.create_text(x0, b + 2, text=str(frs[0]), anchor="nw", fill="#888", font=("TkDefaultFont", 8))
        c.create_text(x1, b + 2, text=str(frs[-1]), anchor="ne", fill="#888", font=("TkDefaultFont", 8))
        self._draw_playhead()

    def _draw_playhead(self):
        if not self._tl_frames():
            return
        self.tl.delete("playhead")
        _x0, _x1, t, b = self._tl_geom()
        frs = self._tl_frames()
        i = min(int(self.vars["frame"].get()), len(frs) - 1)
        x = self._tl_x(frs[i])
        self.tl.create_line(x, t - 4, x, b + 4, fill="#ffffff", width=1, tags="playhead")
        self._pose_show_now()
        if self.audio is not None and self.player.position() is None:
            self._draw_wave()                                # the frame's spot in the track

    # mouse
    def _tl_press(self, e):
        if not self._tl_frames():
            return
        if self.mode == "pose":
            j = self._ph_item(e.x, e.y)
            if j is not None:
                self._pose_frozen = self._pose_range(self._pose_ch())
                self._tl_drag = ("phint", j)
                return
            self._tl_drag = ("scrub", None)
            self._tl_scrub(e.x)
            return
        j = self._tl_trans_item(e.x, e.y)
        if j is not None:
            self._tl_drag = ("trans", j)
            return
        j = self._tl_key_item(e.x, e.y)
        if j is not None:
            self._tl_drag = ("key", j)
            return
        j = self._tl_hit_item(e.x, e.y)
        if j is not None:
            self._tl_drag = ("hit", j)
            return
        self._tl_drag = ("scrub", None)
        self._tl_scrub(e.x)

    def _tl_scrub(self, x):
        if self.playing:
            self._stop_play()
        self.vars["frame"].set(self._tl_idx(x))
        self._draw_playhead()
        self._frame_moved()

    def _tl_motion(self, e):
        if not self._tl_drag:
            return
        kind, j = self._tl_drag
        if kind == "scrub":
            self._tl_scrub(e.x)
            return
        fr = self._tl_frames()[self._tl_idx(e.x)]
        if kind == "phint":
            hs = self.pose_hints[self._pose_ch()]
            if j < len(hs):
                hs[j] = (fr, self._pose_round(self._ph_val(e.y)))
                self._draw_timeline()
                self._pose_show_now()
            return
        if kind == "trans" and j < len(self.tl_trans):
            self.tl_trans[j] = (fr,) + tuple(self.tl_trans[j][1:])
        elif kind == "key" and j < len(self.tl_keys):
            self.tl_keys[j] = (fr, round(self._tl_val(e.y), 2))
        elif kind == "hit" and j < len(self.tl_hits):
            self.tl_hits[j] = (fr, round(self._tl_val(e.y, 1.0), 2), self.tl_hits[j][2])
        self._tl_act = (None, None)
        self._draw_timeline()

    def _tl_release(self, _e):
        drag, self._tl_drag = self._tl_drag, None
        if drag and drag[0] == "phint":
            ch = self._pose_ch()
            hs = self.pose_hints[ch]
            if drag[1] < len(hs):
                moved = hs[drag[1]]
                self.pose_hints[ch] = sorted([h for h in hs if h[0] != moved[0]] + [moved])
            self._pose_frozen = None
            self._pose_hints_changed()
            return
        if drag and drag[0] == "trans":
            self.tl_trans.sort()
            self._timing_changed()
            return
        if drag and drag[0] in ("key", "hit"):
            if drag[0] == "key":                         # one key per frame: the moved one wins
                moved = self.tl_keys[drag[1]]
                self.tl_keys = sorted({f: v for f, v in self.tl_keys if f != moved[0]} .items()) + [moved]
                self.tl_keys.sort()
            else:
                self.tl_hits.sort()
            self._timing_changed()

    def _tl_double(self, e):
        if self.mode == "pose":
            if self._tl_frames() and self._ph_item(e.x, e.y) is None:
                self._pose_add_hint(self._tl_frames()[self._tl_idx(e.x)], self._ph_val(e.y))
            self._tl_drag = None
            return
        if not self._tl_frames() or self._tl_key_item(e.x, e.y) is not None:
            return
        fr = self._tl_frames()[self._tl_idx(e.x)]
        self.tl_keys = sorted([k for k in self.tl_keys if k[0] != fr] + [(fr, round(self._tl_val(e.y), 2))])
        self._tl_drag = None
        self._timing_changed()

    def _tl_shift(self, e):
        if not self._tl_frames():
            return
        if self.mode == "pose":                          # hint right on the current curve
            fr = self._tl_frames()[self._tl_idx(e.x)]
            now = self._pose_at(fr)
            self._pose_add_hint(fr, now[self._pose_ch()] if now else self._ph_val(e.y))
            self._tl_drag = None
            return
        j = self._tl_hit_item(e.x, e.y, tol=10)
        if j is not None:
            del self.tl_hits[j]
        else:
            fr = self._tl_frames()[self._tl_idx(e.x)]
            self.tl_hits = sorted([h for h in self.tl_hits if h[0] != fr] + [(fr, 1.0, None)])
        self._tl_drag = None
        self._timing_changed()

    def _tl_delete(self, e):
        if self.mode == "pose":
            j = self._ph_item(e.x, e.y, tol=12)
            if j is not None:
                del self.pose_hints[self._pose_ch()][j]
                self._pose_hints_changed()
            return
        j = self._tl_trans_item(e.x, e.y, tol=10)
        if j is not None:
            del self.tl_trans[j]
            self._timing_changed()
            return
        j = self._tl_key_item(e.x, e.y, tol=10)
        if j is not None:
            del self.tl_keys[j]
            self._timing_changed()
            return
        j = self._tl_hit_item(e.x, e.y, tol=12)
        if j is not None:
            del self.tl_hits[j]
            self._timing_changed()

    def _playhead_frame(self):
        frs = self._tl_frames()
        return frs[min(int(self.vars["frame"].get()), len(frs) - 1)] if frs else None

    def _tl_key_here(self):
        fr = self._playhead_frame()
        if fr is None:
            return
        kf = [k for k in self.tl_keys if k[0] != fr]
        v = float(glitch.np.interp(fr, *zip(*self.tl_keys))) if self.tl_keys else 1.0
        self.tl_keys = sorted(kf + [(fr, round(v, 2))])
        self._timing_changed()

    def _tl_hit_here(self):
        fr = self._playhead_frame()
        if fr is None:
            return
        self.tl_hits = sorted([h for h in self.tl_hits if h[0] != fr] + [(fr, 1.0, None)])
        self._timing_changed()

    def _tl_trans_here(self, kind):
        fr = self._playhead_frame()
        if fr is None:
            return
        self.tl_trans = sorted([x for x in self.tl_trans if x[0] != fr] + [(fr, kind, None)])
        self._timing_changed()

    def _tl_clear(self):
        if (self.tl_keys or self.tl_hits or self.tl_extra or self.tl_trans) and messagebox.askyesno(
                "Clear timeline", "Remove all keys, hits, appears and vanishes?"):
            self.tl_keys, self.tl_hits, self.tl_extra, self.tl_trans = [], [], {}, []
            self._timing_changed()

    def _tl_load(self):
        f = filedialog.askopenfilename(title="Load timing curve (CSV)",
                                       filetypes=[("CSV / text", "*.csv *.txt"), ("All", "*.*")])
        if not f:
            return
        try:
            tm = glitch.Timing.from_file(f, self._timing().fps, self._hit_len(), self._trans_len())
        except Exception as ex:                       # noqa: BLE001
            messagebox.showerror("Cannot load curve", str(ex))
            return
        keys = dict(tm.keys)
        self.tl_keys = keys.pop("intensity", [])
        self.tl_extra = keys
        hl = self._hit_len()
        self.tl_hits = [(fr, st, None if abs(sec - hl) < 1e-9 else sec) for fr, st, sec in tm.hits]
        tl = self._trans_len()
        self.tl_trans = [(fr, k, None if abs(sec - tl) < 1e-9 else sec) for fr, k, sec in tm.trans]
        self._timing_changed()
        extra = f" (plus curves for {', '.join(keys)}, kept but not drawn)" if keys else ""
        self.set_status(f"Loaded {len(self.tl_keys)} key(s), {len(self.tl_hits)} hit(s){extra}")

    def _tl_save(self):
        f = filedialog.asksaveasfilename(title="Save timing curve", defaultextension=".csv",
                                         filetypes=[("CSV", "*.csv")])
        if f:
            self._timing().to_csv(f)
            self.set_status(f"Saved timing to {f} (use with glitch.py --curve)")

    # ----------------------------------------------------------------- audio tab
    def _build_audio_tab(self, p):
        v = self.vars
        f = self._section(p, "Audio file (WAV, MP3)", "audio_file", pady=(0, 0))
        row = ttk.Frame(f)
        row.pack(fill="x")
        v["audio"] = tk.StringVar()
        ent = ttk.Entry(row, textvariable=v["audio"])
        ent.pack(side="left", fill="x", expand=True)
        ent.bind("<Return>", lambda _e: self._path_changed("audio"))
        ent.bind("<FocusOut>", lambda _e: self._path_changed("audio"))
        ttk.Button(row, text="File…", command=self._pick_audio).pack(side="left", padx=(4, 0))
        ttk.Button(row, text="Clear", command=lambda: (v["audio"].set(""), self._path_changed("audio"))
                   ).pack(side="left", padx=(4, 0))
        self.audio_info = ttk.Label(f, style="Muted.TLabel", wraplength=400, justify="left",
                                    text="Load a track: loudness drives the glitch, beats trigger bursts.")
        self.audio_info.pack(fill="x", pady=(4, 0))

        ps = self._section(p, "Passage", "audio_passage")
        self.wave = tk.Canvas(ps, height=120, bg="#161616", highlightthickness=0, cursor="crosshair")
        self.wave.pack(fill="x")
        self.wave.bind("<Configure>", lambda _e: self._draw_wave(full=True))
        self.wave.bind("<Button-1>", self._wave_press)
        self.wave.bind("<B1-Motion>", self._wave_motion)
        self.wave.bind("<ButtonRelease-1>", self._wave_release)
        for seq in ("<MouseWheel>", "<Button-4>", "<Button-5>", "<Shift-MouseWheel>",
                    "<Shift-Button-4>", "<Shift-Button-5>"):
            self.wave.bind(seq, self._wave_wheel)
        br = ttk.Frame(ps)
        br.pack(fill="x", pady=(4, 0))
        self.aplay_btn = ttk.Button(br, text="▶ Play", style="Accent.TButton", command=self.audio_play)
        self.aplay_btn.pack(side="left")
        ttk.Button(br, text="Fit", width=4, command=lambda: self._audio_view_fit()).pack(side="left", padx=4)
        ttk.Button(br, text="Passage = shot length", command=self._audio_shot_len).pack(side="left")
        tr = ttk.Frame(ps)
        tr.pack(fill="x", pady=(4, 0))
        for key, lab in (("audio_start", "Start (s)"), ("audio_end", "End (s, 0 = file end)")):
            ttk.Label(tr, text=lab).pack(side="left", padx=(0, 2))
            v[key] = tk.DoubleVar(value=0.0)
            sp = ttk.Spinbox(tr, from_=0, to=36000, increment=0.1, textvariable=v[key], width=7,
                             command=self._audio_passage_changed)
            sp.pack(side="left", padx=(0, 8))
            sp.bind("<Return>", lambda _e: self._audio_passage_changed())
            sp.bind("<FocusOut>", lambda _e: self._audio_passage_changed())
        v["audio_sync"] = tk.BooleanVar(value=True)
        ttk.Checkbutton(ps, text="Play audio along with the preview's ▶ Play",
                        variable=v["audio_sync"]).pack(anchor="w", pady=(4, 0))
        ttk.Label(ps, style="Muted.TLabel", wraplength=400, justify="left", text=(
            "drag: choose the passage (its start lines up with the first frame) · drag its edges to "
            "adjust · click: set the cursor (inside the shot it also moves the frame) · wheel: "
            "zoom · Shift+wheel: scroll")).pack(fill="x", pady=(4, 0))

        rs = self._section(p, "Reaction", "audio_react")
        self._slider(rs, "audio_strength", "Strength (0 = off)", 0.0, 1.0, 0.05, 0.8)
        self._slider(rs, "audio_sens", "Sensitivity", 0.0, 1.0, 0.05, 0.5)
        self.audio_beats_lbl = ttk.Label(rs, style="Muted.TLabel", wraplength=400, justify="left", text="")
        self.audio_beats_lbl.pack(fill="x", pady=(4, 0))
        ttk.Label(rs, style="Muted.TLabel", wraplength=400, justify="left", text=(
            "Loudness scales the glitch (quiet calms it, loud pushes it), each beat forces a short "
            "burst. Sensitivity lowers what counts as loud and how strong an onset must be to count "
            "as a beat. The main timeline shows the result.")).pack(fill="x", pady=(6, 0))

    def _pick_audio(self):
        f = filedialog.askopenfilename(title="Audio", filetypes=[("Audio", "*.wav *.mp3 *.flac *.ogg"),
                                                                 ("All", "*.*")])
        if f:
            self.vars["audio"].set(native_path(f))
            self._path_changed("audio")

    def _audio_file_changed(self):
        path = self.vars["audio"].get().strip()
        self.player.stop()
        self.audio = None
        if path:
            try:
                x, sr = glitch.load_audio(path)
                self.audio = dict(path=path, x=x, sr=sr, dur=len(x) / sr)
                self._aview = (0.0, self.audio["dur"])
                self._acursor = 0.0
                end = float(self.vars["audio_end"].get() or 0)
                if end > self.audio["dur"]:
                    self.vars["audio_end"].set(0.0)
                self.audio_info.configure(
                    text=f"{Path(path).name}: {self.audio['dur']:.1f} s, {sr} Hz", style="Muted.TLabel")
            except (OSError, ValueError) as ex:
                self.audio_info.configure(text=f"⚠ {ex}", style="TLabel")
        else:
            self.audio_info.configure(text="Load a track: loudness drives the glitch, beats trigger bursts.",
                                      style="Muted.TLabel")
        self._draw_wave(full=True)

    def _audio_passage(self):
        """(start, end) in seconds, clipped to the file."""
        dur = self.audio["dur"] if self.audio else 0.0
        try:
            st = float(self.vars["audio_start"].get() or 0)
            en = float(self.vars["audio_end"].get() or 0)
        except (tk.TclError, ValueError):
            st, en = 0.0, 0.0
        st = min(max(0.0, st), dur)
        en = dur if en <= 0 else min(en, dur)
        return st, max(st, en)

    def _shot_seconds(self) -> float:
        n = len(self.plan["work"]) if self.plan else 0
        try:
            fps = float(self.vars["fps"].get() or 25)
        except (tk.TclError, ValueError):
            fps = 25.0
        return n / max(fps, 1e-6)

    def _audio_passage_changed(self):
        self._draw_wave()
        self._settings_changed()

    def _audio_shot_len(self):
        if not self.audio:
            return
        st, _en = self._audio_passage()
        self.vars["audio_end"].set(round(min(self.audio["dur"], st + self._shot_seconds()), 3))
        self._audio_passage_changed()

    def _audio_view_fit(self):
        if self.audio:
            self._aview = (0.0, self.audio["dur"])
            self._draw_wave(full=True)

    # waveform geometry
    def _wx(self, t):
        v0, v1 = self._aview
        return (t - v0) / max(1e-9, v1 - v0) * max(1, self.wave.winfo_width())

    def _wt(self, x):
        v0, v1 = self._aview
        return v0 + x / max(1, self.wave.winfo_width()) * (v1 - v0)

    def _draw_wave(self, full: bool = False):
        c = self.wave
        W, H = max(50, c.winfo_width()), int(c.cget("height"))
        if not self.audio:
            c.delete("all")
            c.create_text(W / 2, H / 2, text="no audio loaded", fill="#666")
            return
        a = self.audio
        key = (a["path"], self._aview, W)
        if full or key != self._awave_key:                 # the waveform itself: only when the view changes
            c.delete("wave")
            v0, v1 = self._aview
            x, sr = a["x"], a["sr"]
            i0, i1 = max(0, int(v0 * sr)), min(len(x), int(v1 * sr))
            seg = x[i0:i1]
            if seg.size >= W:
                n = seg.size // W
                blk = seg[:n * W].reshape(W, n)
                lo, hi = blk.min(axis=1), blk.max(axis=1)
                xs = np.arange(W)
            else:
                lo = hi = seg
                xs = np.arange(seg.size) * (W / max(1, seg.size))
            mid = H / 2
            pts = [q for xx, l, h_ in zip(xs, lo, hi) for q in (xx, mid - h_ * mid * 0.9, xx, mid - l * mid * 0.9)]
            if len(pts) >= 4:
                for k in range(0, len(pts) - 2, 4):
                    c.create_line(pts[k], pts[k + 1], pts[k + 2], pts[k + 3], fill="#7c8aa5", tags="wave")
            self._awave_key = key
        c.delete("ov")
        st, en = self._audio_passage()
        xa, xb = self._wx(st), self._wx(en)
        c.create_rectangle(xa, 0, xb, H, fill="#7c5cff", stipple="gray25", outline="", tags="ov")
        for xx in (xa, xb):
            c.create_line(xx, 0, xx, H, fill="#9077ff", width=2, tags="ov")
        shot_end = min(en, st + self._shot_seconds())
        c.create_line(xa, H - 3, self._wx(shot_end), H - 3, fill="#3ccf7a", width=3, tags="ov")
        au = self.plan.get("audio") if self.plan else None
        if au is not None:
            for bt in au["beats"]:
                xx = self._wx(float(bt))
                if 0 <= xx <= W:
                    c.create_line(xx, 2, xx, 12, fill="#ff5fa2", width=2, tags="ov")
        # current frame and play cursor
        fr_t = self._frame_audio_time()
        if fr_t is not None:
            c.create_line(self._wx(fr_t), 0, self._wx(fr_t), H, fill="#ffffff", tags="ov")
        pt = self.player.position()
        cur = pt if pt is not None else self._acursor
        c.create_line(self._wx(cur), 0, self._wx(cur), H, fill="#3ccf7a", dash=(3, 2), tags="ov")
        c.create_text(4, H - 8, anchor="w", fill="#888", font=("TkDefaultFont", 8), tags="ov",
                      text=f"{self._aview[0]:.1f}s")
        c.create_text(W - 4, H - 8, anchor="e", fill="#888", font=("TkDefaultFont", 8), tags="ov",
                      text=f"{self._aview[1]:.1f}s")

    def _frame_audio_time(self):
        if not self.audio or not self.plan:
            return None
        st, en = self._audio_passage()
        i = min(int(self.vars["frame"].get()), len(self.plan["work"]) - 1)
        t = st + i / max(1e-6, len(self.plan["work"]) / max(1e-6, self._shot_seconds()))
        return t if t < en else None

    # waveform mouse
    def _wave_press(self, e):
        if not self.audio:
            return
        st, en = self._audio_passage()
        for which, t in (("start", st), ("end", en)):
            if abs(self._wx(t) - e.x) <= 6:
                self._adrag = ("edge", which)
                return
        self._adrag = ("new", e.x)

    def _wave_motion(self, e):
        if not self.audio or not self._adrag:
            return
        t = min(max(0.0, self._wt(e.x)), self.audio["dur"])
        if self._adrag[0] == "edge":
            st, en = self._audio_passage()
            if self._adrag[1] == "start":
                self.vars["audio_start"].set(round(min(t, en - 0.05), 3))
            else:
                self.vars["audio_end"].set(round(max(t, st + 0.05), 3))
        elif abs(e.x - self._adrag[1]) > 4:                  # a drag: a new passage
            t0 = min(max(0.0, self._wt(self._adrag[1])), self.audio["dur"])
            self.vars["audio_start"].set(round(min(t0, t), 3))
            self.vars["audio_end"].set(round(max(t0, t), 3))
            self._adrag = ("new-drag", self._adrag[1])
        self._draw_wave()

    def _wave_release(self, e):
        drag, self._adrag = self._adrag, None
        if not self.audio or drag is None:
            return
        if drag[0] == "new":                                 # a click: cursor (and frame) here
            t = min(max(0.0, self._wt(e.x)), self.audio["dur"])
            self._acursor = t
            st, en = self._audio_passage()
            if self.plan and st <= t < min(en, st + self._shot_seconds()):
                n = len(self.plan["work"])
                i = int((t - st) / self._shot_seconds() * n)
                self.vars["frame"].set(min(n - 1, max(0, i)))
                self._frame_moved()
            self._draw_wave()
            if self.player.position() is not None:          # playing: jump there
                self.audio_play(restart_at=t)
            return
        self._audio_passage_changed()

    def _wave_wheel(self, e):
        if not self.audio:
            return
        up = getattr(e, "delta", 0) > 0 or e.num == 4
        v0, v1 = self._aview
        span = v1 - v0
        if e.state & 0x1:                                   # Shift: scroll
            d = span * 0.15 * (-1 if up else 1)
            v0, v1 = v0 + d, v1 + d
        else:                                               # zoom around the pointer
            t = self._wt(e.x)
            f = 0.8 if up else 1.25
            v0, v1 = t - (t - v0) * f, t + (v1 - t) * f
        dur = self.audio["dur"]
        span = min(max(v1 - v0, 0.05), dur)
        v0 = min(max(0.0, v0), dur - span)
        self._aview = (v0, v0 + span)
        self._draw_wave(full=True)

    # playback
    def audio_play(self, restart_at=None):
        if not self.audio:
            return
        if self.player.position() is not None and restart_at is None:
            self.player.stop()
            self.aplay_btn.configure(text="▶ Play")
            return
        st, en = self._audio_passage()
        t = restart_at if restart_at is not None else (self._acursor if st <= self._acursor < en else st)
        err = self.player.play(self.audio["x"], self.audio["sr"], t, en if t < en else self.audio["dur"])
        if err:
            self.set_status(f"⚠ {err}", error=True)
            return
        self.aplay_btn.configure(text="■ Stop")
        self._audio_tick()

    def _audio_tick(self):
        pos = self.player.position()
        self._draw_wave()
        if pos is None:
            self.aplay_btn.configure(text="▶ Play")
            return
        self._acursor = pos
        self.root.after(40, self._audio_tick)

    # ----------------------------------------------------------------- pose mode
    @property
    def mode(self) -> str:
        try:
            return "pose" if self.nb.index("current") == 1 else "glitch"
        except (tk.TclError, AttributeError):
            return "glitch"

    def _mode_changed(self, redraw: bool = True):
        if self.mode == "pose":
            self.tl_row_glitch.pack_forget()
            self.tl_row_pose.pack(fill="x", pady=(3, 0), before=self.tl_help)
            self.tl_help.config(text="click/drag: scrub · double-click: add hint · Shift+click: hint on the "
                                     "curve · drag a hint: move it · right-click: delete")
        else:
            self.tl_row_pose.pack_forget()
            self.tl_row_glitch.pack(fill="x", pady=(3, 0), before=self.tl_help)
            self.tl_help.config(text="click/drag: scrub · double-click: add key · drag a key, hit or ▲▼: move "
                                     "it · Shift+click: add hit · right-click: delete")
        if redraw:
            self._draw_timeline()
            self._redraw()

    def _build_pose_tab(self, p):
        pv = self.pvars
        est = self._section(p, "Estimate (estimate_pose.py + MediaPipe)", "pose_estimate", pady=(0, 0))

        def spin(parent, key, label, lo, hi, inc, default, row, cmd=None):
            ttk.Label(parent, text=label, width=26).grid(row=row, column=0, sticky="w")
            v = pv[key] = tk.DoubleVar(value=default)
            sp = ttk.Spinbox(parent, from_=lo, to=hi, increment=inc, textvariable=v, width=8,
                             command=cmd or self._pose_params_changed)
            sp.grid(row=row, column=1, sticky="w", pady=1)
            sp.bind("<KeyRelease>", lambda _e: (cmd or self._pose_params_changed)())
        ttk.Label(est, text="Model size", width=26).grid(row=0, column=0, sticky="w")
        pv["variant"] = tk.StringVar(value="heavy")
        ttk.Combobox(est, textvariable=pv["variant"], values=["heavy", "full", "lite"], width=8,
                     state="readonly").grid(row=0, column=1, sticky="w", pady=1)
        ttk.Label(est, text="Model file (optional)", width=26).grid(row=1, column=0, sticky="w")
        mrow = ttk.Frame(est)
        mrow.grid(row=1, column=1, columnspan=2, sticky="ew")
        pv["model"] = tk.StringVar()
        ttk.Entry(mrow, textvariable=pv["model"], width=18).pack(side="left", fill="x", expand=True)
        ttk.Button(mrow, text="…", width=2, command=self._pose_pick_model).pack(side="left", padx=2)
        spin(est, "focal", "Focal length mm (0 = unknown)", 0, 2000, 1, 0, 2, cmd=lambda: None)
        spin(est, "sensor", "Sensor width mm", 1, 100, 0.1, 36.0, 3, cmd=lambda: None)
        spin(est, "max_speed", "Fastest turn (deg/s)", 30, 3000, 30, 720, 4)
        spin(est, "smooth", "Yaw smoothing (s)", 0, 2, 0.05, 0.1, 5)
        spin(est, "smooth_tilt", "Pitch/roll smoothing (s)", 0, 3, 0.05, 0.25, 6)
        est.columnconfigure(1, weight=1)
        brow = ttk.Frame(est)
        brow.grid(row=7, column=0, columnspan=3, sticky="ew", pady=(6, 2))
        self.pose_btn = ttk.Button(brow, text="Estimate pose", style="Accent.TButton", command=self.pose_estimate)
        self.pose_btn.pack(side="left", fill="x", expand=True)
        ttk.Button(brow, text="Copy command", command=self._pose_copy_cmd).pack(side="left", padx=(4, 0))
        self.pose_prog = ttk.Progressbar(est, mode="determinate")
        self.pose_prog.grid(row=8, column=0, columnspan=3, sticky="ew", pady=2)
        self.pose_msg = ttk.Label(est, text="Writes the pose file shown at the top (or <input>_pose.csv) "
                                  "and a landmark cache beside it.", wraplength=400, justify="left")
        self.pose_msg.grid(row=9, column=0, columnspan=3, sticky="w")

        h = self._section(p, "Hints (keyframes on the timeline)", "pose_hints", pady=(8, 0))
        crow = ttk.Frame(h)
        crow.pack(fill="x")
        ttk.Label(crow, text="Edit").pack(side="left")
        pv["ch"] = tk.StringVar(value="yaw")
        for ch, (lab, _flag, _col) in POSE_CH.items():
            ttk.Radiobutton(crow, text=lab, value=ch, variable=pv["ch"],
                            command=self._pose_ch_changed).pack(side="left", padx=4)
        vrow = ttk.Frame(h)
        vrow.pack(fill="x", pady=(4, 0))
        ttk.Label(vrow, text="Value (deg)").pack(side="left")
        pv["hval"] = tk.DoubleVar(value=0.0)
        ttk.Spinbox(vrow, from_=-720, to=720, increment=1, textvariable=pv["hval"], width=7).pack(side="left", padx=4)
        ttk.Button(vrow, text="Pin playhead to value", command=self._pose_pin_value).pack(side="left")
        self.yaw_row = ttk.Frame(h)
        self.yaw_row.pack(fill="x", pady=(4, 0))
        ttk.Label(self.yaw_row, text="Yaw pins").pack(side="left")
        for lab, val in (("0 front", 0), ("90 right", 90), ("180 back", 180), ("-90 left", -90)):
            ttk.Button(self.yaw_row, text=lab, width=7,
                       command=lambda v=val: self._pose_add_hint(self._playhead_frame(), v, "yaw")
                       ).pack(side="left", padx=1)
        self.pose_now = ttk.Label(h, text="", font=("TkFixedFont", 9))
        self.pose_now.pack(fill="x", pady=(6, 0))
        arow = ttk.Frame(h)
        arow.pack(fill="x", pady=(6, 0))
        pv["auto"] = tk.BooleanVar(value=True)
        ttk.Checkbutton(arow, text="Re-solve when hints change", variable=pv["auto"]).pack(side="left")
        ttk.Button(arow, text="Re-solve now", command=lambda: self.pose_estimate(resolve=True)).pack(side="right")
        drow = ttk.Frame(h)
        drow.pack(fill="x", pady=(4, 0))
        ttk.Button(drow, text="Clear this channel", command=lambda: self._pose_clear_hints(self._pose_ch())
                   ).pack(side="left")
        ttk.Button(drow, text="Clear all hints", command=lambda: self._pose_clear_hints(None)).pack(side="left", padx=4)
        ttk.Label(h, style="Muted.TLabel", wraplength=400, justify="left", text=(
            "Yaw: 0 = facing camera, + = front turns to screen right. A yaw hint pins the solver at that "
            "frame and settles front/back around it - put one wherever the curve is mirrored or the "
            "confidence strip is red.\nPitch (+ = camera above) and roll (+ = clockwise): the curve is "
            "shifted through your hints, the correction blended linearly between them - pitch is often "
            "biased on high angles, so one or two hints fix a whole shot.")).pack(fill="x", pady=(6, 0))
        pv["overlay"] = tk.BooleanVar(value=True)
        ttk.Checkbutton(p, text="Show pose readout on the preview (Pose tab)", variable=pv["overlay"],
                        command=self._redraw).pack(anchor="w", pady=(8, 0))

    # paths / files
    def _pose_ch(self) -> str:
        return self.pvars["ch"].get()

    def _pose_path(self) -> Path | None:
        val = self.vars["pose"].get().strip()
        if val:
            return Path(val)
        inp = self.vars["input"].get().strip()
        if not inp:
            return None
        ip = Path(inp).resolve()
        return ip.with_name((ip.stem if ip.is_file() else ip.name) + "_pose.csv")

    @staticmethod
    def _pose_side(pose: Path, what: str) -> Path:
        return pose.with_name(pose.stem + {"cache": ".landmarks.npz", "hints": ".hints.json"}[what])

    def _pose_pick_model(self):
        f = filedialog.askopenfilename(title="MediaPipe pose model", filetypes=[("Model", "*.task"), ("All", "*.*")])
        if f:
            self.pvars["model"].set(f)

    def _pose_clear_file(self):
        self.vars["pose"].set("")
        self._last_paths["pose"] = ""
        self._pose_file_changed()
        self._settings_changed()

    def _lm_path(self) -> Path | None:
        pose = self.vars["pose"].get().strip()
        return Path(pose).with_name(Path(pose).stem + "_landmarks.npz") if pose else None

    def _parts_status(self):
        lp = self._lm_path()
        if lp is not None and lp.is_file():
            txt, st = f"Landmarks: {lp.name}", "Muted.TLabel"
        elif lp is not None:
            txt, st = ("No landmarks next to this pose file - run Estimate pose (Pose tab) to use "
                       "body parts."), "TLabel"
        else:
            txt, st = "Body parts follow the landmarks that Estimate pose (Pose tab) writes.", "Muted.TLabel"
        self.parts_lbl.configure(text=txt, style=st)

    def _pose_file_changed(self, load_hints: bool = True):
        """The pose path changed: load its curves and the hints saved beside it."""
        self.pose_curves = None
        self._parts_status()
        pose = self.vars["pose"].get().strip()
        if pose and Path(pose).is_file():
            try:
                names, rows = glitch._read_table(pose)
                a = glitch.np.array(rows, float)
                cols = {n: a[:, i] for i, n in enumerate(names) if i < a.shape[1]}
                ok = glitch.np.isfinite(cols["frame"])
                self.pose_curves = {k: cols[k][ok] for k in ("frame", "yaw", "pitch", "roll", "confidence")
                                    if k in cols}
            except Exception as ex:                   # noqa: BLE001
                self.pose_msg.config(text=f"⚠ could not read {pose}: {ex}", foreground=self.pal["err"])
        if load_hints:
            self.pose_hints = {ch: [] for ch in POSE_CH}
            if pose:
                hf = self._pose_side(Path(pose), "hints")
                try:
                    d = json.loads(hf.read_text()) if hf.is_file() else {}
                    for ch in POSE_CH:
                        self.pose_hints[ch] = sorted((float(f), float(v)) for f, v in d.get(ch, []))
                except Exception:                     # noqa: BLE001 - a broken sidecar just means no hints
                    pass
        self._draw_timeline()
        self._redraw()

    def _pose_save_hints(self):
        pose = self.vars["pose"].get().strip()
        if not pose:
            return                                      # written next to the pose file once it exists
        hf = self._pose_side(Path(pose), "hints")
        try:
            if any(self.pose_hints.values()):
                hf.write_text(json.dumps({ch: [list(h) for h in hs] for ch, hs in self.pose_hints.items()}))
            elif hf.exists():
                hf.unlink()
        except OSError:
            pass

    # values
    def _pose_at(self, frame):
        c = self.pose_curves
        if not c or not len(c["frame"]):
            return None
        out = {}
        for k in ("yaw", "pitch", "roll", "confidence"):
            if k in c:
                v = c[k]
                ok = glitch.np.isfinite(v)
                out[k] = float(glitch.np.interp(frame, c["frame"][ok], v[ok])) if ok.any() else 0.0
        return out

    def _pose_round(self, v: float) -> float:
        return float(round(v)) if self._pose_ch() == "yaw" else round(v * 2) / 2

    def _pose_range(self, ch):
        if self._pose_frozen is not None:
            return self._pose_frozen
        vals = [v for _f, v in self.pose_hints[ch]]
        if self.pose_curves and ch in self.pose_curves:
            cv = self.pose_curves[ch]
            cv = cv[glitch.np.isfinite(cv)]
            vals += [float(cv.min()), float(cv.max())] if cv.size else []
        if not vals:
            return (-180.0, 180.0) if ch == "yaw" else (-45.0, 45.0)
        lo, hi = min(vals), max(vals)
        span = max(hi - lo, 90.0 if ch == "yaw" else 30.0) * 1.2
        mid = (lo + hi) / 2
        return mid - span / 2, mid + span / 2

    def _ph_y(self, v, rng=None):
        lo, hi = rng or self._pose_range(self._pose_ch())
        _x0, _x1, t, b = self._tl_geom()
        return b - (b - t) * (v - lo) / (hi - lo)

    def _ph_val(self, y) -> float:
        lo, hi = self._pose_range(self._pose_ch())
        _x0, _x1, t, b = self._tl_geom()
        return lo + (hi - lo) * min(1.0, max(0.0, (b - y) / (b - t)))

    def _ph_item(self, x, y, tol=8):
        best = None
        for j, (f, v) in enumerate(self.pose_hints[self._pose_ch()]):
            d = max(abs(self._tl_x(f) - x), abs(self._ph_y(v) - y))
            if d < tol and (best is None or d < best[0]):
                best = (d, j)
        return None if best is None else best[1]

    def _draw_pose_timeline(self):
        c = self.tl
        x0, x1, t, b = self._tl_geom()
        frs = self._tl_frames()
        n = len(frs)
        ch = self._pose_ch()
        lab, _flag, col = POSE_CH[ch]
        rng = self._pose_range(ch)
        lo, hi = rng
        c.create_rectangle(x0, t, x1, b, outline="#333")
        span = hi - lo
        step = next((s for s in (5, 10, 15, 30, 45, 90, 180, 360) if span / s <= 5), 360)
        g = math.ceil(lo / step) * step
        while g <= hi:
            y = self._ph_y(g, rng)
            c.create_line(x0, y, x1, y, fill="#3a3a3a" if g else "#555", dash=(2, 3) if g else ())
            c.create_text(x0 - 4, y, text=f"{g:g}°", anchor="e", fill="#888", font=("TkDefaultFont", 8))
            g += step
        xs = [x0 + (x1 - x0) * i / max(1, n - 1) for i in range(n)]
        cur = self.pose_curves
        if cur and ch in cur and len(cur["frame"]):
            v = cur[ch]
            ok = glitch.np.isfinite(v)
            vals = glitch.np.interp(frs, cur["frame"][ok], v[ok]) if ok.any() else glitch.np.zeros(n)
            if "confidence" in cur:                        # where the estimate is unsure
                cf = glitch.np.interp(frs, cur["frame"], glitch.np.nan_to_num(cur["confidence"]))
                for i in range(n):
                    if cf[i] < 0.5:
                        xa = xs[i] - (x1 - x0) / max(1, n - 1) / 2
                        xb = xs[i] + (x1 - x0) / max(1, n - 1) / 2
                        c.create_rectangle(max(x0, xa), b - 5, min(x1, xb), b,
                                           fill="#c0392b" if cf[i] < 0.25 else "#b9770e", outline="")
            pts = [q for x, vv in zip(xs, vals) for q in (x, self._ph_y(float(vv), rng))]
            if n == 1:
                pts += [x1, pts[1]]
            c.create_line(*pts, fill=col, width=2)
        else:
            c.create_text((x0 + x1) / 2, (t + b) / 2, fill="#777",
                          text=(f"the pose file has no {ch} column" if cur else
                                "no pose yet - run Estimate pose (hints you add now are used by it)"))
        for ch2 in POSE_CH:                                 # other channels' hints: ticks on top
            if ch2 != ch:
                for f, _v in self.pose_hints[ch2]:
                    x = self._tl_x(f)
                    c.create_line(x, t, x, t + 5, fill=POSE_CH[ch2][2], width=2)
        for f, v in self.pose_hints[ch]:
            x, y = self._tl_x(f), self._ph_y(v, rng)
            c.create_polygon(x, y - 6, x + 6, y, x, y + 6, x - 6, y, fill="#ffd166", outline="#000")
        c.create_text(x1 - 4, t + 3, text=lab + (" (re-solving…)" if self.pose_proc else ""),
                      anchor="ne", fill=col, font=("TkDefaultFont", 9, "bold"))
        c.create_text(x0, b + 2, text=str(frs[0]), anchor="nw", fill="#888", font=("TkDefaultFont", 8))
        c.create_text(x1, b + 2, text=str(frs[-1]), anchor="ne", fill="#888", font=("TkDefaultFont", 8))
        self._draw_playhead()

    def _pose_show_now(self):
        fr = self._playhead_frame()
        now = self._pose_at(fr) if fr is not None else None
        hint = None
        if fr is not None:
            hint = ", ".join(f"{ch} {v:g}" for ch in POSE_CH for f, v in self.pose_hints[ch] if f == fr)
        txt = (f"frame {fr}: " + ("  ".join(f"{k if k != 'confidence' else 'conf'} {now[k]:.1f}"
                                            if k != "confidence" else f"conf {now[k]:.2f}"
                                            for k in ("yaw", "pitch", "roll", "confidence") if k in now)
                                  if now else "no pose")) if fr is not None else ""
        if hint:
            txt += f"\nhints here: {hint}"
        self.pose_now.config(text=txt)
        if self.mode == "pose":
            n = sum(len(v) for v in self.pose_hints.values())
            self.tl_pose_lbl.config(text=f"{n} hint(s): " + ", ".join(
                f"{POSE_CH[ch][0].lower()} {len(v)}" for ch, v in self.pose_hints.items()))

    # editing
    def _pose_add_hint(self, frame, value, ch=None):
        if frame is None:
            return
        ch = ch or self._pose_ch()
        if ch != self._pose_ch():
            self.pvars["ch"].set(ch)
        v = float(round(value)) if ch == "yaw" else round(float(value) * 2) / 2
        self.pose_hints[ch] = sorted([h for h in self.pose_hints[ch] if h[0] != frame] + [(frame, v)])
        self._pose_hints_changed()

    def _pose_hint_here(self):
        fr = self._playhead_frame()
        now = self._pose_at(fr) if fr is not None else None
        self._pose_add_hint(fr, now[self._pose_ch()] if now else 0.0)

    def _pose_pin_value(self):
        try:
            v = float(self.pvars["hval"].get())
        except (tk.TclError, ValueError):
            return
        self._pose_add_hint(self._playhead_frame(), v)

    def _pose_clear_hints(self, ch):
        chans = [ch] if ch else list(POSE_CH)
        if any(self.pose_hints[c] for c in chans) and messagebox.askyesno(
                "Clear hints", f"Remove all {'hints' if ch is None else POSE_CH[ch][0].lower() + ' hints'}?"):
            for c in chans:
                self.pose_hints[c] = []
            self._pose_hints_changed()

    def _pose_ch_changed(self):
        self._pose_frozen = None
        self._draw_timeline()

    def _pose_hints_changed(self):
        self._undo_mark()
        self._pose_save_hints()
        self._draw_timeline()
        self._pose_show_now()
        self._pose_schedule()

    def _pose_params_changed(self):
        self._pose_schedule()

    def _pose_schedule(self):
        """Re-solve shortly after the last edit - only when the landmarks are cached (instant)."""
        if self._pose_debounce is not None:
            self.root.after_cancel(self._pose_debounce)
            self._pose_debounce = None
        pose = self._pose_path()
        if not self.pvars["auto"].get() or pose is None or not self._pose_side(pose, "cache").is_file():
            return
        self._pose_debounce = self.root.after(350, lambda: self.pose_estimate(resolve=True))

    # running estimate_pose.py
    def _pose_state(self) -> str:
        return json.dumps([self.pose_hints, {k: v.get() for k, v in self.pvars.items()
                                             if k in ("max_speed", "smooth", "smooth_tilt", "focal",
                                                      "sensor", "variant", "model")}])

    def _pose_cmd(self, out: Path, pose: Path) -> list[str]:
        pv = self.pvars
        g = lambda k: float(pv[k].get())                     # noqa: E731
        est = ([sys.executable, "--estimate-pose"] if getattr(sys, "frozen", False)    # packaged app
               else [sys.executable, str(HERE / "estimate_pose.py")])
        a = est + [self.vars["input"].get().strip(),
             "-o", str(out), "--cache", str(self._pose_side(pose, "cache")),
             "--landmarks-out", str(pose.with_name(pose.stem + "_landmarks.npz")),
             "--fps", f"{float(self.vars['fps'].get() or 25):g}", "--variant", pv["variant"].get()]
        if pv["model"].get().strip():
            a += ["--model", pv["model"].get().strip()]
        a += ["--max-speed", f"{g('max_speed'):g}", "--smooth", f"{g('smooth'):g}",
              "--smooth-tilt", f"{g('smooth_tilt'):g}"]
        if float(pv["focal"].get() or 0) > 0:
            a += ["--focal-mm", f"{float(pv['focal'].get()):g}", "--sensor-mm", f"{float(pv['sensor'].get()):g}"]
        for key, default in (("exr_space", "linear"), ("exr_alpha", "premultiplied")):
            if self.vars[key].get() != default:
                a += ["--" + key.replace("_", "-"), self.vars[key].get()]
        for ch, (_lab, flag, _c) in POSE_CH.items():
            for f, v in self.pose_hints[ch]:
                a += [flag, f"{int(f)}:{v:g}"]
        return a

    def _pose_copy_cmd(self):
        try:
            pose = self._pose_path()
            if pose is None:
                raise glitch.PlanError("no input chosen")
            argv = ["python", "estimate_pose.py"] + self._pose_cmd(pose, pose)[2:]
        except Exception as ex:                       # noqa: BLE001
            self.set_status(f"⚠ {ex}", error=True)
            return
        cmd = subprocess.list2cmdline(argv) if os.name == "nt" else shlex.join(argv)
        self.root.clipboard_clear()
        self.root.clipboard_append(cmd)
        self.set_status("Copied: " + cmd)

    def pose_estimate(self, resolve: bool = False):
        """Run estimate_pose.py (button: start / cancel). resolve=True: an automatic re-solve."""
        self._pose_debounce = None
        if self.pose_proc is not None:
            if resolve:                                   # finish the current run, then re-solve
                return
            self._pose_cancelled = True
            try:
                self.pose_proc.terminate()
            except OSError:
                pass
            return
        pose = self._pose_path()
        if pose is None or not self.scan_done:
            self.pose_msg.config(text="Pick input frames first.", foreground=self.pal["err"])
            return
        try:
            for k in ("max_speed", "smooth", "smooth_tilt", "focal", "sensor"):
                float(self.pvars[k].get())
        except (tk.TclError, ValueError):
            self.pose_msg.config(text="⚠ check the number fields", foreground=self.pal["err"])
            return
        if resolve and not self._pose_side(pose, "cache").is_file():
            return
        pose.parent.mkdir(parents=True, exist_ok=True)
        tmp = pose.with_name(pose.stem + ".tmp.csv")          # the old pose file stays until success
        cmd = self._pose_cmd(tmp, pose) + ["--progress"]
        self._pose_job = (pose, tmp, self._pose_state(), resolve)
        self._pose_cancelled = False
        try:
            proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, text=True,
                                    bufsize=1, encoding="utf-8", errors="replace",
                                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        except OSError as ex:
            self.pose_msg.config(text=f"⚠ could not start estimate_pose.py: {ex}", foreground=self.pal["err"])
            return
        self.pose_proc = proc
        self.pose_btn.config(text="Cancel")
        self.pose_prog.config(value=0, maximum=max(1, len(self._tl_frames())))
        self.pose_msg.config(text="Re-solving with hints…" if resolve else "Starting (loading MediaPipe)…",
                             foreground="")
        self._draw_timeline()

        def pump():                                       # background: no Tk calls here
            lines = []
            for ln in proc.stdout:
                ln = ln.strip()
                if ln:
                    lines.append(ln)
                    self.q.put(("pose_line", proc, ln))
            rc = proc.wait()
            self.q.put(("pose_done", proc, rc, lines[-6:]))
        threading.Thread(target=pump, daemon=True).start()

    def _pose_line(self, ln: str):
        m = re.match(r"landmarks (\d+)/(\d+)", ln)
        if m:
            i, n = int(m.group(1)), int(m.group(2))
            self.pose_prog.config(value=i, maximum=n)
            self.pose_msg.config(text=f"Finding landmarks: frame {i}/{n}", foreground="")
        elif not ln.startswith(("W0", "I0", "INFO", "WARNING: All log")):   # MediaPipe log noise
            self.pose_msg.config(text=ln[:300], foreground="")

    def _pose_done(self, rc: int, tail):
        pose, tmp, state, resolve = self._pose_job
        self.pose_proc = None
        self.pose_btn.config(text="Estimate pose")
        if self._pose_cancelled:
            tmp.unlink(missing_ok=True)
            self.pose_msg.config(text="Cancelled.", foreground="")
            self._draw_timeline()
            return
        if rc != 0 or not tmp.is_file():
            tmp.unlink(missing_ok=True)
            err = "\n".join(l for l in tail if not l.startswith(("landmarks ", "W0", "I0")))[-600:]
            self.pose_msg.config(text=f"⚠ estimate_pose.py failed:\n{err}", foreground=self.pal["err"])
            if not resolve:
                messagebox.showerror("Pose estimation failed", err or f"exit code {rc}")
            self._draw_timeline()
            return
        os.replace(tmp, pose)
        self.pose_prog.config(maximum=1, value=1)
        summary = next((l for l in reversed(tail) if "frames ->" in l), "done")
        self.pose_msg.config(text=summary.replace(str(tmp), pose.name), foreground="")
        if self.vars["pose"].get().strip() != str(pose):
            self.vars["pose"].set(native_path(pose))
            self._last_paths["pose"] = str(pose)
            if float(self.vars["follow"].get()) == 0.0:
                self.vars["follow"].set(0.4)              # 3D volume should travel with the subject
        self._pose_save_hints()
        self._pose_file_changed(load_hints=False)
        self._settings_changed()                          # pose file stamp changed -> re-plan + preview
        if self._pose_state() != state:                   # edited while it ran: solve again
            self._pose_schedule() if self.pvars["auto"].get() else None

    def _curve_args(self, with_output: bool) -> list[str]:
        """Write the keyframes where the command can find them; hits go on the command line."""
        a = []
        if self.tl_keys or self.tl_extra:
            out = (self.vars["output"].get().strip() if with_output else "")
            path = Path(out).with_name(Path(out).name + "_timing.csv") if out else self._tmp_curve
            path.parent.mkdir(parents=True, exist_ok=True)
            keys = dict(self.tl_extra)
            if self.tl_keys:
                keys["intensity"] = self.tl_keys
            glitch.Timing(keys, [], self._timing().fps).to_csv(path)
            a += ["--curve", str(path)]
        if self.tl_hits:
            a += ["--hits", ",".join(f"{f:g}:{st:g}" + (f":{sec:g}" if sec else "")
                                     for f, st, sec in self.tl_hits)]
        if abs(self._hit_len() - 0.5) > 1e-9:
            a += ["--hit-len", f"{self._hit_len():g}"]
        for kind, flag in (("in", "--appear"), ("out", "--vanish")):
            ts = [x for x in self.tl_trans if x[1] == kind]
            if ts:
                a += [flag, ",".join(f"{f:g}" + (f":{sec:g}" if sec else "") for f, _k, sec in ts)]
        if self.tl_trans and abs(self._trans_len() - 1.0) > 1e-9:
            a += ["--trans-len", f"{self._trans_len():g}"]
        return a

    # ----------------------------------------------------------------- undo / redo
    def _snapshot(self) -> str:
        s = {k: v for k, v in self.settings().items() if k not in VIEW_KEYS}
        s["pose_hints"] = {ch: [list(h) for h in hs] for ch, hs in self.pose_hints.items()}
        return json.dumps(s, sort_keys=True)

    def _undo_mark(self):
        """Something changed: record it once things settle (a slider drag = one step)."""
        if getattr(self, "_restoring", True) or not hasattr(self, "undo_btn"):
            return                                          # (still building the window)
        if self._undo_id is not None:
            self.root.after_cancel(self._undo_id)
        self._undo_id = self.root.after(UNDO_SETTLE_MS, self._undo_commit)

    def _undo_commit(self):
        self._undo_id = None
        try:
            snap = self._snapshot()
        except (tk.TclError, ValueError):
            return
        if self._undo_i >= 0 and self._undo[self._undo_i] == snap:
            return
        del self._undo[self._undo_i + 1:]                   # a new edit drops the redo branch
        self._undo.append(snap)
        if len(self._undo) > UNDO_DEPTH:
            del self._undo[0]
        self._undo_i = len(self._undo) - 1
        self._undo_buttons()

    def _undo_buttons(self):
        self.undo_btn.configure(state="normal" if self._undo_i > 0 else "disabled")
        self.redo_btn.configure(state="normal" if self._undo_i < len(self._undo) - 1 else "disabled")

    @staticmethod
    def _undo_what(a: str, b: str) -> str:
        """Names of what differs between two snapshots, for the status bar."""
        da, db = json.loads(a), json.loads(b)
        names = {"timing": "timeline", "pose_hints": "pose hints"}
        diff = [names.get(k, k.replace("_", " ")) for k in sorted(set(da) | set(db)) if da.get(k) != db.get(k)]
        return ", ".join(diff[:4]) + (f" (+{len(diff) - 4} more)" if len(diff) > 4 else "") or "nothing"

    def undo(self):
        self._undo_step(-1)

    def redo(self):
        self._undo_step(+1)

    def _undo_step(self, d: int):
        if self._undo_id is not None:                       # an edit still settling counts first
            self.root.after_cancel(self._undo_id)
            self._undo_commit()
        j = self._undo_i + d
        if not 0 <= j < len(self._undo):
            self.set_status("Nothing to " + ("undo" if d < 0 else "redo"), note=True)
            return
        what = self._undo_what(self._undo[self._undo_i], self._undo[j])
        snap = json.loads(self._undo[j])
        self._restoring = True
        try:
            hints = snap.pop("pose_hints", None)
            self.apply_settings(snap)
            if hints is not None:
                self.pose_hints = {ch: sorted(tuple(h) for h in hints.get(ch, [])) for ch in POSE_CH}
                self._pose_save_hints()
                self._pose_schedule()
            self._tl_act = (None, None)
            self._draw_timeline()
            self._audio_file_changed() if snap.get("audio", "") != (self.audio or {}).get("path", "") else None
            self._settings_changed()
        finally:
            self._restoring = False
        self._undo_i = j
        self._undo_buttons()
        self.set_status(("Undid: " if d < 0 else "Redid: ") + what
                        + f"   (step {self._undo_i} of {len(self._undo) - 1})", note=True)

    # ----------------------------------------------------------------- sections
    def _section(self, parent, title, key, pady=(0, 6), side=None, padding=6):
        """A titled section whose header folds it away (state remembered between sessions).
        Returns the body frame to fill."""
        outer = ttk.Frame(parent)
        outer.pack(fill="x", pady=pady, **({"side": side} if side else {}))
        body = ttk.Frame(outer, padding=padding)
        folded = key in self._collapsed

        def toggle():
            if body.winfo_manager():
                body.pack_forget()
                self._collapsed.add(key)
            else:
                body.pack(fill="x")
                self._collapsed.discard(key)
            hdr.configure(text=("▸ " if key in self._collapsed else "▾ ") + title)
            _save_prefs(collapsed=sorted(self._collapsed))
            self.root.update_idletasks()                     # let the new layout settle first...
            for f in getattr(self, "_refits", []):          # ...then fit the scroll areas to it
                f()
        hdr = ttk.Button(outer, text=("▸ " if folded else "▾ ") + title, style="Section.TButton",
                         command=toggle)
        hdr.pack(fill="x")
        if not folded:
            body.pack(fill="x")
        self._section_toggles[key] = toggle
        return body

    def toggle_panel(self):
        """Hide / show the whole side panel (Ctrl+B): the preview gets the full width."""
        if str(self.nb) in self.body.panes():
            self.body.forget(self.nb)
            self.panel_btn.configure(text="▶ Panel")
        else:
            self.body.insert(0, self.nb, weight=0)
            self.panel_btn.configure(text="◀ Panel")

    # ----------------------------------------------------------------- look
    def _style_ttk(self):
        """Flat, low-contrast ttk styling on top of 'clam' (the one built-in theme that takes
        colours everywhere, on Windows, macOS and Linux alike)."""
        P = self.pal
        st = ttk.Style(self.root)
        try:
            st.theme_use("clam")
        except tk.TclError:
            pass
        base = tkfont.nametofont("TkDefaultFont")
        if sys.platform.startswith("win"):
            for name in ("TkDefaultFont", "TkTextFont", "TkMenuFont", "TkHeadingFont"):
                tkfont.nametofont(name).configure(family="Segoe UI", size=9)
        bold = base.copy()
        bold.configure(weight="bold")
        self._bold = bold
        st.configure(".", background=P["bg"], foreground=P["fg"], fieldbackground=P["field"],
                     bordercolor=P["border"], darkcolor=P["bg"], lightcolor=P["bg"],
                     troughcolor=P["trough"], focuscolor=P["accent"], selectbackground=P["accent"],
                     selectforeground=P["accent_fg"], insertcolor=P["fg"], arrowcolor=P["muted"])
        st.map(".", foreground=[("disabled", P["muted"])])
        st.configure("TFrame", background=P["bg"])
        st.configure("TLabel", background=P["bg"], foreground=P["fg"])
        st.configure("Muted.TLabel", foreground=P["muted"])
        st.configure("Section.TButton", background=P["panel"], foreground=P["fg"], anchor="w",
                     font=bold, padding=(8, 4), relief="flat", bordercolor=P["panel"],
                     lightcolor=P["panel"], darkcolor=P["panel"])
        st.map("Section.TButton", background=[("active", P["hover"])], lightcolor=[("active", P["hover"])],
               darkcolor=[("active", P["hover"])], bordercolor=[("focus", P["panel"])])
        st.configure("Status.TLabel", background=P["panel"], foreground=P["muted"])
        st.configure("TLabelframe", background=P["bg"], bordercolor=P["border"], relief="solid",
                     borderwidth=1, lightcolor=P["bg"], darkcolor=P["bg"])
        st.configure("TLabelframe.Label", background=P["bg"], foreground=P["muted"], font=bold)
        for name, bgc, hov, prs, fgc in (("TButton", P["btn"], P["hover"], P["press"], P["fg"]),
                                         ("Accent.TButton", P["accent"], P["accent_hover"], P["accent"],
                                          P["accent_fg"]),
                                         ("Ghost.TButton", P["bg"], P["hover"], P["press"], P["muted"])):
            st.configure(name, background=bgc, foreground=fgc, bordercolor=bgc, lightcolor=bgc,
                         darkcolor=bgc, relief="flat", padding=(10, 4), focusthickness=0)
            st.map(name, background=[("disabled", P["panel"]), ("pressed", prs), ("active", hov)],
                   bordercolor=[("focus", P["accent"]), ("active", hov)],
                   lightcolor=[("active", hov)], darkcolor=[("active", hov)],
                   foreground=[("disabled", P["muted"])])
        for name in ("TCheckbutton", "TRadiobutton"):
            st.configure(name, background=P["bg"], foreground=P["fg"], indicatorbackground=P["field"],
                         indicatorforeground=P["accent_fg"], upperbordercolor=P["border"],
                         lowerbordercolor=P["border"], focusthickness=0)
            st.map(name, background=[("active", P["bg"])],
                   indicatorbackground=[("selected", P["accent"]), ("active", P["hover"])])
        for name in ("TEntry", "TSpinbox", "TCombobox"):
            st.configure(name, fieldbackground=P["field"], foreground=P["fg"], bordercolor=P["border"],
                         lightcolor=P["field"], darkcolor=P["field"], background=P["btn"],
                         arrowcolor=P["fg"], insertcolor=P["fg"], padding=(6, 3))
            st.map(name, bordercolor=[("focus", P["accent"])], lightcolor=[("focus", P["field"])],
                   fieldbackground=[("readonly", P["field"]), ("disabled", P["panel"])],
                   selectbackground=[("readonly", P["field"])], selectforeground=[("readonly", P["fg"])],
                   background=[("active", P["hover"])])
        st.configure("TNotebook", background=P["bg"], bordercolor=P["border"], tabmargins=(2, 4, 2, 0),
                     lightcolor=P["bg"], darkcolor=P["bg"])
        st.configure("TNotebook.Tab", background=P["panel"], foreground=P["muted"], padding=(16, 5),
                     bordercolor=P["border"], lightcolor=P["panel"], darkcolor=P["panel"])
        st.map("TNotebook.Tab", background=[("selected", P["bg"]), ("active", P["hover"])],
               foreground=[("selected", P["fg"])], lightcolor=[("selected", P["bg"])],
               expand=[("selected", (1, 1, 1, 0))])
        st.configure("TScrollbar", background=P["btn"], troughcolor=P["bg"], bordercolor=P["bg"],
                     lightcolor=P["btn"], darkcolor=P["btn"], arrowcolor=P["muted"], gripcount=0)
        st.map("TScrollbar", background=[("active", P["hover"])])
        st.configure("TProgressbar", background=P["accent"], troughcolor=P["trough"], bordercolor=P["border"],
                     lightcolor=P["accent"], darkcolor=P["accent"], thickness=6)
        st.configure("Horizontal.TScale", background=P["accent"], troughcolor=P["trough"],
                     bordercolor=P["bg"], lightcolor=P["accent"], darkcolor=P["accent"], gripcount=0,
                     sliderthickness=12)
        st.map("Horizontal.TScale", background=[("active", P["accent_hover"])],
               lightcolor=[("active", P["accent_hover"])], darkcolor=[("active", P["accent_hover"])])
        st.configure("TPanedwindow", background=P["border"])
        st.configure("Sash", sashthickness=5, gripcount=0, background=P["bg"])
        for opt, val in (("*TCombobox*Listbox.background", P["field"]),
                         ("*TCombobox*Listbox.foreground", P["fg"]),
                         ("*TCombobox*Listbox.selectBackground", P["accent"]),
                         ("*TCombobox*Listbox.selectForeground", P["accent_fg"])):
            self.root.option_add(opt, val)
        self.root.configure(background=P["bg"])

    def _recolor_tk(self, w=None):
        """Classic Tk widgets (scroll canvases, combobox lists) don't follow ttk styles."""
        P = self.pal
        w = w or self.root
        for c in w.winfo_children():
            if isinstance(c, tk.Canvas) and c not in (self.view, getattr(self, "tl", None)):
                c.configure(background=P["bg"])
            elif isinstance(c, ttk.Combobox):
                try:
                    pop = c.tk.call("ttk::combobox::PopdownWindow", c)
                    c.tk.call(f"{pop}.f.l", "configure", "-background", P["field"], "-foreground", P["fg"],
                              "-selectbackground", P["accent"], "-selectforeground", P["accent_fg"])
                except tk.TclError:
                    pass
            self._recolor_tk(c)

    def _apply_theme(self, name: str, save: bool = True):
        self.theme, self.pal = name, THEMES[name]
        self._style_ttk()
        self._recolor_tk()
        self.view.configure(background=self.pal["stage"])
        self.theme_btn.configure(text="☀ Light" if name == "dark" else "☾ Dark")
        if save:
            _save_prefs(theme=name)
            self._redraw()

    def toggle_theme(self):
        self._apply_theme("light" if self.theme == "dark" else "dark")

    def _slider(self, parent, key, label, lo, hi, step, default):
        row = ttk.Frame(parent)
        row.pack(fill="x")
        ttk.Label(row, text=label, width=24).pack(side="left")
        v = self.vars[key] = tk.DoubleVar(value=default)
        self._scale(row, v, lo, hi, step, self._settings_changed).pack(side="left", fill="x", expand=True)

    def _scale(self, parent, var, lo, hi, step, cmd, show=True):
        """Themed slider snapped to `step`, with a small value readout on its right. Returns the
        ttk.Scale (the readout is packed next to it when the scale is packed)."""
        dec = max(0, -int(math.floor(math.log10(step)))) if step < 1 else 0
        sc = ttk.Scale(parent, from_=lo, to=hi, orient="horizontal", variable=var, length=120)

        def moved(v):
            snapped = round(round(float(v) / step) * step, dec)
            if dec == 0 and step >= 1:
                snapped = int(snapped)
            if abs(float(var.get()) - snapped) > 1e-9:
                var.set(snapped)
            cmd()
        sc.configure(command=moved)
        if show:
            lab = ttk.Label(parent, width=5, anchor="e", style="Muted.TLabel")

            def upd(*_):
                try:
                    lab.configure(text=f"{float(var.get()):.{dec}f}")
                except (tk.TclError, ValueError):
                    pass
            var.trace_add("write", upd)
            upd()
            lab.pack(side="right", padx=(4, 0))
        return sc

    # ----------------------------------------------------------------- settings <-> argv
    def settings(self) -> dict:
        d = {}
        for k, v in self.vars.items():
            if k in ("frame",):
                continue
            try:
                d[k] = v.get()
            except tk.TclError:          # half-typed number in a spinbox
                d[k] = None
        d["timing"] = self.timing_state()
        return d

    def apply_settings(self, d: dict):
        self._missing_paths = []
        paths = {k: native_path(d[k]) for k in PATH_KEYS if k in d}
        d = {k: v for k, v in d.items() if k not in PATH_KEYS}
        if "timing" in d:
            self._set_timing_state(d["timing"] or {})
            self._tl_act = (None, None)
            self._draw_timeline()
        bad = []
        for k, val in d.items():
            if k in self.vars and val is not None:
                var = self.vars[k]
                try:                                           # keep each variable's type
                    if isinstance(var, tk.BooleanVar):
                        val = bool(val)
                    elif isinstance(var, tk.IntVar):
                        val = int(val)
                    elif isinstance(var, tk.DoubleVar):
                        val = float(val)
                    else:
                        val = str(val)
                    var.set(val)
                except (TypeError, ValueError, tk.TclError):
                    bad.append(k)
        self._bad_keys = bad
        if paths:
            self._apply_paths(paths)

    def _apply_paths(self, paths: dict):
        """Paths from a preset. Ones that don't exist here (another machine, moved files) are
        skipped and named in the status bar; the output folder may not exist yet, that's fine."""
        missing = []
        for k in ("output", "pose", "plate", "audio"):
            if k not in paths:
                continue
            v = paths[k]
            if v and k != "output" and not Path(v).exists():
                missing.append(v)
                continue
            if self.vars[k].get().strip() != v:
                self.vars[k].set(v)
                self._last_paths[k] = v
                if k == "pose":
                    self._pose_file_changed()
                elif k == "audio":
                    self._audio_file_changed()
        if paths.get("output"):
            self._auto_output = None                           # a chosen folder: keep it
        inp = paths.get("input")
        if inp and not Path(inp).exists():
            missing.append(inp)
        elif inp and inp != self.vars["input"].get().strip():
            self.vars["input"].set(inp)
            self._input_changed()
        self._missing_paths = missing

    def argv(self, with_output: bool = True) -> list[str]:
        s = self.settings()
        if not s.get("input"):
            raise glitch.PlanError("no input chosen")
        a = [s["input"]]
        if with_output and s.get("output"):
            a += ["-o", s["output"]]
        a += ["-s", str(int(s["seed"] or 0)), "--fps", f"{float(s['fps'] or 25):g}"]
        for key, _l, _lo, _hi, _st, _d, flag in GLOBALS:
            a += [flag, f"{float(s[key]):g}"]
        if s.get("jpeg_quality"):
            a += ["--jpeg-quality", str(int(s["jpeg_quality"]))]
        skip = [fx for fx in glitch.EFFECT_ORDER if not s[f"on_{fx}"]]
        if skip:
            a += ["--skip", ",".join(skip)]
        gains = [f"{fx}={float(s[f'gain_{fx}']):g}" for fx in glitch.EFFECT_ORDER
                 if s[f"on_{fx}"] and abs(float(s[f"gain_{fx}"]) - 1.0) > 1e-9]
        if gains:
            a += ["--gain", ",".join(gains)]
        if s.get("pose"):
            a += ["--pose", s["pose"]]
        if s.get("plate"):
            a += ["--plate", s["plate"]]
        if s.get("audio") and float(s.get("audio_strength") or 0) > 0:
            a += ["--audio", s["audio"], "--audio-start", f"{float(s.get('audio_start') or 0):g}",
                  "--audio-strength", f"{float(s['audio_strength']):g}",
                  "--audio-sens", f"{float(s.get('audio_sens') or 0):g}"]
            if float(s.get("audio_end") or 0) > 0:
                a += ["--audio-end", f"{float(s['audio_end']):g}"]
        for mode in ("only", "protect"):
            parts = [pt for pt in glitch.PARTS if s.get(f"{mode}_{pt}")]
            if parts:
                a += ["--" + mode, ",".join(parts)]
        for fx in glitch.EFFECT_ORDER:
            tg = s.get(f"target_{fx}", "all")
            if s[f"on_{fx}"] and tg and tg != "all":
                a += ["--target", f"{fx}={tg}"]
        if any(x in a for x in ("--only", "--protect", "--target")) and \
                abs(float(s.get("part_feather") or 1.0) - 1.0) > 1e-9:
            a += ["--part-feather", f"{float(s['part_feather']):g}"]
        if s.get("format", "auto") != "auto":
            a += ["--format", s["format"]]
        for key, default in (("exr_space", "linear"), ("exr_alpha", "premultiplied"),
                             ("exr_depth", "half"), ("exr_compression", "zip")):
            if s.get(key, default) != default:
                a += ["--" + key.replace("_", "-"), s[key]]
        if s.get("input") and Path(s["input"]).is_file():
            a += ["-n", str(int(s.get("frames") or 1))]
        a += self._curve_args(with_output)
        return a

    @staticmethod
    def _plan_from_argv(argv, scan_cache):
        """Pure function (no Tk access), safe to run in a background thread."""
        args = glitch.build_parser().parse_args(argv)
        return glitch.plan_render(args, scan_cache=scan_cache, check_alpha=False)

    def _make_plan(self):
        """Re-plan from the current settings; cheap after the first scan (alpha boxes cached).
        Main thread only (reads Tk variables)."""
        argv = self.argv(with_output=False)
        key = self._key(argv)
        if key == self.plan_key and self.plan is not None:
            return self.plan, key
        plan = self._plan_from_argv(argv, self.scan_cache)
        self.plan, self.plan_key = plan, key
        self._draw_timeline()
        self._draw_wave()
        au = plan.get("audio")
        self.audio_beats_lbl.configure(text="" if au is None else
                                       f"{len(au['beats'])} beat(s) in the passage; the shot uses "
                                       f"{min(self._shot_seconds(), self._audio_passage()[1] - self._audio_passage()[0]):.1f} s of it")
        return plan, key

    def _key(self, argv) -> str:
        """Plan identity: the argv plus the pose file's size/mtime (re-run estimate_pose.py and
        the new curve is picked up even though the path is the same)."""
        stamp = None
        pose = self.vars["pose"].get().strip()
        if pose and Path(pose).is_file():
            st = Path(pose).stat()
            stamp = [st.st_size, st.st_mtime_ns]
            lp = self._lm_path()
            if lp is not None and lp.is_file():
                lst = lp.stat()
                stamp += [lst.st_size, lst.st_mtime_ns]
        au = self.vars["audio"].get().strip()
        if au and Path(au).is_file():                     # re-exported track, same name: re-analyse
            ast_ = Path(au).stat()
            stamp = (stamp or []) + [ast_.st_size, ast_.st_mtime_ns]
        return json.dumps([[x for x in argv if x != str(self._tmp_curve)], stamp, self.timing_state()])

    def pscale(self) -> float:
        return int(self.vars["pscale"].get().rstrip("%")) / 100.0

    # ----------------------------------------------------------------- events
    def _pick_input_dir(self):
        d = filedialog.askdirectory(title="Folder of RGBA PNG frames")
        if d:
            self.vars["input"].set(native_path(d))
            self._input_changed()

    def _pick_input_file(self):
        f = filedialog.askopenfilename(title="Single RGBA image",
                                       filetypes=[("Images", "*.png *.exr *.webp *.tif *.tiff"), ("All", "*.*")])
        if f:
            self.vars["input"].set(native_path(f))
            self._input_changed()

    def _pick_output(self):
        d = filedialog.askdirectory(title="Output folder")
        if d:
            self.vars["output"].set(native_path(d))

    def _pick_pose(self):
        f = filedialog.askopenfilename(title="Pose file (estimate_pose.py output)",
                                       filetypes=[("CSV / text", "*.csv *.txt"), ("All", "*.*")])
        if f:
            self.vars["pose"].set(native_path(f))
            self._last_paths["pose"] = f
            self._pose_file_changed()
            if float(self.vars["follow"].get()) == 0.0:
                self.vars["follow"].set(0.4)          # 3D volume should travel with the subject
            self._settings_changed()

    def _path_changed(self, key):
        raw = self.vars[key].get()
        val = native_path(raw)                                 # typed / pasted: tidy the separators
        if val != raw:
            self.vars[key].set(val)
        if val == self._last_paths.get(key):
            return                                      # focus left the box, nothing changed
        if key == "input":
            self._input_changed()
        elif key == "pose":
            self._last_paths["pose"] = val
            self._pose_file_changed()
            self._settings_changed()
        elif key == "plate":
            self._last_paths["plate"] = val
            self._settings_changed()
        elif key == "audio":
            self._last_paths["audio"] = val
            self._audio_file_changed()
            self._settings_changed()

    def _pick_plate(self, folder: bool):
        f = (filedialog.askdirectory(title="Background plate frames") if folder else
             filedialog.askopenfilename(title="Background plate (still)",
                                        filetypes=[("Images", "*.png *.exr *.tif *.tiff *.webp"), ("All", "*.*")]))
        if f:
            self.vars["plate"].set(native_path(f))
            self._path_changed("plate")

    def _input_changed(self):
        inp = self.vars["input"].get().strip()
        if not inp:
            return
        self._last_paths["input"] = inp
        self._stop_play()
        self._cancel_play_jobs()
        out_now = self.vars["output"].get().strip()
        if not out_now or out_now == self._auto_output:     # never overwrite a folder the user chose
            p = Path(inp).resolve()
            self._auto_output = native_path(p.with_name(p.stem + "_glitch") if p.is_file()
                                            else p.with_name(p.name + "_glitch"))
            self.vars["output"].set(self._auto_output)
        self.scan_cache = {}
        self.scan_done = False
        self.plan, self.plan_key = None, None
        self._clear_cache()
        try:
            argv = self.argv(with_output=False)          # read Tk state here, on the main thread
        except Exception as ex:                       # noqa: BLE001
            self.set_status(f"⚠ {ex}", error=True)
            return
        self.set_status("Scanning frames…")
        cache = self.scan_cache
        key = self._key(argv)

        def work():                                     # background: no Tk calls in here
            try:
                plan = self._plan_from_argv(argv, cache)
                self.q.put(("planned", key, plan, cache))
            except SystemExit:                          # argparse error
                self.q.put(("error", "invalid settings"))
            except Exception as ex:                   # noqa: BLE001
                self.q.put(("error", f"{ex}"))
        threading.Thread(target=work, daemon=True).start()

    def _settings_changed(self):
        if self._debounce is not None:
            self.root.after_cancel(self._debounce)
        self._debounce = self.root.after(180, self._refresh)
        self._undo_mark()

    def _refresh(self):
        self._debounce = None
        self._stop_play()
        self._cancel_play_jobs()
        self._clear_cache()
        self.request_live()

    # ----------------------------------------------------------------- cache + job helpers
    def _clear_cache(self):
        self.cache.clear()
        self.cache_bytes = 0

    def _cache_put(self, ck, img):
        if ck in self.cache:
            return
        nbytes = 0 if img is None else img.width * img.height * 4
        self.cache[ck] = img
        self.cache_bytes += nbytes
        while self.cache_bytes > CACHE_BYTES and len(self.cache) > 1:
            _k, old = self.cache.popitem(last=False)              # evict the oldest
            self.cache_bytes -= 0 if old is None else old.width * old.height * 4

    def _cancel_play_jobs(self):
        for fut in self.pending.values():
            fut.cancel()                                          # queued ones never start
        self.pending.clear()

    def _submit(self, which: str, *args):
        """Submit to a preview pool; if a worker died (e.g. out of memory), start a new pool."""
        pool = getattr(self, which)
        try:
            return pool.submit(*args)
        except BrokenProcessPool:
            pool.shutdown(wait=False, cancel_futures=True)
            setattr(self, which, ProcessPoolExecutor(1 if which == "live_pool" else max(1, _cpus() - 1)))
            self.set_status("A preview worker crashed (out of memory?) - restarted it.", error=True)
            return getattr(self, which).submit(*args)

    def _frame_moved(self):
        self._draw_playhead()
        n = len(self.plan["work"]) if self.plan else 0
        i = min(int(self.vars["frame"].get()), max(0, n - 1))
        if n:
            fr = self.plan["work"][i][2]
            self.frame_lbl.config(text=f"frame {fr}  ({i + 1}/{n})")
        if not self.playing:                             # dragging: render only where it stops
            if self._frame_debounce is not None:
                self.root.after_cancel(self._frame_debounce)
            self._frame_debounce = self.root.after(60, self.request_live)

    # ----------------------------------------------------------------- live preview
    def request_live(self):
        self._frame_debounce = None
        if not self.vars["input"].get().strip() or not self.scan_done:
            return                                      # still scanning: 'planned' will call back
        try:
            plan, key = self._make_plan()
        except SystemExit:                              # argparse rejected a value
            self.set_status("⚠ invalid setting (check the number fields)", error=True)
            return
        except Exception as ex:                         # noqa: BLE001
            self.set_status(f"⚠ {ex}", error=True)
            return
        n = len(plan["work"])
        self.frame_scale.config(to=max(0, n - 1))
        i = min(int(self.vars["frame"].get()), n - 1)
        if self.vars["compare"].get():
            img, _raw = glitch.load_frame(plan["work"][i][0], plan["work"][i][7])
            self.show(glitch.to_display8(img))
            return
        sc = self.pscale()
        ck = (key, sc, i)
        if self.cache.get(ck) is not None:
            self.show(self.cache[ck])
            return
        token = ck
        self.live_token = token
        self.set_status(f"Rendering frame {plan['work'][i][2]}…")
        if self.live_future is not None:
            self.live_future.cancel()                   # superseded before it started
        t0 = time.time()
        fut = self.live_future = self._submit("live_pool", preview_job, plan["work"][i], sc)
        fut.add_done_callback(lambda f: self.q.put(("live", token, f, time.time() - t0)))

    # ----------------------------------------------------------------- play
    def _stop_play(self):
        self.playing = False
        if self._audio_sync is not None:
            self._audio_sync = None
            self.player.stop()
        if self._tick_id is not None:
            self.root.after_cancel(self._tick_id)
            self._tick_id = None
        self.play_btn.config(text="▶ Play")

    def _ensure(self, key, sc, i):
        """Make sure frame i is cached or on its way (also re-renders frames evicted from the cache)."""
        ck = (key, sc, i)
        if ck in self.cache or ck in self.pending:
            return
        fut = self.pending[ck] = self._submit("play_pool", preview_job, self.plan["work"][i], sc)
        fut.add_done_callback(lambda f, ck=ck: self.q.put(("cached", ck, f)))

    def toggle_play(self):
        if self.playing:
            self._stop_play()
            return
        if not self.scan_done:
            return
        try:
            plan, key = self._make_plan()
        except SystemExit:
            self.set_status("⚠ invalid setting (check the number fields)", error=True)
            return
        except Exception as ex:                       # noqa: BLE001
            self.set_status(f"⚠ {ex}", error=True)
            return
        n = len(plan["work"])
        self.frame_scale.config(to=max(0, n - 1))
        sc = self.pscale()
        start = min(int(self.vars["frame"].get()), n - 1)
        for i in list(range(start, n)) + list(range(0, start)):    # fill from the playhead on
            self._ensure(key, sc, i)
        self.playing = True
        self.play_btn.config(text="■ Stop")
        self.play_idx = start
        self._audio_sync = None
        if self.audio and self.vars["audio_sync"].get() and float(self.vars["audio_strength"].get() or 0) >= 0:
            st, en = self._audio_passage()
            t = st + start / max(1e-6, n / max(1e-6, self._shot_seconds()))
            if t < en and not self.player.play(self.audio["x"], self.audio["sr"], t, en):
                self._audio_sync = (time.monotonic(), start)
                self.aplay_btn.configure(text="■ Stop")
                self._audio_tick()
        self._tick()

    def _tick(self):
        self._tick_id = None
        if not self.playing or self.plan is None:
            return
        key, sc = self.plan_key, self.pscale()
        n = len(self.plan["work"])
        if self._audio_sync is not None:                   # with audio: the audio clock sets the frame
            try:
                fps_ = float(self.vars["fps"].get() or 25)
            except (tk.TclError, ValueError):
                fps_ = 25.0
            t0, i0 = self._audio_sync
            want = i0 + int((time.monotonic() - t0) * fps_)
            if want >= n:                                  # the shot ended: stop together
                self._stop_play()
                return
            if self.player.position() is None:             # audio ended / failed: picture carries on
                self._audio_sync = None
            elif (key, sc, want) in self.cache or want > self.play_idx:
                self.play_idx = want                        # (not rendered yet: the last frame stays)
        self.play_idx %= n
        ck = (key, sc, self.play_idx)
        if ck in self.cache:
            img = self.cache[ck]
            self.cache.move_to_end(ck)
            self.vars["frame"].set(self.play_idx)
            self._draw_playhead()
            fr = self.plan["work"][self.play_idx][2]
            self.frame_lbl.config(text=f"frame {fr}  ({self.play_idx + 1}/{n})")
            if img is not None:                          # None = that frame failed to render
                self.show(img)
            if self._audio_sync is None:
                self.play_idx = (self.play_idx + 1) % n
        else:
            self._ensure(key, sc, self.play_idx)         # evicted from the cache: render again
        done = sum(1 for i in range(n) if (key, sc, i) in self.cache)
        failed = sum(1 for i in range(n) if self.cache.get((key, sc, i), 0) is None)
        msg = (f"Playing at {int(sc * 100)}% — all {n} frames cached" if done >= n else
               f"Playing at {int(sc * 100)}% — rendered {done}/{n} frames (waits for frames still rendering)")
        self.set_status(msg + (f"; {failed} frame(s) failed" if failed else ""), error=bool(failed))
        try:
            fps = float(self.vars["fps"].get() or 25)
        except (tk.TclError, ValueError):
            fps = 24.0
        self._tick_id = self.root.after(max(10, int(1000 / max(fps, 1))), self._tick)

    # ----------------------------------------------------------------- full render
    def render_all(self):
        if self.render_cancel is not None:           # button acts as Cancel while rendering
            self.render_cancel.set()
            return
        try:
            argv = self.argv(with_output=True)
            if not self.vars["output"].get().strip():
                raise glitch.PlanError("choose an output folder")
            args = glitch.build_parser().parse_args(argv)
            plan = glitch.plan_render(args, scan_cache=self.scan_cache, check_alpha=False)
        except Exception as ex:                       # noqa: BLE001
            messagebox.showerror("Cannot render", str(ex))
            return
        work = plan["work"]
        if Path(work[0][1]).exists() and not messagebox.askyesno(
                "Overwrite?", f"{work[0][1].parent} already contains rendered frames. Overwrite?"):
            return
        cancel = self.render_cancel = threading.Event()
        self.render_btn.config(text="Cancel render")
        self.progress.config(maximum=len(work), value=0)

        self._stop_play()
        self._cancel_play_jobs()                        # don't compete with stale preview jobs

        def run():
            t0 = time.time()
            done = 0
            ex = ProcessPoolExecutor(_cpus())
            try:
                # runs of consecutive frames per task: time effects reuse neighbours and motion
                ch = max(1, min(8, -(-len(work) // (_cpus() * 3))))
                chunks = [work[i:i + ch] for i in range(0, len(work), ch)]
                futs = [ex.submit(glitch._render_chunk, c) for c in chunks]
                for f, c in zip(futs, chunks):
                    while not f.done():
                        if cancel.is_set():
                            ex.shutdown(wait=False, cancel_futures=True)
                            self.q.put(("render_done", done, len(work), time.time() - t0, True, None))
                            return
                        time.sleep(0.05)
                    f.result()                          # raises the frame's error, if any
                    done += len(c)
                    self.q.put(("render_progress", done, len(work)))
                ex.shutdown(wait=True)
                self.q.put(("render_done", done, len(work), time.time() - t0, False, str(work[0][1].parent)))
            except Exception as exc:                  # noqa: BLE001 - stop the rest right away
                ex.shutdown(wait=False, cancel_futures=True)
                self.q.put(("render_done", done, len(work), time.time() - t0, True, f"error: {exc}"))
        threading.Thread(target=run, daemon=True).start()

    # ----------------------------------------------------------------- presets / CLI
    def _preset_json(self) -> str:
        s = {"glitcher_version": glitch.__version__, **self.settings()}
        for k in PATH_KEYS:                                    # project-style: the shot's paths too
            s[k] = portable_path(s.get(k))
        return json.dumps(s, indent=2)

    def _write_preset(self, path: Path, text: str):
        tmp = path.with_name(path.name + ".tmp")              # never leave a half-written preset
        tmp.write_text(text)
        os.replace(tmp, path)
        self._preset_saved = text

    def save_preset(self, ask: bool = False):
        """Button: always asks where. Ctrl+S: saves to the current preset, asks only the first
        time. From the first save on, the preset is also autosaved every 2 minutes."""
        path = self.preset_path
        if ask or path is None:
            hint = self.preset_path or self._preset_loaded
            f = filedialog.asksaveasfilename(
                title="Save preset", defaultextension=".json", filetypes=[("Preset", "*.json")],
                initialdir=str(hint.parent) if hint else None, initialfile=hint.name if hint else "")
            if not f:
                return
            path = Path(f)
        try:
            self._write_preset(path, self._preset_json())
        except OSError as ex:
            self.set_status(f"⚠ could not save preset: {ex}", error=True)
            return
        self.preset_path = path
        self.root.title(f"{TITLE} — {path.name}")
        self.set_status(f"Saved preset {path}  (autosaves every {AUTOSAVE_MS // 60000} min)")
        if self._autosave_id is None:
            self._autosave_id = self.root.after(AUTOSAVE_MS, self._autosave)

    def _autosave(self, final: bool = False):
        self._autosave_id = None
        if self.preset_path is None:
            return
        try:
            text = self._preset_json()
            if text != self._preset_saved:                     # only when something changed
                self._write_preset(self.preset_path, text)
                if not final:
                    self.set_status(f"Autosaved {self.preset_path.name} at {time.strftime('%H:%M')}")
        except (OSError, tk.TclError) as ex:
            if not final:
                self.set_status(f"⚠ autosave failed: {ex}", error=True)
        if not final:
            self._autosave_id = self.root.after(AUTOSAVE_MS, self._autosave)

    def load_preset(self):
        f = filedialog.askopenfilename(title="Load preset", filetypes=[("Preset", "*.json")])
        if f:
            try:
                d = json.loads(Path(f).read_text(encoding="utf-8"))
                if not isinstance(d, dict):
                    raise ValueError("not a preset (expected a JSON object)")
            except (OSError, ValueError) as ex:
                messagebox.showerror("Cannot load preset", f"{f}\n\n{ex}")
                return
            self.apply_settings(d)
            if self._bad_keys:                                 # say so where it can't be missed
                messagebox.showwarning("Preset partly loaded",
                                       "These values had the wrong type and were skipped:\n"
                                       + ", ".join(self._bad_keys))
            self._settings_changed()
            # a loaded preset is not autosaved over until you save once (Ctrl+S suggests its name)
            self._preset_loaded = Path(f)
            if self.preset_path is not None and Path(f) != self.preset_path:
                self.preset_path = None
                if self._autosave_id is not None:
                    self.root.after_cancel(self._autosave_id)
                    self._autosave_id = None
                self.root.title(TITLE)
            miss = getattr(self, "_missing_paths", [])
            if miss:
                self.set_status(f"Loaded preset {f} — not found here, kept current: " + ", ".join(miss),
                                error=True)
            else:
                self.set_status(f"Loaded preset {f}")

    def reset(self):
        for key, _l, _lo, _hi, _st, default, _f in GLOBALS:
            self.vars[key].set(default)     # includes motion (0 = off)
        for fx in glitch.EFFECT_ORDER:
            self.vars[f"on_{fx}"].set(True)
            self.vars[f"gain_{fx}"].set(1.0)
        self.vars["jpeg_quality"].set(0)
        for fx in glitch.EFFECT_ORDER:
            self.vars[f"target_{fx}"].set("all")
        for pt in glitch.PARTS:
            self.vars[f"only_{pt}"].set(False)
            self.vars[f"protect_{pt}"].set(False)
        self.vars["part_feather"].set(1.0)
        self.vars["audio_strength"].set(0.8)
        self.vars["audio_sens"].set(0.5)
        self.vars["hit_len"].set(0.5)
        self.tl_keys, self.tl_hits, self.tl_extra, self.tl_trans = [], [], {}, []
        self.vars["trans_len"].set(1.0)
        self._draw_timeline()
        for key, default in (("format", "auto"), ("exr_space", "linear"), ("exr_alpha", "premultiplied"),
                             ("exr_depth", "half"), ("exr_compression", "zip")):
            self.vars[key].set(default)
        self._settings_changed()

    def cli_command(self) -> str:
        argv = ["python", "glitch.py"] + self.argv(with_output=True)
        return subprocess.list2cmdline(argv) if os.name == "nt" else shlex.join(argv)

    def copy_cli(self):
        try:
            cmd = self.cli_command()
        except Exception as ex:                       # noqa: BLE001
            self.set_status(f"⚠ {ex}", error=True)
            return
        self.root.clipboard_clear()
        self.root.clipboard_append(cmd)
        self.set_status("Copied: " + cmd)

    # ----------------------------------------------------------------- display + message pump
    def show(self, img: Image.Image):
        self.current_img = img
        self._redraw()

    def _redraw(self):
        img = self.current_img
        if img is None:
            return
        cw, ch = max(10, self.view.winfo_width()), max(10, self.view.winfo_height())
        r = min(cw / img.width, ch / img.height)
        size = (max(1, int(img.width * r)), max(1, int(img.height * r)))
        disp = img.resize(size, Image.BILINEAR if r < 1 else Image.NEAREST)
        bgc = BACKGROUNDS[self.vars["bg"].get()]
        if bgc is None:
            if size not in self._bg_cache:
                self._bg_cache = {size: _checker(size)}           # one size at a time
            bg = self._bg_cache[size].copy()
        else:
            bg = Image.new("RGB", size, bgc)
        bg.paste(disp, (0, 0), disp)
        self._photo = ImageTk.PhotoImage(bg)
        self.view.delete("all")
        self.view.create_image(cw // 2, ch // 2, image=self._photo, anchor="center")
        if self.mode == "pose" and self.pvars["overlay"].get():
            self._draw_pose_overlay()

    def _draw_pose_overlay(self):
        """Three small dials to check the estimate against the image:
        yaw   - top view, camera at the bottom, arrow = where the chest points
        pitch - side view, subject standing in the middle, camera on the right (up = camera above)
        roll  - camera view, the torso line as it leans on screen (+ = clockwise)."""
        fr = self._playhead_frame()
        now = self._pose_at(fr) if fr is not None else None
        if not now:
            return
        v, r, cy = self.view, 24, 40
        dials = [k for k in ("yaw", "pitch", "roll") if k in now]
        w = 16 + 66 * len(dials)
        v.create_rectangle(8, 8, 8 + w, 118 if "confidence" in now else 104, fill="#000000",
                           stipple="gray50", outline="")
        font, small = ("TkFixedFont", 9), ("TkDefaultFont", 7)
        for n, k in enumerate(dials):
            cx = 8 + 41 + 66 * n
            col = POSE_CH[k][2]
            ang = math.radians(now[k])
            v.create_oval(cx - r, cy - r, cx + r, cy + r, outline="#666")
            if k == "yaw":
                dx, dy = math.sin(ang), math.cos(ang)            # 0 = facing the camera (down)
                v.create_line(cx - dy * r * .7, cy + dx * r * .7, cx + dy * r * .7, cy - dx * r * .7,
                              fill="#ccc", width=3)                # shoulders
                v.create_line(cx, cy, cx + dx * r, cy + dy * r, fill=col, width=2, arrow="last")
                v.create_rectangle(cx - 3, cy + r - 3, cx + 3, cy + r + 3, fill="#ddd", outline="")
                sub = "top view"
            elif k == "pitch":
                v.create_line(cx - r, cy, cx + r, cy, fill="#444", dash=(2, 2))   # eye level
                v.create_line(cx, cy - r * .7, cx, cy + r * .7, fill="#ccc", width=3)  # subject
                v.create_oval(cx - 3, cy - r * .7 - 6, cx + 3, cy - r * .7, fill="#ccc", outline="")
                px, py = cx + r * math.cos(ang), cy - r * math.sin(ang)  # camera; + = above
                v.create_line(px, py, cx, cy, fill=col, dash=(3, 2))
                v.create_arc(cx - r * .5, cy - r * .5, cx + r * .5, cy + r * .5, start=0,
                             extent=now[k], style="arc", outline=col)
                v.create_rectangle(px - 4, py - 3, px + 4, py + 3, fill=col, outline="#000")
                sub = "side view"
            else:
                v.create_line(cx, cy - r, cx, cy + r, fill="#444", dash=(2, 2))   # upright
                ux, uy = math.sin(ang), -math.cos(ang)           # 'up' turned clockwise by roll
                v.create_line(cx - ux * r * .7, cy - uy * r * .7, cx + ux * r * .7, cy + uy * r * .7,
                              fill=col, width=3)
                hx, hy = cx + ux * r * .7, cy + uy * r * .7
                v.create_oval(hx - 3, hy - 3, hx + 3, hy + 3, fill=col, outline="")
                sub = "camera view"
            v.create_text(cx, cy + r + 3, text=sub, fill="#888", anchor="n", font=small)
            v.create_text(cx, cy + r + 15, text=f"{k} {now[k]:.1f}", fill="#eee", anchor="n", font=font)
        if "confidence" in now:
            v.create_text(8 + w / 2, 104, text=f"confidence {now['confidence']:.2f}", anchor="n",
                          fill="#c0392b" if now["confidence"] < 0.25 else "#aaa", font=small)

    def set_status(self, text: str, error: bool = False, note: bool = False):
        """note=True: a message that stays attached to the next few seconds of routine status
        updates (e.g. 'Undid: seed' would otherwise vanish behind 'Rendering frame …')."""
        if note:
            self._note = (text, time.monotonic() + 4.0)
        elif not error and getattr(self, "_note", None) and time.monotonic() < self._note[1]:
            text = f"{text}   ·   {self._note[0]}"
        self.status.config(text=text, foreground=self.pal["err"] if error else self.pal["muted"])

    def _poll(self):
        try:
            while True:
                msg = self.q.get_nowait()
                kind = msg[0]
                if kind == "planned":
                    _k, key, plan, cache = msg
                    if cache is not self.scan_cache:           # input changed again meanwhile
                        continue
                    self.plan, self.plan_key = plan, key
                    self.scan_done = True
                    n = len(plan["work"])
                    self.frame_scale.config(to=max(0, n - 1))
                    self.vars["frame"].set(min(int(self.vars["frame"].get()), n - 1))
                    miss = getattr(self, "_missing_paths", [])
                    self.set_status(f"{n} frame(s) ready. Adjust settings; the preview updates live."
                                    + (" — preset paths not found here: " + ", ".join(miss) if miss else ""),
                                    error=bool(miss))
                    self._missing_paths = []
                    self._draw_timeline()
                    self._frame_moved()
                elif kind == "error":
                    self.set_status(f"⚠ {msg[1]}", error=True)
                elif kind == "live":
                    _k, token, fut, dt = msg
                    if fut.cancelled():
                        continue
                    try:
                        img = fut.result()
                    except Exception as ex:           # noqa: BLE001
                        if token == self.live_token:
                            self.set_status(f"⚠ preview failed: {ex}", error=True)
                        continue
                    if token[0] == self.plan_key:      # don't keep results of old settings
                        self._cache_put(token, img)
                    if token == self.live_token and not self.playing:
                        self.show(img)
                        self.set_status(f"Frame rendered in {dt:.2f} s at {int(token[1] * 100)}% "
                                        f"({img.width}×{img.height}).")
                elif kind == "cached":
                    _k, ck, fut = msg
                    if self.pending.get(ck) is fut:
                        del self.pending[ck]
                    if ck[0] == self.plan_key and not fut.cancelled():
                        try:
                            self._cache_put(ck, fut.result())
                        except Exception:              # noqa: BLE001 - mark it so Play skips it
                            self._cache_put(ck, None)
                elif kind == "pose_line":
                    if msg[1] is self.pose_proc:
                        self._pose_line(msg[2])
                elif kind == "pose_done":
                    if msg[1] is self.pose_proc:
                        self._pose_done(msg[2], msg[3])
                elif kind == "render_progress":
                    _k, done, n = msg
                    self.progress.config(value=done)
                    self.set_status(f"Rendering {done}/{n}…")
                elif kind == "render_done":
                    _k, done, n, dt, cancelled, info = msg
                    self.render_cancel = None
                    self.render_btn.config(text="Render sequence")
                    if cancelled:
                        self.set_status(f"Render stopped after {done}/{n} frames. {info or ''}",
                                        error=bool(info))
                        if info:
                            messagebox.showerror("Render failed", f"Stopped after {done}/{n} frames.\n\n{info}")
                    else:
                        self.set_status(f"Rendered {n} frames in {dt:.1f} s → {info}")
        except queue.Empty:
            pass
        self.root.after(30, self._poll)

    def close(self):
        if self._autosave_id is not None:
            self.root.after_cancel(self._autosave_id)
        self._autosave(final=True)                             # keep the last changes, if saved before
        self._stop_play()
        self._cancel_play_jobs()
        if self.render_cancel is not None:
            self.render_cancel.set()
        self.player.stop()
        if self.pose_proc is not None:
            try:
                self.pose_proc.terminate()
            except OSError:
                pass
        for pool in (self.live_pool, self.play_pool):
            pool.shutdown(wait=False, cancel_futures=True)
        try:
            shutil.rmtree(self._tmp_dir, ignore_errors=True)
        except OSError:
            pass
        self.root.destroy()


def main():
    import multiprocessing
    multiprocessing.freeze_support()          # packaged app: worker processes start here, not a new GUI
    # A windowed Windows build started without console handles has sys.stdout / sys.stderr = None;
    # the command-line modes below print progress, so give them a sink instead of crashing.
    for name in ("stdout", "stderr"):
        if getattr(sys, name) is None:
            setattr(sys, name, open(os.devnull, "w", encoding="utf-8"))
    # a packaged app has no separate Python: it doubles as the estimator and the glitch CLI
    if len(sys.argv) > 1 and sys.argv[1] == "--estimate-pose":
        import estimate_pose
        raise SystemExit(estimate_pose.main(sys.argv[2:]))
    if len(sys.argv) > 1 and sys.argv[1] == "--glitch":
        raise SystemExit(glitch.main(sys.argv[2:]))
    if len(sys.argv) > 1 and sys.argv[1] in ("--version", "-V"):
        print(f"{glitch.APP_NAME} {glitch.__version__}")
        raise SystemExit(0)
    if len(sys.argv) > 2 and sys.argv[1] == "--self-test":
        raise SystemExit(self_test(sys.argv[2]))
    root = tk.Tk()
    App(root, sys.argv[1] if len(sys.argv) > 1 else None)
    root.mainloop()


def self_test(inp: str, timeout: float = 120.0) -> int:
    """--self-test FRAMES: open the editor on FRAMES, wait for the first preview, close.
    Exit code 0 = the GUI, its preview workers and the renderer all run in this build (used to
    check the packaged app; your preferences file is not touched)."""
    global PREFS_FILE
    PREFS_FILE = Path(tempfile.mkdtemp(prefix="glitcher_selftest_")) / "prefs.json"
    try:
        root = tk.Tk()
        app = App(root, inp)
        t0 = time.time()
        while time.time() - t0 < timeout:
            root.update()
            if app.plan is not None and app.current_img is not None:
                app.close()
                print(f"self-test OK: first preview in {time.time() - t0:.1f} s")
                return 0
            time.sleep(0.02)
        msg = app.status.cget("text")
        app.close()
        print(f"self-test FAILED: no preview after {timeout:.0f} s ({msg})", file=sys.stderr)
        return 1
    finally:
        shutil.rmtree(PREFS_FILE.parent, ignore_errors=True)


if __name__ == "__main__":
    main()
