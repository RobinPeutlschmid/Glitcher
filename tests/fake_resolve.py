"""A stand-in for DaVinci Resolve's scripting API, as the README describes it, so the plugin's
whole round trip can run in tests: rendering from the timeline writes real EXR frames, imports
check the files exist, Fusion comps record nodes and connections.

It proves the plugin's logic and its use of the API as documented; whether Resolve behaves
exactly like this can only be checked in Resolve."""
import json
import os
import re
from pathlib import Path

import glitch


class Obj:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def lua(items):
    """Resolve hands Lua tables to Python as {1.0: a, 2.0: b}."""
    return {float(i + 1): x for i, x in enumerate(items)}


# ------------------------------------------------------------------------- media
class MediaPoolItem:
    def __init__(self, props):
        self.props = dict(props)
        self.name = props.get("Clip Name", "clip")

    def GetClipProperty(self, name=None):
        return dict(self.props) if name is None else self.props.get(name)

    def SetClipProperty(self, k, v):
        self.props[k] = v
        return True

    def SetName(self, n):
        self.name = n
        return True


class Folder:
    def __init__(self, name):
        self.name, self.subs, self.clips = name, [], []

    def GetName(self):
        return self.name

    def GetSubFolderList(self):
        return lua(self.subs)


class MediaPool:
    def __init__(self, timeline):
        self.root = Folder("Master")
        self.current = self.root
        self.timeline = timeline
        self.imports = []

    def GetRootFolder(self):
        return self.root

    def GetCurrentFolder(self):
        return self.current

    def SetCurrentFolder(self, f):
        self.current = f
        return True

    def AddSubFolder(self, parent, name):
        f = Folder(name)
        parent.subs.append(f)
        return f

    def ImportMedia(self, infos):
        out = []
        for info in infos:
            pat, a, b = info["FilePath"], info["StartIndex"], info["EndIndex"]
            files = [pat % i for i in range(a, b + 1)]
            missing = [f for f in files if not os.path.isfile(f)]
            assert not missing, f"ImportMedia: missing {missing[:2]}"
            self.imports.append(dict(info, folder=self.current.name))
            m = re.match(r"^(.*)%0(\d)d(\.\w+)$", os.path.basename(pat))
            shown = f"{m.group(1)}[{a:0{int(m.group(2))}d}-{b:0{int(m.group(2))}d}]{m.group(3)}"
            mpi = MediaPoolItem({"File Path": os.path.join(os.path.dirname(pat), shown),
                                 "Frames": str(b - a + 1)})
            mpi.files = files
            self.current.clips.append(mpi)
            out.append(mpi)
        return lua(out)

    def AppendToTimeline(self, infos):
        out = []
        for info in infos:
            n = info["endFrame"] - info["startFrame"] + 1
            it = TimelineItem(info["mediaPoolItem"].name, info["recordFrame"], n, info["trackIndex"],
                              info["mediaPoolItem"])
            self.timeline.put(it)
            out.append(it)
        return lua(out)


# ------------------------------------------------------------------------- timeline
class TimelineItem:
    _n = 0

    def __init__(self, name, start, duration, track, mpi=None, left_offset=0, source_start=None,
                 comps=0):
        TimelineItem._n += 1
        self.name, self.start, self.duration, self.track = name, start, duration, track
        self.mpi, self.left_offset, self.source_start = mpi, left_offset, source_start
        self.uid = f"uid-{TimelineItem._n:04d}-abcdef"
        self.comps = [Comp() for _ in range(comps)]
        self.enabled = True
        self.props = {"ZoomX": 1.0, "Pan": 0.0}

    def GetName(self): return self.name
    def GetUniqueId(self): return self.uid
    def GetStart(self, *_a): return self.start
    def GetEnd(self, *_a): return self.start + self.duration
    def GetDuration(self, *_a): return self.duration
    def GetLeftOffset(self, *_a): return self.left_offset
    def GetSourceStartFrame(self): return self.source_start
    def GetMediaPoolItem(self): return self.mpi
    def GetTrackTypeAndIndex(self): return ["video", self.track]
    def GetFusionCompCount(self): return len(self.comps)
    def GetClipEnabled(self): return self.enabled
    def GetProperty(self, k): return self.props.get(k)

    def SetProperty(self, k, v):
        self.props[k] = v
        return True

    def SetClipEnabled(self, b):
        self.enabled = bool(b)
        return True

    def AddFusionComp(self):
        c = Comp()
        self.comps.append(c)
        return c

    def ExportFusionComp(self, path, idx):
        Path(path).write_text(json.dumps(self.comps[idx - 1].dump()))
        return True

    def ImportFusionComp(self, path):
        c = Comp.load(json.loads(Path(path).read_text()))
        self.comps.append(c)
        return c


class Timeline:
    def __init__(self, n_tracks=2):
        self.tracks = {t: [] for t in range(1, n_tracks + 1)}
        self.enabled = {t: True for t in self.tracks}
        self.names = {}
        self.current = None
        self.enable_log = []

    def put(self, it):
        self.tracks.setdefault(it.track, []).append(it)

    def GetName(self): return "Timeline 1"
    def GetSetting(self, k): return "25" if k == "timelineFrameRate" else ""
    def GetCurrentVideoItem(self): return self.current
    def GetTrackCount(self, kind): return len(self.tracks) if kind == "video" else 0
    def GetItemListInTrack(self, kind, t): return lua(self.tracks.get(t, []))
    def GetIsTrackEnabled(self, kind, t): return self.enabled[t]

    def SetTrackEnable(self, kind, t, b):
        self.enabled[t] = bool(b)
        self.enable_log.append((t, bool(b)))
        return True

    def AddTrack(self, kind, *_a):
        t = len(self.tracks) + 1
        self.tracks[t], self.enabled[t] = [], True
        return True

    def SetTrackName(self, kind, t, name):
        self.names[t] = name
        return True


# ------------------------------------------------------------------------- render
class Project:
    def __init__(self, timeline, sources):
        """sources: {track: folder of frames for that track (PNG), indexed by timeline frame
        - start of the timeline}; what Deliver would composite, per enabled track."""
        self.timeline, self.sources = timeline, sources
        self.media_pool = MediaPool(timeline)
        self.settings, self.jobs, self.presets = {}, {}, {"Custom": {}}
        self.mode, self.fmt = 0, None
        self.restored = []

    def GetName(self): return "Test Project"
    def GetCurrentTimeline(self): return self.timeline
    def GetMediaPool(self): return self.media_pool
    def GetSetting(self, k): return "25"
    def GetRenderFormats(self): return {"QuickTime": "mov", "EXR": "exr"}

    def GetRenderCodecs(self, fmt):
        assert fmt == "exr"
        return {"RGB Float (No compression)": "RGBFloat", "RGB Half (ZIP)": "RGBHalfZIP"}

    def SetCurrentRenderFormatAndCodec(self, f, c):
        self.fmt = (f, c)
        return True

    def GetCurrentRenderMode(self): return self.mode

    def SetCurrentRenderMode(self, m):
        self.mode = m
        return True

    def SetRenderSettings(self, d):
        self.settings.update(d)
        return True

    def SaveAsNewRenderPreset(self, name):
        self.presets[name] = dict(self.settings)
        return True

    def LoadRenderPreset(self, name):
        self.settings = dict(self.presets[name])
        self.restored.append(name)
        return True

    def DeleteRenderPreset(self, name):
        self.presets.pop(name, None)
        return True

    def AddRenderJob(self):
        jid = f"job{len(self.jobs) + 1}"
        self.jobs[jid] = dict(self.settings)
        return jid

    def DeleteRenderJob(self, jid):
        self.jobs.pop(jid, None)
        return True

    def StartRendering(self, jobs, interactive=False):
        for jid in jobs:
            self._render(self.jobs[jid])
        return True

    def IsRenderingInProgress(self): return False
    def GetRenderJobStatus(self, jid): return {"JobStatus": "Complete", "CompletionPercentage": 100}

    def _render(self, s):
        """Composite the enabled tracks (bottom first) for MarkIn..MarkOut and write EXR."""
        import numpy as np
        on = [t for t in sorted(self.sources) if self.timeline.enabled.get(t)]
        assert self.fmt == ("exr", "RGBHalfZIP")
        for fr in range(s["MarkIn"], s["MarkOut"] + 1):
            acc = None
            for t in on:
                f = Path(self.sources[t]) / f"f_{fr + 1:04d}.png"
                a = np.asarray(glitch.Image.open(f).convert("RGBA"), np.float32)
                if acc is None:
                    acc = a
                else:                                          # "over", straight alpha in 0..255
                    al = a[..., 3:] / 255.0
                    acc = np.concatenate([a[..., :3] * al + acc[..., :3] * (1 - al),
                                          np.maximum(a[..., 3:], acc[..., 3:])], -1)
            if not s.get("ExportAlpha"):
                acc[..., 3] = 255.0
            dst = Path(s["TargetDir"]) / f"{s['CustomName']}{fr:08d}.exr"
            glitch.save_frame(glitch.Image.fromarray(acc.clip(0, 255).astype(np.uint8), "RGBA"), dst,
                              {"exr_space": "display"})


# ------------------------------------------------------------------------- Fusion
INPUTS = {
    "Loader": [("Clip", "Filename"), ("GlobalIn", "Global In"), ("GlobalOut", "Global Out")],
    "BitmapMask": [("Image", "Image"), ("Channel", "Channel"), ("Low", "Low"), ("High", "High")],
    "Glow": [("Input", "Input"), ("EffectMask", "Effect Mask"), ("GlowMask", "Glow Mask"),
             ("Glow", "Glow")],
    "ColorCorrector": [("Input", "Input"), ("EffectMask", "Effect Mask")],
    "MediaIn": [], "MediaOut": [("Input", "Input")], "Note": [("Comments", "Comments")],
}


class Input:
    def __init__(self, i, n):
        self.i, self.n = i, n

    def GetAttrs(self):
        return {"INPS_ID": self.i, "INPS_Name": self.n}


class Tool:
    def __init__(self, kind, name):
        self.kind, self.name = kind, name
        self.values, self.links = {}, {}

    def _ids(self):
        return {i for i, _ in INPUTS.get(self.kind, [])}

    def GetInputList(self):
        return lua([Input(i, n) for i, n in INPUTS.get(self.kind, [])])

    def SetAttrs(self, d):
        self.name = d.get("TOOLS_Name", self.name)
        return True

    def GetAttrs(self):
        return {"TOOLS_Name": self.name, "TOOLS_RegID": self.kind}

    def SetInput(self, k, v, *_t):
        assert k in self._ids(), f"{self.kind} has no input {k}"
        self.values[k] = v
        return True

    def ConnectInput(self, k, other):
        assert k in self._ids(), f"{self.kind} has no input {k}"
        self.links[k] = other
        return True


class Flow:
    def __init__(self):
        self.pos = {}

    def GetPosTable(self, t):
        return lua(list(self.pos.get(id(t), (0.0, 0.0))))

    def SetPos(self, t, x, y):
        self.pos[id(t)] = (x, y)


class Comp:
    def __init__(self):
        self.tools = [Tool("MediaIn", "MediaIn1"), Tool("MediaOut", "MediaOut1")]
        self.CurrentFrame = Obj(FlowView=Flow())
        self.locked = 0

    def GetAttrs(self): return {"COMPN_RenderStart": 0.0, "COMPN_RenderEnd": 100.0}
    def Lock(self): self.locked += 1
    def Unlock(self): self.locked -= 1

    def GetToolList(self, selected=False, kind=None):
        return lua([t for t in self.tools if kind is None or t.kind == kind])

    def AddTool(self, kind, x=-32768, y=-32768):
        assert kind in INPUTS, f"unknown tool {kind}"
        t = Tool(kind, f"{kind}{sum(x.kind == kind for x in self.tools) + 1}")
        self.tools.append(t)
        return t

    def FindTool(self, name):
        return next((t for t in self.tools if t.name == name), None)

    def dump(self):
        idx = {id(t): i for i, t in enumerate(self.tools)}
        return [{"kind": t.kind, "name": t.name, "values": t.values,
                 "links": {k: idx[id(o)] for k, o in t.links.items()}} for t in self.tools]

    @classmethod
    def load(cls, d):
        c = cls()
        c.tools = [Tool(x["kind"], x["name"]) for x in d]
        for t, x in zip(c.tools, d):
            t.values = dict(x["values"])
            t.links = {k: c.tools[i] for k, i in x["links"].items()}
        return c


# ------------------------------------------------------------------------- UIManager
class Widget:
    def __init__(self, kind, props):
        self.kind = kind
        self.__dict__.update(props)
        self.items, self.lines = [], []
        self.__dict__.setdefault("CurrentIndex", 0)
        self.__dict__.setdefault("Text", "")
        self.__dict__.setdefault("Checked", False)              # as Qt widgets have them
        self.__dict__.setdefault("Value", 0)

    def AddItem(self, x): self.items.append(x)
    def Append(self, x): self.lines.append(x)


class Events:
    def __init__(self):
        object.__setattr__(self, "_e", {})

    def __getattr__(self, name):
        return self._e.setdefault(name, Obj())


class UI:
    def __init__(self):
        self.widgets = {}

    def _mk(kind):
        def make(self, props=None, children=None):
            if isinstance(props, list):
                props, children = {}, props
            w = Widget(kind, props or {})
            w.children = children or []
            if "ID" in (props or {}):
                self.widgets[props["ID"]] = w
            return w
        return make
    for _k in ("VGroup", "HGroup", "Label", "Button", "ComboBox", "CheckBox", "SpinBox",
               "DoubleSpinBox", "TextEdit", "LineEdit"):
        locals()[_k] = _mk(_k)
    del _k

    def Timer(self, props):
        w = Widget("Timer", props)
        w.running = False
        w.Start = lambda: setattr(w, "running", True)
        w.Stop = lambda: setattr(w, "running", False)
        self.widgets[props["ID"]] = w
        return w


class Window:
    def __init__(self, ui):
        self.ui, self.On, self.shown = ui, Events(), False

    def GetItems(self): return self.ui.widgets
    def Show(self): self.shown = True
    def Hide(self): self.shown = False


class Dispatcher:
    def __init__(self, ui):
        self.ui, self.On, self.exited = ui, Events(), False

    def AddWindow(self, props, root):
        self.win = Window(self.ui)
        return self.win

    def RunLoop(self): pass
    def ExitLoop(self): self.exited = True


class Fusion:
    def __init__(self):
        self.UIManager = UI()
        self.request = None

    def RequestFile(self, *_a):
        return self.request


class Bmd:
    @staticmethod
    def UIDispatcher(ui):
        return Dispatcher(ui)


class Resolve:
    def __init__(self, project):
        self.project = project

    def GetProjectManager(self):
        return Obj(GetCurrentProject=lambda: self.project)

    def Fusion(self):
        return Fusion()
