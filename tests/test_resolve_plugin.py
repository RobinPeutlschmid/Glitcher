"""The DaVinci Resolve plugin, end to end against a fake Resolve (tests/fake_resolve.py):
choosing the source, rendering from the timeline, Glitcher rendering the job, importing,
the Fusion comp, versions, and the panel's buttons."""
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pytest

import fake_resolve as R
import glitch
import glitch_gui
import synth
from conftest import ROOT, skip_or_fail

spec = importlib.util.spec_from_file_location("glitcher_resolve", ROOT / "resolve" / "Glitcher.py")
G = importlib.util.module_from_spec(spec)
spec.loader.exec_module(G)


@pytest.fixture(autouse=True)
def need_exr():
    try:
        import OpenEXR  # noqa: F401
    except ImportError:
        skip_or_fail("EXR", "OpenEXR is not installed (the plugin renders EXR)")


def cfg(tmp_path):
    return {"glitcher": [sys.executable, str(ROOT / "glitch_gui.py")], "cache": str(tmp_path / "cache")}


def wait(plugin, timeout=300):
    t = time.time()
    while time.time() - t < timeout:
        r = plugin.poll()
        if r and r[0] == "done":
            return r
        time.sleep(0.2)
    raise AssertionError("render did not finish")


def files_setup(tmp_path, n=12, left=2, dur=8, alpha="Premultiplied", extra_before=True):
    """A timeline whose V2 clip is an RGBA PNG sequence on disk, trimmed to frames 3..10."""
    seq = tmp_path / "shot"
    synth.make_frames(seq, n)
    if extra_before:                                           # another take in the same folder
        synth.make_frames(tmp_path / "x", 2)
        for i, f in enumerate(sorted((tmp_path / "x").glob("*.png"))):
            f.rename(seq / f"take2_{i:04d}.png")
    tl = R.Timeline(2)
    mpi = R.MediaPoolItem({"File Path": str(seq / f"f_[0001-{n:04d}].png"), "Alpha mode": alpha,
                           "Clip Name": "f_[0001-0012].png"})
    item = R.TimelineItem("hero", 86400 + 100, dur, 2, mpi, left_offset=left)
    tl.put(item)
    tl.current = item
    proj = R.Project(tl, {})
    return R.Resolve(proj), proj, tl, item, seq


# ------------------------------------------------------------------ pure helpers
def test_media_sequence_and_offsets(tmp_path):
    _res, _p, _tl, item, seq = files_setup(tmp_path)
    s = G.media_sequence(item.GetMediaPoolItem().GetClipProperty())
    assert s["alpha"] and s["count"] == 12 and s["first"] == 1
    assert len(s["files"]) == 12 and s["offset"] == 0          # the other take isn't in the glob
    info = {"left_offset": 2, "source_start": None}
    assert G.clip_offset(info, s) == 2
    assert G.clip_offset({"left_offset": 0, "source_start": 3}, s) == 2   # frame number 3
    assert G.clip_offset({"left_offset": 0, "source_start": 0}, s) == 0   # offset into the clip
    assert G.media_sequence({"File Path": str(seq / "movie.mov")}) is None


def test_source_choice(tmp_path):
    res, _p, _tl, item, _seq = files_setup(tmp_path)
    s = G.Session(res, log=lambda m: None)
    info = s.describe(item)
    assert s.plan_source(info, "auto", False)[0] == "files"
    assert s.plan_source(info, "render", False)[0] == "render"
    assert s.plan_source(info, "auto", True)[0] == "render"        # a plate needs timeline size
    with pytest.raises(G.GlitcherError):
        s.plan_source(info, "files", True)
    item.comps.append(R.Comp())                                     # keyed in Fusion
    assert s.plan_source(s.describe(item), "auto", False)[0] == "render"
    item.comps.clear()
    item.mpi.props["Alpha mode"] = "None"
    assert s.plan_source(s.describe(item), "auto", False)[0] == "render"
    assert s.plan_source(s.describe(item), "files", False)[0] == "files"   # you said so
    item.mpi.props["Alpha mode"] = "Premultiplied"
    item.duration = 20                                              # longer than its media
    assert s.plan_source(s.describe(item), "auto", False)[0] == "render"


def test_preset_and_versions(tmp_path):
    f = tmp_path / "job"
    p, d = G.ensure_preset(str(f), seed=42, intensity=0.8)
    assert json.loads(Path(p).read_text()) == {"seed": 42, "intensity": 0.8}
    _p, d = G.ensure_preset(str(f))                                 # reopening keeps the values
    assert d["seed"] == 42
    assert G.next_version(str(f)) == "v001"
    (f / "out" / "v001").mkdir(parents=True)
    (f / "out" / "v007").mkdir()
    assert G.versions(str(f)) == ["v001", "v007"] and G.next_version(str(f)) == "v008"
    ed = tmp_path / "look.json"
    ed.write_text(json.dumps({"seed": 3, "input": "/x", "output": "/y", "tint": 0.4}))
    G.copy_preset(str(ed), str(f))
    assert json.loads((f / "preset.json").read_text()) == {"seed": 3, "tint": 0.4}


# ------------------------------------------------------------------ the round trip
def test_files_round_trip(tmp_path):
    res, proj, tl, item, seq = files_setup(tmp_path)
    logs = []
    pl = G.Plugin(res, cfg(tmp_path), log=logs.append)
    folder, job = pl.prepare(item, seed=11, intensity=0.7)
    assert job["source"] == "files" and job["range"] == [2, 9] and job["exr_space"] == "linear"
    assert job["frames"] == [f"f_{i:04d}.exr" for i in range(3, 11)]
    pl.render(folder, job)
    assert pl.busy()
    r = wait(pl)
    assert r[1] == 0, (Path(folder) / "render.log").read_text()[-800:]
    out = Path(job["output"])
    assert out.name == "v001" and sorted(p.name for p in out.glob("*.exr")) == job["frames"]
    # the job renders exactly what the editor's settings give on the command line
    ref = tmp_path / "ref"
    s = glitch_gui.job_settings(glitch_gui.read_job(Path(folder) / "job.json"))
    s["output"] = str(ref)
    assert glitch.main(glitch_gui.preset_argv(s) + ["--range", "2-9", "-j", "1"]) == 0
    for name in job["frames"]:
        a, _ = glitch.load_frame(out / name, {"exr_space": "linear"})
        b, _ = glitch.load_frame(ref / name, {"exr_space": "linear"})
        assert np.array_equal(np.asarray(a), np.asarray(b)), name
    # import
    new = pl.import_latest(folder, json.loads((Path(folder) / "job.json").read_text()))
    imp = proj.media_pool.imports
    assert [i["StartIndex"] for i in imp] == [3, 3] and [i["EndIndex"] for i in imp] == [10, 10]
    assert imp[0]["FilePath"].endswith("f_%04d.exr") and "glitch_pass" in imp[1]["FilePath"]
    assert {i["folder"] for i in imp} == {"Glitcher"}
    assert new.track == 3 and tl.names[3] == "Glitcher"
    assert new.start == item.start and new.duration == 8
    assert not item.enabled                                         # the original is off
    assert new.props["ZoomX"] == item.props["ZoomX"]                # same framing
    comp = new.comps[0]
    t = {x.name: x for x in comp.tools}
    assert t["GlitchPass"].values["Clip"].endswith(os.path.join("glitch_pass", "f_0003.exr"))
    assert t["GlitchPass"].values["GlobalIn"] == 0 and t["GlitchPass"].values["GlobalOut"] == 7
    assert t["PassMatte"].links["Image"] is t["GlitchPass"]
    assert t["GlitchGlow"].links["Input"] is t["MediaIn1"]
    assert t["GlitchGlow"].links["GlowMask"] is t["PassMatte"]     # light from the glitch only
    assert t["GlitchGrade"].links["EffectMask"] is t["PassMatte"]
    assert t["MediaOut1"].links["Input"] is t["GlitchGrade"]
    assert comp.locked == 0


def test_second_version_keeps_the_comp(tmp_path):
    res, proj, tl, item, seq = files_setup(tmp_path)
    pl = G.Plugin(res, cfg(tmp_path), log=lambda m: None)
    folder, job = pl.prepare(item)
    pl.render(folder, job)
    assert wait(pl)[1] == 0
    first = pl.import_latest(folder, G.read_job(folder))
    first.comps[0].FindTool("GlitchGlow").values["Glow"] = 0.9      # your tweak in Fusion
    folder, job = pl.prepare(item, intensity=0.3)
    assert job["output"].endswith("v002")
    pl.render(folder, job)
    assert wait(pl)[1] == 0
    second = pl.import_latest(folder, G.read_job(folder))
    assert not first.enabled and second.enabled
    comp = second.comps[0]
    assert comp.FindTool("GlitchGlow").values["Glow"] == 0.9       # carried over
    assert "v002" in comp.FindTool("GlitchPass").values["Clip"]
    assert [i["version"] for i in G.read_job(folder)["imported"]] == ["v001", "v002"]


def test_render_from_timeline_with_plate(tmp_path):
    """Keyed in Resolve: the subject track rendered alone (with alpha), the tracks below as the
    plate; the glitch spills into it and comes back opaque."""
    n = 12
    subj, bg = tmp_path / "subj", tmp_path / "bg"
    synth.make_frames(subj, n)
    bg.mkdir()
    for i in range(n):
        synth.make_plate(bg / f"f_{i + 1:04d}.png")
    tl = R.Timeline(3)
    item = R.TimelineItem("keyed", 3, 6, 2, None, comps=1)           # frames 3..8, has a comp
    tl.put(item)
    tl.current = item
    tl.enabled[3] = False                                           # a disabled track stays off
    proj = R.Project(tl, {1: bg, 2: subj, 3: subj})
    res = R.Resolve(proj)
    pl = G.Plugin(res, cfg(tmp_path), log=lambda m: None)
    folder, job = pl.prepare(item, want_plate=True)
    assert job["source"] == "render" and job["exr_space"] == "display" and job["range"] is None
    src, plate = sorted(Path(folder, "src").glob("*.exr")), sorted(Path(folder, "plate").glob("*.exr"))
    assert len(src) == len(plate) == 6
    assert tl.enabled == {1: True, 2: True, 3: False}               # restored afterwards
    assert proj.restored == [G.RESTORE_PRESET] * 2 and G.RESTORE_PRESET not in proj.presets
    a, _ = glitch.load_frame(src[0], {"exr_space": "display"})
    assert np.asarray(a)[..., 3].min() == 0                          # subject: has alpha
    b, _ = glitch.load_frame(plate[0], {"exr_space": "display"})
    assert np.asarray(b)[..., 3].min() == 255                        # plate: opaque
    pl.render(folder, job)
    assert wait(pl)[1] == 0
    out = sorted(Path(job["output"]).glob("*.exr"))
    assert len(out) == 6
    o, _ = glitch.load_frame(out[0], {"exr_space": "display"})
    assert np.asarray(o)[..., 3].min() >= 254                        # composited over the plate
    pl.import_latest(folder, G.read_job(folder))
    assert len(tl.tracks[4]) == 1                                   # new top track


def test_resolve_render_off_by_one_is_trimmed(tmp_path):
    """If Resolve treats MarkOut as exclusive-plus-one, the extra frame is dropped."""
    _res, proj, tl, item, _seq = files_setup(tmp_path)
    item.start = 1                                              # timeline frames 1..8
    synth.make_frames(tmp_path / "t", 12)
    proj.sources = {2: tmp_path / "t"}
    real = proj._render
    proj._render = lambda s: real(dict(s, MarkOut=s["MarkOut"] + 1))
    s = G.Session(R.Resolve(proj), log=lambda m: None)
    seq = s.render_range(s.describe(item), str(tmp_path / "r"), "src_", "subject")
    assert len(seq["files"]) == item.duration


def test_bad_job_and_missing_glitcher(tmp_path):
    bad = tmp_path / "job.json"
    bad.write_text("{}")
    assert glitch_gui.render_job(bad) == 2
    res, _p, _tl, item, _seq = files_setup(tmp_path)
    pl = G.Plugin(res, {"glitcher": [str(tmp_path / "nope.exe")], "cache": str(tmp_path / "c")},
                  log=lambda m: None)
    folder, job = pl.prepare(item)
    with pytest.raises(G.GlitcherError, match="Install DaVinci Resolve plugin"):
        pl.render(folder, job)
    with pytest.raises(G.GlitcherError, match="no complete render"):
        pl.import_latest(folder, job)


def test_install_writes_script_and_config(tmp_path, monkeypatch):
    monkeypatch.setattr(glitch_gui, "RESOLVE_CONFIG", tmp_path / "cfg.json")
    assert glitch_gui.install_resolve(str(tmp_path / "Scripts" / "Utility"), quiet=True) == 0
    assert (tmp_path / "Scripts" / "Utility" / "Glitcher.py").read_text() == \
        (ROOT / "resolve" / "Glitcher.py").read_text()
    c = json.loads((tmp_path / "cfg.json").read_text())
    assert c["version"] == glitch.__version__
    assert G.glitcher_command(c) == [sys.executable, str(ROOT / "glitch_gui.py")]
    assert G.PLUGIN_VERSION == glitch.__version__, "bump PLUGIN_VERSION in resolve/Glitcher.py too"


def test_scripts_folder_per_os(monkeypatch):
    for plat, part in (("win32", "Support"), ("darwin", "Library"), ("linux", ".local")):
        monkeypatch.setattr(glitch_gui.sys, "platform", plat)
        p = glitch_gui.resolve_scripts_dir()
        assert p.name == "Utility" and p.parent.name == "Scripts" and part in p.parts


# ------------------------------------------------------------------ the panel
def test_panel_buttons(tmp_path, monkeypatch):
    res, proj, tl, item, seq = files_setup(tmp_path)
    monkeypatch.setattr(G, "load_config", lambda *a: cfg(tmp_path))
    fu, bmd = R.Fusion(), R.Bmd()
    panel = G.Panel(res, fu, bmd)
    w = fu.UIManager.widgets
    assert "hero" in w["Clip"].Text and w["Seed"].Value == 7
    assert w["Mode"].items[0] == "Auto" and len(w["Space"].items) == 3
    w["Seed"].Value, w["Intensity"].Value = 5, 0.6
    panel.win.On.Render.Clicked(None)
    assert w["Render"].Text == "Cancel" and w["GlitcherPoll"].running
    t = time.time()
    while w["GlitcherPoll"].running and time.time() - t < 300:
        panel.disp.On.GlitcherPoll.Timeout(None)
        time.sleep(0.2)
    assert not w["GlitcherPoll"].running and w["Render"].Text == "Render"
    assert w["Status"].Text.startswith("imported"), w["Status"].Text + "\n" + "\n".join(w["Log"].lines)
    assert len(tl.tracks[3]) == 1
    job = G.read_job(G.job_folder(str(tmp_path / "cache"), "Test Project", "hero", item.uid))
    assert json.loads(Path(job["preset"]).read_text())["seed"] == 5
    panel.win.On.GlitcherWin.Close(None)
    assert panel.disp.exited


def test_panel_reports_problems(tmp_path, monkeypatch):
    res, proj, tl, item, seq = files_setup(tmp_path)
    tl.current = None
    monkeypatch.setattr(G, "load_config", lambda *a: cfg(tmp_path))
    fu = R.Fusion()
    G.Panel(res, fu, R.Bmd())
    assert "playhead" in fu.UIManager.widgets["Status"].Text      # a message, not a crash


def test_install_never_blocks_on_a_dialog(tmp_path, monkeypatch):
    """--install-resolve may only show a message box when there is no console at all; with
    captured output (scripts, CI) a dialog nobody can click would hang forever."""
    monkeypatch.setattr(glitch_gui, "RESOLVE_CONFIG", tmp_path / "cfg.json")
    shown = []
    monkeypatch.setattr(glitch_gui.messagebox, "showinfo", lambda *a, **k: shown.append(a))
    monkeypatch.setattr(glitch_gui.messagebox, "showerror", lambda *a, **k: shown.append(a))
    monkeypatch.setattr(glitch_gui, "_NO_CONSOLE", False)
    for frozen, plat in ((True, "win32"), (False, "win32"), (True, "darwin")):
        monkeypatch.setattr(glitch_gui.sys, "frozen", frozen, raising=False)
        monkeypatch.setattr(glitch_gui.sys, "platform", plat)
        assert glitch_gui.install_resolve(str(tmp_path / "S"), quiet=False) == 0
    assert shown == []
