"""GUI smoke test: the editor starts, its preview matches the command line, and the main
interactions (undo, presets, folding, theme, full render) work. Needs a display (xvfb on Linux)."""
import json
import time
from pathlib import Path

import numpy as np
import pytest

import glitch
import synth
from conftest import skip_or_fail

tk = pytest.importorskip("tkinter")


def pump(root, sec, until=None):
    t = time.time()
    while time.time() - t < sec:
        root.update()
        if until and until():
            return True
        time.sleep(0.01)
    return bool(until and until())


@pytest.fixture(scope="module")
def app(frames, tmp_path_factory):
    import glitch_gui as GG
    home = tmp_path_factory.mktemp("home")
    GG.PREFS_FILE = home / "prefs.json"                    # never touch the real ~/.glitcher.json
    try:
        root = tk.Tk()
    except tk.TclError as ex:
        skip_or_fail("GUI", f"no display for Tk: {ex}")
    root.geometry("1280x860")
    a = GG.App(root, str(frames))
    assert pump(root, 90, lambda: a.plan is not None and a.current_img is not None), a.status.cget("text")
    pump(root, 0.8)                                        # let the first undo snapshot land
    yield a
    a.close()


def wait_preview(app, frame=None, scale=None):
    def ready():
        t = app.live_token
        return (t is not None and t[0] == app.plan_key and t in app.cache
                and (scale is None or t[1] == scale) and (frame is None or t[2] == frame))
    assert pump(app.root, 90, ready), app.status.cget("text")
    return np.array(app.cache[app.live_token])


def test_window_title_has_version(app):
    v = glitch.__version__
    short = v[:-2] if v.endswith(".0") else v              # "1.3.0" is shown as "Glitcher 1.3"
    assert app.root.title().startswith(f"{glitch.APP_NAME} {short}")


def test_preview_matches_cli(app, tmp_path):
    app.vars["intensity"].set(0.8)
    app.vars["gain_echo"].set(2.0)
    app.vars["pscale"].set("100%")
    app.vars["frame"].set(6)
    app._frame_moved()
    app._settings_changed()
    prev = wait_preview(app, frame=6, scale=1.0)
    out = tmp_path / "cli"
    assert glitch.main(app.argv(with_output=False) + ["-o", str(out), "-j", "1"]) == 0
    cli = synth.load(out / "f_0007.png")
    assert prev.shape == cli.shape and (prev == cli).all()


def test_undo_and_redo(app):
    pump(app.root, 0.6)
    before = app.vars["intensity"].get()
    for v in np.linspace(0.2, 0.3, 8):                    # one drag = one undo step
        app.vars["intensity"].set(round(float(v), 2))
        app._settings_changed()
        pump(app.root, 0.02)
    pump(app.root, 0.7)
    n = len(app._undo)
    app.undo()
    pump(app.root, 0.2)
    assert app.vars["intensity"].get() == pytest.approx(before)
    app.redo()
    pump(app.root, 0.2)
    assert app.vars["intensity"].get() == pytest.approx(0.3)
    assert len(app._undo) == n


def test_preset_round_trip(app, tmp_path, monkeypatch):
    import glitch_gui as GG
    p = tmp_path / "look.json"
    monkeypatch.setattr(GG.filedialog, "asksaveasfilename", lambda **k: str(p))
    monkeypatch.setattr(GG.filedialog, "askopenfilename", lambda **k: str(p))
    app.vars["seed"].set(4242)
    app.save_preset(ask=True)
    d = json.loads(p.read_text("utf-8"))
    assert d["seed"] == 4242 and d.get("glitcher_version") == glitch.__version__
    assert Path(d["input"]) == Path(app.vars["input"].get())
    app.vars["seed"].set(1)
    app.load_preset()
    pump(app.root, 0.2)
    assert app.vars["seed"].get() == 4242


def test_sections_fold_theme_and_panel(app):
    for toggle in list(app._section_toggles.values()):
        toggle()
        toggle()
    app.toggle_theme()
    app.toggle_theme()
    app.toggle_panel()
    app.toggle_panel()
    pump(app.root, 0.3)
    assert not app.status.cget("text").startswith("⚠")


def test_full_render_matches_cli(app, tmp_path):
    out = tmp_path / "gui_out"
    app.vars["output"].set(str(out))
    app.render_all()
    assert pump(app.root, 300, lambda: app.render_cancel is None and len(list(out.glob("*.png"))) == 12)
    ref = tmp_path / "cli_out"
    assert glitch.main(app.argv(with_output=False) + ["-o", str(ref), "-j", "1"]) == 0
    for f in sorted(out.glob("*.png")):
        assert synth.same_images(f, ref / f.name), f.name
