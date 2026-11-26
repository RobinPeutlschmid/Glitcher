"""The app and the CLI rendering through the native engine (glitch_core): glitch_native.glitch()
is a drop-in for glitch.glitch(), the planner's frames go through it, the pool runs it."""
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import glitch
import synth

ROOT = Path(__file__).resolve().parent.parent

gn = pytest.importorskip("glitch_native")


@pytest.fixture(scope="module", autouse=True)
def need_engine():
    if not gn.available():
        from conftest import skip_or_fail
        skip_or_fail("NATIVE", f"native engine not built: {gn.unavailable_reason()}")


@pytest.fixture(scope="module")
def fig(tmp_path_factory):
    d = tmp_path_factory.mktemp("fig")
    paths, npz, _csv = synth.make_figure(d, 12, 160, 220)
    return [Image.open(p).convert("RGBA") for p in paths], glitch.Landmarks(npz)


def test_same_types_and_sizes(fig):
    frames, lm = fig
    o = gn.glitch(frames[6], t=6 / 25, frame=6, seed=3, intensity=0.8, pad=0.1,
                  history=[frames[6 - k] for k in range(1, 7)])
    assert isinstance(o, Image.Image) and o.size == (192, 264)     # padded like glitch.glitch
    ref = glitch.glitch(frames[6], t=6 / 25, frame=6, seed=3, intensity=0.8, pad=0.1)
    assert o.size == ref.size
    f = np.asarray(frames[6]).astype(np.float32) * 1.5               # HDR above 255 survives
    o2 = gn.glitch(f, frame=6, seed=3, intensity=0.0)
    assert o2.dtype == np.float32 and o2.shape == f.shape and o2[..., :3].max() > 255
    t = gn.glitch(frames[6], frame=6, seed=3, pad=0.2, trim=True)
    assert t.size[0] <= 160 + 2 and t.size[1] <= 220 + 2


def test_reveal_and_landmarks_planned(fig):
    frames, lm = fig
    hid = gn.glitch(frames[5], frame=5, seed=1, reveal=("hidden",))
    assert np.asarray(hid)[..., 3].max() == 0
    vis = np.asarray(gn.glitch(frames[5], frame=5, seed=1, intensity=0.0, reveal=("visible",)))
    assert vis[..., 3].max() == 255
    half = np.asarray(gn.glitch(frames[5], frame=5, seed=1, intensity=0.0, reveal=("in", 0.5, 0)))
    assert 0 < (half[..., 3] > 0).sum() < (vis[..., 3] > 0).sum()
    # protecting the head with the frame's landmarks keeps the head pixels as they were
    src = np.asarray(frames[5])
    out = np.asarray(gn.glitch(frames[5], frame=5, seed=2, intensity=1.0, landmarks=lm.at(5),
                               protect=("head",), gains={"split": 0, "tint": 0}))
    x, y = lm.at(5)[0][:2].astype(int)
    assert np.abs(out[y - 3:y + 3, x - 3:x + 3].astype(int) - src[y - 3:y + 3, x - 3:x + 3]).max() <= 2


def test_matte_and_plate(fig):
    frames, _ = fig
    plate = Image.new("RGBA", frames[0].size, (40, 90, 160, 255))
    o, m = gn.glitch(frames[8], frame=8, seed=4, intensity=1.0, plate=plate, matte=True,
                     history=[frames[8 - k] for k in range(1, 9)])
    assert o.size == frames[0].size and m.shape == (220, 160)
    assert np.asarray(o)[..., 3].min() == 255                       # over the plate: opaque
    assert 0 < m.max() <= 1


def _run(args, engine, tmp):
    env = dict(os.environ, GLITCHER_ENGINE=engine)
    r = subprocess.run([sys.executable, str(ROOT / "glitch.py"), *args], env=env, capture_output=True,
                       text=True, timeout=600, cwd=tmp)
    assert r.returncode == 0, r.stderr[-2000:]
    return r.stderr


@pytest.mark.parametrize("jobs", ["1", "3"])
def test_cli_renders_natively(tmp_path, jobs):
    paths, npz, _ = synth.make_figure(tmp_path / "in", 16, 160, 200)
    Image.new("RGBA", (160, 200), (40, 90, 160, 255)).save(tmp_path / "plate.png")
    err = _run([str(tmp_path / "in"), "-o", str(tmp_path / "out"), "-s", "4", "-j", jobs, "--pitch", "15",
                "--plate", str(tmp_path / "plate.png"), "--hits", "8", "--appear", "2:0.2",
                "--landmarks", str(npz), "--protect", "head", "--glitch-pass", "--follow", "0.4"],
               "native", tmp_path)
    assert "render code: native" in err
    outs = sorted((tmp_path / "out").glob("*.png"))
    assert len(outs) == 16
    assert len(list((tmp_path / "out" / "glitch_pass").glob("*.png"))) == 16
    first = np.asarray(Image.open(outs[0]))
    assert first[..., 3].min() == 255 and (first[..., :3] == (40, 90, 160)).all()   # hidden: plate
    # the same command renders the same frames again (seeded, no state between runs)
    _run([str(tmp_path / "in"), "-o", str(tmp_path / "again"), "-s", "4", "-j", "2", "--pitch", "15",
          "--plate", str(tmp_path / "plate.png"), "--hits", "8", "--appear", "2:0.2",
          "--landmarks", str(npz), "--protect", "head", "--follow", "0.4"], "native", tmp_path)
    for a in outs:
        assert np.array_equal(np.asarray(Image.open(a)), np.asarray(Image.open(tmp_path / "again" / a.name)))


def test_forced_python_and_missing_library(tmp_path):
    synth.make_figure(tmp_path / "in", 4, 64, 80)
    err = _run([str(tmp_path / "in"), "-o", str(tmp_path / "o1")], "python", tmp_path)
    assert "render code: native" not in err
    env = dict(os.environ, GLITCHER_ENGINE="native", GLITCHER_CORE=str(tmp_path / "none.so"))
    # with the library hidden: the auto mode falls back, native asks for it
    code = ("import glitch_native as g, sys; g.library_path = lambda: None; import glitch; "
            "print(glitch.native()); ")
    r = subprocess.run([sys.executable, "-c", code], env=dict(env, GLITCHER_ENGINE="auto"),
                       capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0 and r.stdout.strip() == "None", r.stderr
    r = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, cwd=ROOT)
    assert r.returncode != 0 and "GLITCHER_ENGINE=native" in r.stderr


def test_editor_previews_natively(tmp_path):
    pytest.importorskip("tkinter")
    if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
        pytest.skip("no display")
    synth.make_figure(tmp_path / "in", 6, 96, 120)
    env = dict(os.environ, GLITCHER_ENGINE="native")
    r = subprocess.run([sys.executable, str(ROOT / "glitch_gui.py"), "--self-test", str(tmp_path / "in")],
                       env=env, capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "renders with native (glitch_core" in r.stdout
    r = subprocess.run([sys.executable, str(ROOT / "glitch_gui.py"), "--engine-check"], env=env,
                       capture_output=True, text=True, timeout=60)
    assert r.returncode == 0 and r.stdout.startswith("native"), r.stdout + r.stderr


def test_engine_version_is_the_apps():
    assert gn.engine().version == glitch.__version__, "bump engine/Cargo.toml's version too"


def test_conversions_match_numpy():
    """The engine's straight <-> premultiplied conversions give numpy's exact numbers."""
    lib = gn.engine().lib
    rng = np.random.default_rng(1)
    a = rng.integers(0, 256, (37, 53, 4), dtype=np.uint8)
    c = gn._canvas(lib, Image.fromarray(a, "RGBA"), (60, 40), (3, 2))
    assert np.array_equal(c[2:39, 3:56], gn.to_premul(a)) and c[:2].max() == 0 and c[:, :3].max() == 0
    f = (rng.random((37, 53, 4)) * 400).astype(np.float32)
    fa = f / 255.0
    al = np.clip(fa[..., 3:4], 0, 1)
    assert np.array_equal(gn._canvas(lib, f, (53, 37), (0, 0)), np.concatenate([fa[..., :3] * al, al], -1))
    p = (rng.random((30, 40, 4)) * 1.3).astype(np.float32)
    p[..., 3] = np.clip(p[..., 3], 0, 1)
    p[5, 5, 3] = 0
    assert np.array_equal(gn._from_canvas(lib, p, (2, 3, 40, 30), False), gn.from_premul(p)[3:30, 2:40])
    assert np.array_equal(gn._from_canvas(lib, p, (0, 0, 40, 30), True), gn.from_premul(p, as_uint8=False))
