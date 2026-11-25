"""The compiled kernels (glitch_numba.py) must give exactly the numpy code's bytes: on edge-case
pixels, on real frames, through every effect that uses them, through whole renders (PNG and
EXR, 2-D and 3-D, worker processes), so "faster" never means "different". Plus: the fallback
to numpy, the disk cache, and that the editor's own process never loads Numba."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pytest
from conftest import skip_or_fail

import glitch
import synth

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def F():
    if glitch._fast() is False:
        skip_or_fail("NUMBA", f"compiled kernels not available: {glitch.FAST_STATUS}")
    return glitch._FAST


@contextmanager
def numpy_only():
    saved = glitch._FAST
    glitch._FAST = False
    try:
        yield
    finally:
        glitch._FAST = saved


def edge_rgba(rng, shape):
    """Random RGBA with many exact 0 / 255 alphas and colours (where the branches are)."""
    a = rng.integers(0, 256, shape, dtype=np.uint8)
    al = a[..., 3]
    r = rng.random(shape[:2])
    al[r < 0.25] = 0
    al[(r >= 0.25) & (r < 0.45)] = 255
    a[..., :3][rng.random(shape[:2] + (3,)) < 0.1] = 255
    a[..., :3][rng.random(shape[:2] + (3,)) < 0.1] = 0
    return a


def test_under_exact_on_edge_cases(F):
    rng = np.random.default_rng(1)
    for i in range(20):
        shape = (int(rng.integers(1, 90)), int(rng.integers(1, 90)), 4)
        f, g = edge_rgba(rng, shape), edge_rgba(rng, shape)
        if i % 5 == 0:
            g[..., 3] = 0                                      # no ghost at all: front returned as is
        with numpy_only():
            ref = glitch._under(f, g)
        got = glitch._under(f, g)
        assert got.dtype == np.uint8 and np.array_equal(got, ref), i
    # every (front alpha, ghost alpha) pair, with extreme colours
    fa, ga = np.meshgrid(np.arange(256, dtype=np.uint8), np.arange(256, dtype=np.uint8))
    f = np.dstack([np.full_like(fa, 255), np.zeros_like(fa), np.full_like(fa, 7), fa])
    g = np.dstack([np.zeros_like(ga), np.full_like(ga, 255), np.full_like(ga, 128), ga])
    with numpy_only():
        ref = glitch._under(f, g)
    assert np.array_equal(glitch._under(f, g), ref)


def test_under_takes_views_and_read_only_arrays(F):
    rng = np.random.default_rng(2)
    big, g = edge_rgba(rng, (60, 80, 4)), edge_rgba(rng, (30, 40, 4))
    f = big[10:40, 20:60]                                      # a view: not contiguous
    ro = g.copy()
    ro.flags.writeable = False                                 # history frames are read-only
    with numpy_only():
        ref = glitch._under(f, ro)
    assert np.array_equal(glitch._under(f, ro), ref)
    assert np.array_equal(big[10:40, 20:60], f)               # inputs untouched


def test_float_images_stay_on_numpy(F, monkeypatch):
    calls = []
    monkeypatch.setattr(F, "under_u8", lambda *a: calls.append(1))
    f = np.random.default_rng(3).random((8, 8, 4)).astype(np.float32) * 300
    glitch._under(f, f[::-1].copy())
    assert not calls                                           # HDR floats: numpy code


def test_noise_exact(F):
    rng = np.random.default_rng(4)
    from types import SimpleNamespace
    for i in range(8):
        a = edge_rgba(rng, (int(rng.integers(1, 120)), int(rng.integers(1, 120)), 4))
        c = SimpleNamespace(gain=lambda fx, g=0.5 + i: g, p=SimpleNamespace(seed=i, intensity=i / 7), fi=i * 3)
        with numpy_only():
            ref = glitch.fx_noise(a, c)
        assert np.array_equal(glitch.fx_noise(a, c), ref), i


def _seq(tmp_path, n=16, w=160, h=120):
    return synth.make_frames(tmp_path / "f", n, w, h, move=4)


def test_time_effects_exact_frame_by_frame(F, tmp_path):
    """echo (_under) and mosh (replay, with 'bloom' gains) through glitch(), uint8 and float."""
    from PIL import Image
    paths = _seq(tmp_path)
    for is_float in (False, True):
        imgs = [np.asarray(Image.open(p).convert("RGBA"), np.float32) if is_float else Image.open(p).convert("RGBA")
                for p in paths]
        for fi in range(6, len(imgs)):
            hist = imgs[fi - 1::-1]
            kw = dict(t=fi / 10, frame=fi, seed=11, intensity=0.95, fps=10,
                      gains={"echo": 3, "mosh": 3, "noise": 2}, history=hist)
            with numpy_only():
                ref = np.asarray(glitch.glitch(imgs[fi], **kw))
            got = np.asarray(glitch.glitch(imgs[fi], **kw))
            assert np.array_equal(got, ref), (is_float, fi)


def render(tmp_path, name, opts, numba: bool, jobs=2):
    env = dict(os.environ, GLITCHER_NUMBA="1" if numba else "0")
    out = tmp_path / name
    r = subprocess.run([sys.executable, str(ROOT / "glitch.py"), str(tmp_path / "f"), "-o", str(out),
                        "-j", str(jobs), *opts], env=env, capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stderr[-2000:]
    assert ("render code: numba" in r.stderr) == numba, r.stderr[-500:]
    return out


@pytest.mark.parametrize("opts", [
    ["-s", "5", "-i", "0.9", "--gain", "echo=3,mosh=3,stutter=2,noise=2", "--fps", "10"],
    ["-s", "6", "-i", "0.8", "--pitch", "12", "--roll", "4", "--gain", "echo=2,mosh=2", "--fps", "10"],
    ["-s", "7", "-i", "0.9", "--format", "exr", "--gain", "echo=3,mosh=3", "--fps", "10"],
], ids=["2d-png", "3d-layered", "exr-float"])
def test_whole_render_identical(F, tmp_path, opts):
    if "exr" in opts:
        pytest.importorskip("OpenEXR")
    _seq(tmp_path)
    a = render(tmp_path, "np", opts, numba=False)
    b = render(tmp_path, "nb", opts, numba=True)
    fa = sorted(p.name for p in a.iterdir() if p.is_file())
    assert fa and fa == sorted(p.name for p in b.iterdir() if p.is_file())
    for n in fa:
        if n.endswith(".png"):
            assert synth.same_images(a / n, b / n), n
        else:
            ra, rb = glitch._exr_rgba(a / n), glitch._exr_rgba(b / n)
            assert all(np.array_equal(x, y) for x, y in zip(ra, rb)), n


def test_disabled_and_status(tmp_path):
    code = ("import glitch, json; print(json.dumps(glitch.fast_status()))")
    r = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=dict(os.environ, GLITCHER_NUMBA="0"),
                       capture_output=True, text=True)
    st = json.loads(r.stdout)
    assert st["backend"] == "numpy" and "GLITCHER_NUMBA=0" in st["status"]


def test_memory_estimate_follows_the_backend(monkeypatch):
    monkeypatch.setenv("GLITCHER_NUMBA", "0")
    off = glitch.worker_bytes(1000)
    monkeypatch.delenv("GLITCHER_NUMBA")
    import importlib.util
    on = glitch.worker_bytes(1000)
    assert on - off == (glitch.NUMBA_WORKER_BYTES if importlib.util.find_spec("numba") else 0)


def test_cache_compiles_once(F, tmp_path):
    """First process compiles into the cache; the next one only loads (that's what keeps the
    render workers' start-up and memory low)."""
    env = dict(os.environ, NUMBA_CACHE_DIR=str(tmp_path / "cache"))
    env.pop("GLITCHER_NUMBA", None)
    code = "import glitch, json; print(json.dumps(glitch.fast_status()))"
    st = [json.loads(subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env, capture_output=True,
                                    text=True, timeout=300).stdout) for _ in range(2)]
    assert st[0]["backend"] == "numba" and st[0]["cache_misses"] > 0
    assert st[1]["from_cache"] is True and st[1]["cache_misses"] == 0
    assert any((tmp_path / "cache").rglob("*.nbi"))


def test_playback_workers_stay_on_numpy(F):
    glitch_gui = pytest.importorskip("glitch_gui")
    from concurrent.futures import ProcessPoolExecutor
    with glitch_gui._new_pool("play_pool") as ex:
        assert ex.submit(glitch.warm_fast).result(timeout=120).startswith("numpy")
    assert not glitch_gui.PLAYBACK_NUMBA
    with glitch_gui._new_pool("live_pool") as ex:
        assert isinstance(ex, ProcessPoolExecutor)
        assert ex.submit(glitch.warm_fast).result(timeout=300).startswith("numba")


def test_editor_process_never_loads_numba(F):
    """The editor (Tk) process only hands jobs to workers: opening it and rendering a preview
    must not import Numba there (it would cost the editor ~35 MB for nothing)."""
    pytest.importorskip("tkinter")
    code = ("import sys, glitch_gui, synth, tempfile, pathlib; d = pathlib.Path(tempfile.mkdtemp()); "
            "synth.make_frames(d, 6); rc = glitch_gui.self_test(str(d)); "
            "print('NUMBA_IN_EDITOR', 'numba' in sys.modules); sys.exit(rc)")
    r = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, timeout=300,
                       env=dict(os.environ, PYTHONPATH=os.pathsep.join([str(ROOT), str(ROOT / "tests")])))
    if r.returncode != 0 and "TclError" in r.stderr:
        skip_or_fail("GUI", "no display for Tk")
    assert r.returncode == 0, r.stderr[-1500:]
    assert "NUMBA_IN_EDITOR False" in r.stdout
    assert "renders with numba" in r.stdout
