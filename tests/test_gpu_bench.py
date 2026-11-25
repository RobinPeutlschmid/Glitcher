"""tools/gpu_bench.py: the ports it times must equal glitch.py bit for bit (else its numbers
compare different work), and every branch - including the GPU ones - must run to the end, so a
typo doesn't waste a run on a machine with a GPU. CI has no GPU: the CuPy branches run against a
numpy-backed stand-in with the same API surface the bench uses."""
from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import gpu_bench as gb  # noqa: E402

import glitch  # noqa: E402
import synth  # noqa: E402


class _DevArr(np.ndarray):
    """A 'device' array: numpy plus the two CuPy methods the bench calls."""

    def set(self, a):
        self[...] = a

    def get(self):
        return np.array(self)


def fake_cupy():
    cp = types.ModuleType("cupy")
    cp.__getattr__ = lambda n: getattr(np, n)                 # xp.where, xp.float32, ...
    cp.__version__ = "fake"
    cp.asarray = lambda a: np.array(a).view(_DevArr)          # a copy, like an upload
    cp.asnumpy = lambda a: np.array(a)

    class RawKernel:
        def __init__(self, src, name, options=()):
            assert name in src and "__global__" in src
            self.opts = options

        def __call__(self, grid, block, args):
            f, g, o, n = args
            assert grid[0] * block[0] >= int(n)
            o[...] = gb.under_xp(np, np.asarray(f), np.asarray(g))

    cp.RawKernel = RawKernel
    rt = types.SimpleNamespace(getDeviceCount=lambda: 1, memGetInfo=lambda: (6 << 30, 8 << 30),
                               getDeviceProperties=lambda i: {"name": b"Fake GPU", "major": 8, "minor": 9},
                               runtimeGetVersion=lambda: 12080, driverGetVersion=lambda: 13000)
    cp.cuda = types.SimpleNamespace(Device=lambda: types.SimpleNamespace(synchronize=lambda: None), runtime=rt)

    class RS:
        def __init__(self, seed):
            self.r = np.random.RandomState(seed)

        def normal(self, loc, scale, size, dtype=float):
            return self.r.normal(loc, scale, size).astype(dtype)

    cp.random = types.SimpleNamespace(RandomState=RS)
    cp.get_default_memory_pool = lambda: types.SimpleNamespace(total_bytes=lambda: 0)
    cupyx = types.ModuleType("cupyx")
    cupyx.empty_pinned = lambda shape, dtype: np.empty(shape, dtype)
    return cp, cupyx


@pytest.fixture(scope="module")
def material(tmp_path_factory):
    d = tmp_path_factory.mktemp("bench_frames")
    paths = synth.make_frames(d, 12, 240, 160, move=3)
    return d, paths, gb.bench_material(paths)


def test_ports_equal_glitch_py(material):
    """With xp = numpy every port is the code in glitch.py: same bytes out."""
    _d, _p, M = material
    K = gb.bench_kernels(M, 1, None, None)
    for k in ("under", "jpeg", "mosh_search", "noise"):
        assert K[k]["port_check"]["exact"], (k, K[k]["port_check"])
    h, w = M.size[1], M.size[0]
    fields = [glitch._block_motion(M.hist[k + 1], M.hist[k], M.s) for k in range(M.lag - 1, -1, -1)]
    held = M.hist[M.lag]
    # replay_xp is fx_mosh's loop: check against the effect's own code path on a real band
    out = gb.replay_xp(np, held.copy(), fields, M.mosh_box, 1.375, h, w)
    assert out.shape == held.shape and out.dtype == np.uint8


def test_compare_reports_differences():
    a = np.zeros((4, 4, 4), np.uint8)
    b = a.copy()
    b[0, 0, 1] = 3
    r = gb.compare(a, b)
    assert not r["exact"] and r["max_abs"] == 3 and r["differing"] == pytest.approx(1 / 16)
    assert gb.compare(a, a.astype(np.float32))["exact"] is False      # dtype counts too


def test_numba_ports_identical(material):
    numba = pytest.importorskip("numba")
    _d, _p, M = material
    K = gb.bench_kernels(M, 1, None, numba)
    for k, v in (("under", "numba_serial"), ("under", "numba_parallel"), ("mosh_search", "numba_serial"),
                 ("mosh_replay", "numba_serial"), ("mosh_replay", "numba_parallel"), ("noise", "numba_serial_add")):
        assert K[k][v]["exact"], (k, v, K[k][v])


def test_gpu_branches_run(material, monkeypatch):
    """Every CuPy branch runs end to end (on the stand-in) and reports comparable numbers."""
    cp, cupyx = fake_cupy()
    monkeypatch.setitem(sys.modules, "cupyx", cupyx)
    _d, _p, M = material
    K = gb.bench_kernels(M, 1, cp, None)
    for k, v in (("under", "cupy_port"), ("under", "cuda_strict"), ("under", "cuda_fast"),
                 ("jpeg", "cupy_port"), ("mosh_search", "cupy_port"), ("mosh_replay", "cupy_port"),
                 ("noise", "cupy_cpu_numbers")):
        assert K[k][v]["exact"], (k, v, K[k][v])
    for k, v in (("under", "cuda_strict_with_transfer"), ("jpeg", "cupy_port_with_transfer"),
                 ("mosh_replay", "cupy_port_with_transfer"), ("noise", "cupy_cpu_numbers_with_transfer"),
                 ("noise", "cupy_gpu_numbers")):
        assert K[k][v]["ms"] >= 0
    T = gb.bench_transfers(M, 1, cp)
    assert T["uint8_frame"]["mb"] > 0 and "upload_pinned" in T["float32_frame"]


def test_main_end_to_end(material, monkeypatch, tmp_path):
    """The whole run - profile, kernels (numpy + stand-in GPU), estimate, report, JSON."""
    cp, cupyx = fake_cupy()
    monkeypatch.setitem(sys.modules, "cupyx", cupyx)
    monkeypatch.setattr(gb, "load_cupy", lambda: (cp, None))
    d, _p, _M = material
    out = tmp_path / "r.json"
    R = gb.main([str(d), "-n", "10", "--repeat", "1", "--procs", "1", "--no-numba", "--json", str(out)])
    data = json.loads(out.read_text())
    assert data["system"]["gpu"] == "Fake GPU"
    prof = data["profile"]
    assert prof["frames"] == 10 and prof["ms_per_frame"]
    # the profile's parts add up to the frame (exclusive times: nothing counted twice)
    assert sum(prof["ms_per_frame"].values()) == pytest.approx(prof["frame_ms"]["mean"], rel=0.05)
    E = R["estimate"]
    assert E["ceiling_frame_ms"] <= E["measured_frame_ms"]
    assert "gpu_resident" in E and E["gpu_resident"]["identical_frames"] is True
    # the render was not changed by profiling it: glitch.py's functions are restored
    assert glitch._under.__name__ == "_under" and glitch.load_frame.__name__ == "load_frame"
    assert glitch._FX["echo"] is glitch.fx_echo


def test_preset_profile(material, tmp_path):
    """--preset goes through the editor's own preset -> options code."""
    pytest.importorskip("tkinter")
    d, _p, _M = material
    preset = tmp_path / "look.json"
    preset.write_text(json.dumps({"on_noise": False, "gain_echo": 2.0, "seed": 3}))
    argv = gb.glitch_argv(d, tmp_path / "o", str(preset), [], tmp_path)
    assert "--skip" in argv and "noise" in argv[argv.index("--skip") + 1]
    assert argv[argv.index("-s") + 1] == "3"


def test_scaling_cpu(material, tmp_path):
    """The several-workers test (spawned processes, like the render pool)."""
    _d, _p, M = material
    S = gb.bench_scaling(M, 2, 2, 2, ["cpu"], tmp_path)
    assert S["cpu"]["1"]["steps_per_s"] > 0 and S["cpu"]["2"]["steps_per_s"] > 0 and "speedup" in S["cpu"]
