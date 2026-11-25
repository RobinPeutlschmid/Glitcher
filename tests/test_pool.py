"""The render pool: sized to memory, and a worker the system kills doesn't fail the render."""
import os
import threading
from pathlib import Path

import pytest

import glitch


def memory_chunk(jobs):
    """Stand-in worker that runs out of memory (a normal MemoryError, like Windows gives)."""
    for name, mode, marker in jobs:
        if mode == "oom-once" and not Path(marker).exists():
            Path(marker).write_text("x")
            raise MemoryError("Unable to allocate 31.6 MiB for an array")
        if mode == "oom":
            raise MemoryError("Unable to allocate 31.6 MiB for an array")
    return [j[0] for j in jobs]


def flaky_chunk(jobs):
    """Stand-in worker: ("die-once", marker) kills its process the first time (like the system
    killing a worker that ran out of memory), ("die", _) every time (a real crash)."""
    out = []
    for name, mode, marker in jobs:
        if mode == "die" or (mode == "die-once" and not Path(marker).exists()):
            if marker:
                Path(marker).write_text("x")
            os._exit(1)
        out.append(name)
    return out


def jobs(tmp_path, special=None):
    return [(f"f_{i:04d}.exr", (special or {}).get(i, "ok"), str(tmp_path / f"m{i}")) for i in range(12)]


def test_memory_caps_the_workers(monkeypatch):
    monkeypatch.setattr(glitch, "_numba_expected", lambda: False)        # numpy workers
    four_k = 3840 * 2160
    assert glitch.safe_workers(16, four_k, avail=16_000_000_000) == 6     # ~1.9 GB each
    assert glitch.safe_workers(16, 1920 * 1080, avail=16_000_000_000) == 14  # ~0.8 GB each
    assert glitch.safe_workers(16, 1920 * 1080, avail=8_000_000_000) == 7
    assert glitch.safe_workers(16, four_k, avail=1_000_000_000) == 1       # never 0
    assert glitch.safe_workers(4, four_k, avail=0) == 4                    # unknown: as asked
    assert glitch.available_memory() is None or glitch.available_memory() > 0
    monkeypatch.setattr(glitch, "_numba_expected", lambda: True)         # + Numba in each worker
    assert glitch.worker_bytes(1920 * 1080) == 450_000_000 + 170 * 1920 * 1080 + glitch.NUMBA_WORKER_BYTES
    assert glitch.safe_workers(16, 1920 * 1080, avail=16_000_000_000) == 13


def test_a_killed_worker_is_retried(tmp_path):
    notes, seen = [], []
    done = glitch.render_pool(jobs(tmp_path, {5: "die-once"}), 4, progress=seen.append,
                              log=notes.append, chunk_fn=flaky_chunk)
    assert done == 12 and seen[-1] == 12
    assert any("died" in n and "2 worker" in n for n in notes), notes


def test_a_real_crash_names_the_frame(tmp_path):
    notes = []
    with pytest.raises(glitch.RenderCrash, match="f_0007.exr"):
        glitch.render_pool(jobs(tmp_path, {7: "die"}), 4, log=notes.append, chunk_fn=flaky_chunk)
    assert "one frame at a time" in notes[-1]                  # it tried hard before giving up


def test_cancel_stops_early(tmp_path):
    ev = threading.Event()
    ev.set()
    assert glitch.render_pool(jobs(tmp_path), 2, cancel=ev, chunk_fn=flaky_chunk) < 12


def test_real_render_through_the_pool_matches_serial(frames, tmp_path):
    args = [str(frames), "-s", "7", "-i", "0.9", "--gain", "echo=2,stutter=2,mosh=2"]
    assert glitch.main(args + ["-o", str(tmp_path / "a"), "-j", "1"]) == 0
    assert glitch.main(args + ["-o", str(tmp_path / "b"), "-j", "3"]) == 0
    import synth
    for f in sorted((tmp_path / "a").glob("*.png")):
        assert synth.same_images(f, tmp_path / "b" / f.name)


def test_out_of_memory_in_a_worker_is_retried(tmp_path):
    """The error you saw: a worker raises MemoryError. Retry with fewer workers, not stop."""
    notes = []
    assert glitch.render_pool(jobs(tmp_path, {4: "oom-once"}), 4, log=notes.append,
                              chunk_fn=memory_chunk) == 12
    assert "ran out of memory" in notes[0]


def test_out_of_memory_even_alone_says_so(tmp_path):
    with pytest.raises(glitch.RenderCrash, match="not enough memory to render f_0004.exr"):
        glitch.render_pool(jobs(tmp_path, {4: "oom"}), 4, log=lambda m: None, chunk_fn=memory_chunk)
