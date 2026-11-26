"""glitch_native (ctypes) <-> the native engine, when it has been built (cd engine; cargo build
--release). The engine's own tests are Rust (cargo test); these check the Python side: the
premultiplied conversion, the needed-frames handshake and the C structs' layout."""
from __future__ import annotations

import numpy as np
import pytest

import glitch_native
import synth


@pytest.fixture(scope="module")
def eng():
    if glitch_native.library_path() is None:
        pytest.skip("native engine not built")
    return glitch_native.Engine()


def test_premultiplied_round_trip():
    rng = np.random.default_rng(1)
    a = rng.integers(0, 256, (40, 50, 4), dtype=np.uint8)
    a[..., 3][rng.random((40, 50)) < 0.3] = 0
    a[..., 3][rng.random((40, 50)) < 0.3] = 255
    back = glitch_native.from_premul(glitch_native.to_premul(a))
    vis = a[..., 3] >= 32                     # premultiplied 0..1 floats keep colour where it shows
    assert np.abs(back.astype(int) - a.astype(int))[vis].max() <= 1
    assert (back[a[..., 3] == 0] == 0).all()


def test_engine_renders_time_effects(eng, tmp_path):
    paths = synth.make_frames(tmp_path / "f", 40, 240, 160, move=4)
    from PIL import Image
    frames = [Image.open(p).convert("RGBA") for p in paths]
    p = eng.params(seed=2, intensity=1.0, bursts=0.0, gains={"echo": 3, "stutter": 3, "mosh": 3})
    assert eng.needed(p, 0, 240, 160, 0) == []                # nothing before the first frame
    changed = 0
    for i in range(40):
        need = eng.needed(p, i, 240, 160, min(i, 24))
        assert all(1 <= k <= min(i, 24) for k in need)
        out = eng.render(frames, i, seed=2, intensity=1.0, bursts=0.0, gains={"echo": 3, "stutter": 3, "mosh": 3})
        assert out.shape == (160, 240, 4) and out.dtype == np.uint8
        ref = glitch_native.from_premul(glitch_native.to_premul(frames[i]))
        changed += int((np.abs(out.astype(int) - ref.astype(int)) > 2).any())
    assert changed > 10


def test_abi_mismatch_is_refused(eng):
    p = eng.params()
    p.abi = 99
    cur = np.zeros((8, 8, 4), np.float32)
    with pytest.raises(RuntimeError, match="-3"):
        eng.render_premul(cur, {}, 0, p)


def test_struct_layout_matches_the_library(eng):
    import ctypes as C
    sz = (C.c_int32 * 3)()
    eng.lib.gc_sizes(sz)
    assert tuple(sz) == (C.sizeof(glitch_native.GcParams), C.sizeof(glitch_native.GcImage),
                         C.sizeof(glitch_native.GcStats))


def test_part_bits():
    assert glitch_native.part_bits("head,hands") == 2 | 16
    assert glitch_native.part_bits(("upper",)) == 2 | 4 | 8 | 16
    assert glitch_native.part_bits("") == 0
    with pytest.raises(ValueError):
        glitch_native.part_bits("tail")


def _figure(tmp_path, n=16):
    from PIL import Image
    paths, npz, csv = synth.make_figure(tmp_path / "fig", n, 160, 220)
    return [Image.open(p).convert("RGBA") for p in paths], npz, csv


def test_3d_parts_and_plate_through_the_binding(eng, tmp_path):
    import glitch
    frames, npz, csv = _figure(tmp_path)
    loud = dict(seed=3, intensity=1.0, bursts=0.0, gains={e: 2.0 for e in glitch.EFFECT_ORDER}, tint=0.3)
    # 3D (from a pose file) differs from flat, and the engine says it ran in 3D
    pose = tmp_path / "pose.csv"
    pose.write_text("frame,yaw,pitch,roll\n0,0,10,0\n15,80,10,12\n")
    differ = 0
    for i in range(16):
        a = eng.render(frames, i, **loud)
        b = eng.render(frames, i, mode3d=1, pose_path=pose, **loud)
        differ += int((a != b).any())
    assert differ > 10
    p = eng.params(mode3d=1, pose_path=pose, **loud)
    _, st = eng.render_premul(glitch_native.to_premul(frames[15]), {}, 15, p)
    assert st.flags & 1 and st.flags & 2 and st.warn == 0          # 3D, rolled (the file's roll)
    # protect the head: inside its mask the frame is the source
    lm = glitch.Landmarks(npz)
    for i in range(0, 16, 3):
        out = eng.render(frames, i, protect="head", landmarks_path=csv, **loud)
        head = glitch.part_mask(lm.at(i), ("head",), (220, 160)) >= 0.999
        src = glitch_native.from_premul(glitch_native.to_premul(frames[i]))
        assert (np.abs(out.astype(int) - src.astype(int))[head] <= 1).all()
    # a missing landmarks file: a warning, not an error
    p = eng.params(protect="head", landmarks_path=tmp_path / "nope.csv", **loud)
    _, st = eng.render_premul(glitch_native.to_premul(frames[3]), {}, 3, p)
    assert st.warn & 2 and "nope.csv" in eng.last_error()
    # a plate: opaque, and the clean plate far from the subject
    plate = np.zeros((220, 160, 4), np.uint8)
    plate[..., 0], plate[..., 1], plate[..., 2], plate[..., 3] = 40, 90, 160, 255
    for i in range(0, 16, 3):
        out = eng.render(frames, i, plate=plate, spill=0.05, **loud)
        assert (out[..., 3] == 255).all()
        assert (np.abs(out[:, :4].astype(int) - plate[:, :4].astype(int)) <= 1).all()   # left edge: far away


def test_landmarks_csv_matches_the_npz(tmp_path):
    import glitch
    sys_path_tools = __import__("sys").path
    _, npz, csv = synth.make_figure(tmp_path / "fig", 6, 120, 160)
    a, b = glitch.Landmarks(npz), glitch.Landmarks(csv)
    for f in (0, 2.5, 5, 9):
        assert np.abs(a.at(f) - b.at(f)).max() < 0.01
    # the converter writes the same file from an .npz
    sys_path_tools.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent / "tools"))
    import landmarks_csv
    out = landmarks_csv.convert(npz)
    assert out == npz.with_suffix(".csv")
    c = glitch.Landmarks(out)
    assert np.abs(a.at(3) - c.at(3)).max() < 0.01


def test_estimate_pose_writes_both_landmark_files(tmp_path):
    import estimate_pose
    import glitch
    lm = np.random.default_rng(2).uniform(10, 90, (4, 33, 3)).astype(np.float32)
    lm[2] = np.nan                                             # no person in frame 2
    p = tmp_path / "shot_pose_landmarks.npz"
    estimate_pose.write_landmarks(p, [10, 11, 12, 13], lm, (100, 120))
    assert p.is_file() and p.with_suffix(".csv").is_file()
    text = p.with_suffix(".csv").read_text()
    assert "size,100,120" in text and text.count("\n") == 3 + 4
    a, b = glitch.Landmarks(p), glitch.Landmarks(p.with_suffix(".csv"))
    assert np.abs(a.at(12) - b.at(12)).max() < 0.01
