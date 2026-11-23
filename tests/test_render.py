"""The renderer: determinism, partial renders, safety, and each feature doing what it says."""
import hashlib
import re
from pathlib import Path

import numpy as np
import pytest

import glitch
import synth
from conftest import ROOT, skip_or_fail

TIME_FX = ["--gain", "echo=2,stutter=2,mosh=2", "-i", "0.9"]


def render(src, out, *extra, seed=7, jobs=1):
    rc = glitch.main([str(src), "-o", str(out), "-j", str(jobs), "-s", str(seed), *map(str, extra)])
    assert rc == 0
    return sorted(Path(out).glob("*.*"))


def test_version_matches_changelog():
    v = glitch.__version__
    assert re.fullmatch(r"\d+\.\d+\.\d+", v), v
    top = re.search(r"^## (\d+\.\d+\.\d+)", (ROOT / "CHANGELOG.md").read_text("utf-8"), re.M)
    assert top and top.group(1) == v, "the newest CHANGELOG entry must be the version in glitch.py"


def test_same_seed_same_pixels(frames, tmp_path):
    a = render(frames, tmp_path / "a", *TIME_FX)
    b = render(frames, tmp_path / "b", *TIME_FX)
    assert len(a) == 12 and [p.name for p in a] == [p.name for p in b]
    assert all(synth.same_images(x, y) for x, y in zip(a, b))
    c = render(frames, tmp_path / "c", *TIME_FX, seed=8)
    assert not all(synth.same_images(x, y) for x, y in zip(a, c)), "a new seed must change the glitch"


def test_parallel_render_matches_serial(frames, tmp_path):
    a = render(frames, tmp_path / "serial", *TIME_FX, jobs=1)
    b = render(frames, tmp_path / "parallel", *TIME_FX, jobs=2)
    assert all(synth.same_images(x, y) for x, y in zip(a, b))


@pytest.mark.parametrize("extra", [[], TIME_FX, [*TIME_FX, "--follow", "0.3", "--motion", "0.7"]],
                         ids=["plain", "time-effects", "follow-motion"])
def test_partial_render_matches_full(frames, tmp_path, extra):
    full = {p.name: p for p in render(frames, tmp_path / "full", *extra)}
    part = render(frames, tmp_path / "part", *extra, "--range", "4-8")
    assert [p.name for p in part] == [f"f_{i + 1:04d}.png" for i in range(4, 9)]
    assert all(synth.same_images(p, full[p.name]) for p in part)


def _digest(folder):
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(folder.iterdir())}


def test_refuses_to_overwrite_the_input(frames):
    before = _digest(frames)
    with pytest.raises(SystemExit) as ex:
        glitch.main([str(frames), "-o", str(frames), "-s", "1", "-j", "1"])
    assert ex.value.code == 2
    assert _digest(frames) == before


def test_refuses_to_overwrite_the_plate(frames, tmp_path):
    p = synth.make_plate(tmp_path / "plate.png")
    with pytest.raises(SystemExit):
        glitch.main([str(frames / "f_0001.png"), "-o", str(p), "--plate", str(p), "-s", "1"])


def test_bad_values_are_rejected(frames, tmp_path):
    for bad in (["--spill", "2"], ["--motion", "-1"], ["--appear", "3:0"], ["--audio-sens", "5"]):
        with pytest.raises(SystemExit):
            glitch.main([str(frames), "-o", str(tmp_path / "x"), *bad])


def test_still_renders_n_time_steps(frames, tmp_path):
    out = render(frames / "f_0001.png", tmp_path / "still", "-n", "3")
    assert [p.name for p in out] == ["f_0001_0001.png", "f_0001_0002.png", "f_0001_0003.png"]
    assert not synth.same_images(out[0], out[2]), "time steps of a still must differ"


def test_output_size_and_padding(frames, tmp_path):
    a = synth.load(render(frames, tmp_path / "nopad")[0])
    assert a.shape == (synth.H, synth.W, 4)                  # default pad 0: same size as input
    b = synth.load(render(frames, tmp_path / "pad", "--pad", "0.1")[0])
    assert b.shape[0] > synth.H and b.shape[1] > synth.W


def test_plate_gives_opaque_frames(frames, plate, tmp_path):
    out = render(frames, tmp_path / "plate", "--plate", plate, "--spill", "0.05")
    a = synth.load(out[0])
    assert a.shape[:2] == (synth.H, synth.W) and a[..., 3].min() == 255
    corner = a[:6, -6:, :3].astype(int)                      # far from the subject: the plate
    pl = synth.load(plate)[:6, -6:, :3].astype(int)
    assert np.abs(corner - pl).max() <= 2


def test_appear_hides_frames_before_it(frames, tmp_path):
    out = render(frames, tmp_path / "appear", "--appear", "3", "--trans-len", "0.04")
    alpha = [int(synth.load(p)[..., 3].max()) for p in out]
    assert alpha[:3] == [0, 0, 0] and min(alpha[3:]) == 255


def test_only_head_leaves_the_rest_untouched(frames, landmarks, tmp_path):
    out = render(frames, tmp_path / "head", "-i", "1", "--only", "head", "--landmarks", landmarks)
    for src, dst in zip(sorted(frames.glob("*.png")), out):
        d = np.abs(_premul(synth.load(src)) - _premul(synth.load(dst))).sum(-1)
        assert d[60:].sum() == 0, "legs and lower torso must not change with --only head"
        assert d[:40].sum() > 0, "the head must be glitched"


def _premul(a):
    """Colour under fully transparent pixels is free to change; compare what is visible."""
    a = a.astype(np.int64)
    return np.concatenate([a[..., :3] * a[..., 3:] // 255, a[..., 3:]], -1)


def test_body_part_names():
    assert glitch.parse_parts("head, hands") == ("head", "hands")
    with pytest.raises(ValueError):
        glitch.parse_parts("tail")


def test_motion_curve_reacts_to_movement(tmp_path):
    still = synth.make_frames(tmp_path / "still", 8, move=0.0)
    moving = synth.make_frames(tmp_path / "moving", 8, move=4.0)
    m0, m1 = glitch.motion_curve(still), glitch.motion_curve(moving)
    assert len(m0) == len(m1) == 8
    assert float(np.mean(m1)) > float(np.mean(m0))


def test_exr_output(frames, tmp_path):
    if not _has_openexr():
        skip_or_fail("EXR", "OpenEXR is not installed")
    out = render(frames, tmp_path / "exr", "--format", "exr", "--range", "0-1")
    assert [p.suffix for p in out] == [".exr", ".exr"]
    img, _raw = glitch.load_frame(out[0], {})
    assert img.shape[:2] == (synth.H, synth.W)


def _has_openexr():
    try:
        import OpenEXR  # noqa: F401
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------- glitch pass (matte)
def _pass(folder):
    return sorted((Path(folder) / "glitch_pass").glob("*.*"))


def test_glitch_pass_written_next_to_frames(frames, tmp_path):
    out = render(frames, tmp_path / "gp", *TIME_FX, "--glitch-pass")
    mats = _pass(tmp_path / "gp")
    assert [m.name for m in mats] == [o.name for o in out]
    for o, m in zip(out, mats):
        a, mt = synth.load(o), synth.load(m)
        assert mt.ndim == 2 and mt.shape == a.shape[:2]        # 8-bit greyscale, same size
        assert mt.max() > 0, "a strong glitch must show up in the matte"
        assert mt[a[..., 3] == 0].max() == 0, "transparent output pixels are never glitch"


def test_glitch_pass_is_black_without_glitch(frames, tmp_path):
    render(frames, tmp_path / "zero", "--skip", ",".join(glitch.EFFECT_ORDER), "--glitch-pass",
           "--range", "0-2")
    assert all(synth.load(m).max() == 0 for m in _pass(tmp_path / "zero"))


def test_glitch_pass_marks_only_changed_pixels(frames, tmp_path):
    out = render(frames, tmp_path / "gp", "-i", "0.8", "--glitch-pass")
    for src, o, m in zip(sorted(frames.glob("*.png")), out, _pass(tmp_path / "gp")):
        same = (_premul(synth.load(src)) == _premul(synth.load(o))).all(-1)
        assert synth.load(m)[same].max() == 0, "an unchanged pixel must be black in the matte"


def test_glitch_pass_hidden_frames_are_black(frames, tmp_path):
    render(frames, tmp_path / "ap", "--appear", "3", "--trans-len", "0.04", "--glitch-pass")
    mats = _pass(tmp_path / "ap")
    assert all(synth.load(m).max() == 0 for m in mats[:3])


def test_glitch_pass_partial_matches_full(frames, tmp_path):
    render(frames, tmp_path / "full", *TIME_FX, "--glitch-pass")
    render(frames, tmp_path / "part", *TIME_FX, "--glitch-pass", "--range", "4-8")
    full = {m.name: m for m in _pass(tmp_path / "full")}
    part = _pass(tmp_path / "part")
    assert len(part) == 5 and all(synth.same_images(m, full[m.name]) for m in part)


def test_glitch_pass_on_plate_leaves_the_clean_plate_black(frames, plate, tmp_path):
    render(frames, tmp_path / "pl", "--plate", plate, "--spill", "0.05", "-i", "0.8", "--glitch-pass")
    for m in _pass(tmp_path / "pl"):
        assert synth.load(m)[:6, -6:].max() == 0             # far corner: untouched plate


def test_glitch_pass_custom_folder_and_guards(frames, tmp_path):
    render(frames, tmp_path / "o", "--glitch-pass", tmp_path / "mattes", "--range", "0-1")
    assert len(list((tmp_path / "mattes").glob("*.png"))) == 2
    for bad in ([str(tmp_path / "o2")], [str(frames)]):       # = the output folder / the input
        with pytest.raises(SystemExit):
            glitch.main([str(frames), "-o", str(tmp_path / "o2"), "--glitch-pass", *bad])
    with pytest.raises(SystemExit):
        glitch.main([str(frames / "f_0001.png"), "-o", str(tmp_path / "t.png"), "--trim",
                     "--glitch-pass"])


def test_glitch_pass_exr(frames, tmp_path):
    if not _has_openexr():
        skip_or_fail("EXR", "OpenEXR is not installed")
    render(frames, tmp_path / "e", "--format", "exr", "-i", "0.8", "--glitch-pass", "--range", "3-3")
    render(frames, tmp_path / "p", "-i", "0.8", "--glitch-pass", "--range", "3-3")
    rgba, _ = glitch._exr_rgba(_pass(tmp_path / "e")[0])
    r, a = rgba[..., 0], rgba[..., 3]
    assert (r == a).all() and r.max() > 0
    png = synth.load(_pass(tmp_path / "p")[0]) / 255.0
    assert np.abs(r - png).max() < 2.5 / 255                  # same matte as the PNG render
