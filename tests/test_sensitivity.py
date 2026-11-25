"""Every render option must do something. Each one is changed alone, in a context where it
applies (--spill needs a plate, --lock needs a pose, ...); if the output doesn't change, the
option is dead or broken. Options that are *meant* not to change pixels (lossless EXR
compression, worker count) are checked the other way round: they must not change anything."""
from pathlib import Path

import numpy as np
import pytest

import glitch
import synth

N = 10


@pytest.fixture(scope="module")
def m(tmp_path_factory):
    d = tmp_path_factory.mktemp("sens")
    frames = d / "frames"
    synth.make_frames(frames, N, move=3.0)
    pose = d / "pose.csv"                          # pitch, roll, scale, and a wrap-around in yaw
    yaws = [150, 165, 178, -175, -160, -150, -140, -130, -120, -110]
    pose.write_text("frame,yaw,pitch,roll,scale\n" + "".join(
        f"{i},{y},{-10 + 3 * i},{2 * i - 8},{0.8 + 0.05 * i}\n" for i, y in enumerate(yaws)))
    curve = d / "curve.csv"
    curve.write_text("frame,intensity\n0,0.1\n9,1.0\n")
    return {"frames": frames, "still": sorted(frames.glob("*.png"))[3], "pose": pose, "curve": curve,
            "plate": synth.make_plate(d / "plate.png"), "lm": synth.make_landmarks(d / "lm.npz", N),
            "audio": synth.make_clicks_wav(d / "a.wav", clicks=(0.15, 0.45, 0.75), dur=1.1)}


BASE = ["-s", "11", "-i", "0.7", "--fps", "10"]

# option: (context options, changed options). Context is in both renders; only the change differs.
CASES = {
    "--intensity": ([], ["-i", "0.2"]),
    "--fps": ([], ["--fps", "40"]),
    "--tempo": ([], ["--tempo", "2.5"]),
    "--density": ([], ["--density", "2.5"]),
    "--motion": ([], ["--motion", "1"]),
    "--bursts": ([], ["--bursts", "1"]),
    "--follow": ([], ["--follow", "0.5"]),
    "--bbox": ([], ["--bbox", "10,10,80,120"]),
    "--offset": ([], ["--offset", "37"]),
    "--use-numbers": (["--offset", "5"], ["--use-numbers"]),
    "--pad": ([], ["--pad", "0.2"]),
    "--tint": ([], ["--tint", "0.6"]),
    "--bleed": ([], ["--bleed", "25"]),
    "--jpeg-quality": ([], ["--jpeg-quality", "5"]),
    "--skip": ([], ["--skip", "noise"]),
    "--gain": ([], ["--gain", "jpeg=3"]),
    "--plate": ([], ["PLATE"]),
    "--spill": (["PLATE"], ["--spill", "0.6"]),
    "--audio": ([], ["AUDIO"]),
    "--audio-start": (["AUDIO"], ["--audio-start", "0.3"]),
    "--audio-end": (["AUDIO"], ["--audio-end", "0.4"]),
    "--audio-strength": (["AUDIO"], ["--audio-strength", "0.2"]),
    "--audio-sens": (["AUDIO"], ["--audio-sens", "1"]),
    "--only": (["LM"], ["--only", "head"]),
    "--protect": (["LM"], ["--protect", "torso"]),
    "--target": (["LM"], ["--target", "jpeg=head"]),
    "--part-feather": (["LM", "--only", "head"], ["--part-feather", "3"]),
    "--curve": ([], ["CURVE"]),
    "--hits": ([], ["--hits", "4:3"]),
    "--hit-len": (["--hits", "4:3"], ["--hit-len", "2"]),
    "--appear": ([], ["--appear", "4"]),
    "--vanish": ([], ["--vanish", "5"]),
    "--trans-len": (["--appear", "2"], ["--trans-len", "0.3"]),
    "--pose": ([], ["POSE"]),
    "--pitch": (["POSE", "--no-pose-pitch"], ["--pitch", "25"]),
    "--roll": (["POSE", "--no-pose-roll"], ["--roll", "20"]),
    "--no-pose-pitch": (["POSE"], ["--no-pose-pitch"]),
    "--no-pose-roll": (["POSE"], ["--no-pose-roll"]),
    "--tilt": (["POSE"], ["--tilt", "0.9"]),
    "--pose-smooth": (["POSE"], ["--pose-smooth", "0.5"]),
    "--pose-unwrap": (["POSE", "--pose-smooth", "0.3"], ["--pose-unwrap"]),   # only matters when angles get blended
    "--lock": (["POSE"], ["--lock", "0.2"]),
    "--cam-dist": (["POSE"], ["--cam-dist", "1.2"]),
    "--no-pose-scale": (["POSE", "--follow", "0.3"], ["--no-pose-scale"]),
    "--depth": (["POSE"], ["--depth", "0.9"]),
    "--exr-space": (["--format", "exr"], ["--exr-space", "display"]),
    "--exr-alpha": (["--format", "exr"], ["--exr-alpha", "straight"]),
    "--exr-depth": (["--format", "exr"], ["--exr-depth", "float"]),
}
# options for a single still
STILL_CASES = {"--frames": (["-n", "3"], ["-n", "4"]), "--trim": ([], ["--trim"])}
# must NOT change the pixels
NEUTRAL = {"--jobs": ([], ["-j", "3"]),
           "--exr-compression piz": (["--format", "exr"], ["--exr-compression", "piz"]),
           "--exr-compression none": (["--format", "exr"], ["--exr-compression", "none"])}


def expand(opts, m):
    out = []
    for o in opts:
        out += {"PLATE": ["--plate", str(m["plate"])], "AUDIO": ["--audio", str(m["audio"]), "--audio-strength", "1"]
                if o == "AUDIO" else None, "LM": ["--landmarks", str(m["lm"])], "POSE": ["--pose", str(m["pose"])],
                "CURVE": ["--curve", str(m["curve"])]}.get(o) or [o]
    return out


def frames_of(src, out, argv):
    assert glitch.main([str(src), "-o", str(out), "-j", "1", *argv]) == 0
    res = []
    for p in sorted(Path(out).iterdir()):
        if p.is_file():
            img, _ = glitch.load_frame(p, {})
            res.append(np.asarray(img))
    return res


def same(a, b):
    return len(a) == len(b) and all(x.shape == y.shape and np.array_equal(x, y) for x, y in zip(a, b))


def test_every_render_option_is_covered():
    ap = glitch.build_parser()
    opts = {a.option_strings[-1] for a in ap._actions if a.option_strings}
    io_only = {"--help", "--version", "--output", "--seed", "--boxes", "--range", "--jobs", "--format",
               "--exr-compression", "--glitch-pass", "--landmarks", "--frames", "--trim"}
    missing = opts - io_only - set(CASES) - {"--intensity"}
    assert not missing, f"options without a sensitivity case: {sorted(missing)}"


@pytest.mark.parametrize("opt", list(CASES), ids=list(CASES))
def test_option_changes_the_output(opt, m, tmp_path):
    ctx, change = CASES[opt]
    a = frames_of(m["frames"], tmp_path / "a", BASE + expand(ctx, m))
    b = frames_of(m["frames"], tmp_path / "b", BASE + expand(ctx, m) + expand(change, m))
    assert not same(a, b), f"{opt} changed nothing (context: {ctx})"


@pytest.mark.parametrize("opt", list(STILL_CASES), ids=list(STILL_CASES))
def test_still_option_changes_the_output(opt, m, tmp_path):
    ctx, change = STILL_CASES[opt]
    a = frames_of(m["still"], tmp_path / "a", BASE + ctx)
    b = frames_of(m["still"], tmp_path / "b", BASE + ctx + change)
    assert not same(a, b), f"{opt} changed nothing"


@pytest.mark.parametrize("opt", list(NEUTRAL), ids=list(NEUTRAL))
def test_neutral_option_changes_nothing(opt, m, tmp_path):
    ctx, change = NEUTRAL[opt]
    a = frames_of(m["frames"], tmp_path / "a", BASE + ctx)
    b = frames_of(m["frames"], tmp_path / "b", BASE + ctx + change)
    assert same(a, b), f"{opt} changed the pixels"
