"""Features in combination. Every pair of feature settings is rendered together at least once
(all-pairs), and every combination must keep the renderer's promises:

  1. it renders, and the same seed gives the same pixels with 1 worker and with 3;
  2. a partial render (--range) equals the same frames of the full render;
  3. alpha is sane: opaque over a plate, empty before the subject appears;
  4. the glitch pass is black wherever the output equals the clean subject or background.

Bugs that live between two features (the 1-vs-4-worker transition bug lived between the
transitions and the time effects) are found here, without trying all 10 000+ combinations."""
import itertools
import random
from pathlib import Path

import numpy as np
import pytest

import glitch
import synth

N = 12
FACTORS = {
    "plate": [None, "plate"],
    "parts": [None, ["--only", "head,arms"], ["--protect", "face"]],
    "pose": [None, "pose"],
    "timefx": [None, ["--skip", "echo,stutter,mosh"], ["--gain", "echo=3,stutter=2,mosh=3"]],
    "trans": [None, ["--appear", "2:0.3", "--vanish", "9:0.3"]],
    "motion": [None, ["--motion", "0.7"]],
    "audio": [None, "audio"],
    "glitch_pass": [None, ["--glitch-pass"]],
    "format": [None, ["--format", "exr"]],
    "follow": [None, ["--follow", "0.3"]],
    "pad": [None, ["--pad", "0.15"]],
    "timing": [None, ["--hits", "4:2:0.2", "--bursts", "1", "--tempo", "1.5"]],
}


def all_pairs(factors, seed=1):
    """A small set of combinations in which every pair of values of every two factors occurs
    (greedy; deterministic)."""
    names = list(factors)
    want = {(a, i, b, j) for a, b in itertools.combinations(range(len(names)), 2)
            for i in range(len(factors[names[a]])) for j in range(len(factors[names[b]]))}
    rng, rows = random.Random(seed), []
    while want:
        best, best_cov = None, -1
        for _ in range(60):
            row = [rng.randrange(len(factors[n])) for n in names]
            cov = sum((a, row[a], b, row[b]) in want for a, b in itertools.combinations(range(len(names)), 2))
            if cov > best_cov:
                best, best_cov = row, cov
        rows.append(best)
        want -= {(a, best[a], b, best[b]) for a, b in itertools.combinations(range(len(names)), 2)}
    return [{n: factors[n][i] for n, i in zip(names, r)} for r in rows]


COMBOS = all_pairs(FACTORS)


@pytest.fixture(scope="module")
def material(tmp_path_factory, frames):
    d = tmp_path_factory.mktemp("material")
    pose = d / "pose.csv"
    pose.write_text("frame,yaw,pitch,roll\n" + "".join(f"{i},{-30 + 8 * i},{-5 + i},{2 * i - 6}\n"
                                                      for i in range(N)))
    return {"frames": frames, "plate": synth.make_plate(d / "plate.png"),
            "lm": synth.make_landmarks(d / "lm.npz", N), "pose": pose,
            "audio": synth.make_clicks_wav(d / "a.wav", clicks=(0.1, 0.5, 0.9), dur=1.2)}


def argv_for(combo, m):
    a = ["-s", "11", "-i", "0.8", "--fps", "10"]
    for key, v in combo.items():
        if v is None:
            continue
        if v == "plate":
            a += ["--plate", str(m["plate"]), "--spill", "0.1"]
        elif v == "pose":
            a += ["--pose", str(m["pose"])]
        elif v == "audio":
            a += ["--audio", str(m["audio"]), "--audio-strength", "1"]
        else:
            a += list(v)
        if key == "parts":
            a += ["--landmarks", str(m["lm"])]
    return a


def _load(p):
    img, _ = glitch.load_frame(p, {})
    return np.asarray(img)


def _render(src, out, argv, jobs):
    assert glitch.main([str(src), "-o", str(out), "-j", str(jobs), *argv]) == 0
    return sorted(p for p in Path(out).iterdir() if p.is_file())


def test_all_pairs_is_small_and_complete():
    names = list(FACTORS)
    for a, b in itertools.combinations(names, 2):
        seen = {(repr(c[a]), repr(c[b])) for c in COMBOS}
        assert len(seen) == len(FACTORS[a]) * len(FACTORS[b]), (a, b)
    assert len(COMBOS) < 30, len(COMBOS)            # vs 10 368 for every combination


@pytest.mark.parametrize("combo", COMBOS, ids=[f"c{i:02d}" for i in range(len(COMBOS))])
def test_combination_keeps_the_promises(combo, material, tmp_path):
    argv = argv_for(combo, material)
    one = _render(material["frames"], tmp_path / "j1", argv, 1)
    three = _render(material["frames"], tmp_path / "j3", argv, 3)
    assert [p.name for p in one] == [p.name for p in three] and len(one) == N
    for x, y in zip(one, three):                                    # 1. same pixels, any worker count
        assert np.array_equal(_load(x), _load(y)), f"{x.name} differs with 3 workers: {argv}"
    part = _render(material["frames"], tmp_path / "part", argv + ["--range", "3-8"], 2)
    full = {p.name: p for p in one}
    for p in part:                                                  # 2. partial == full
        assert np.array_equal(_load(p), _load(full[p.name])), f"{p.name}: --range differs: {argv}"
    alphas = [_load(p)[..., 3] for p in one]                        # 3. alpha
    if combo["plate"]:
        assert all(a.min() >= 254 for a in alphas), argv
    if combo["trans"] and not combo["plate"]:
        assert alphas[0].max() == 0 and alphas[1].max() == 0, "before --appear must be empty"
    if combo["glitch_pass"]:                                        # 4. pass is black on clean pixels
        srcs = sorted(material["frames"].glob("*.png"))
        for p, s in zip(one, srcs):
            mat = _load(tmp_path / "j1" / "glitch_pass" / p.name)
            mat = mat[..., 0] if mat.ndim == 3 else mat
            o = _load(p).astype(np.float64)
            if combo["format"] or combo["pad"] or combo["plate"]:
                continue                                            # EXR / padded / plate: covered elsewhere
            c = _load(s).astype(np.float64)
            same = ((o[..., :3] * o[..., 3:] / 255 == c[..., :3] * c[..., 3:] / 255).all(-1)
                    & (o[..., 3] == c[..., 3]))
            assert (mat[same] == 0).all(), f"{p.name}: matte on an unchanged pixel: {argv}"
