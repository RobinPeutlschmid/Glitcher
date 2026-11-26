#!/usr/bin/env python3
"""Side by side: glitch.py vs the native engine (glitch_core), for the effects ported so far.

The two use different random draws (numpy vs hash), so the same seed gives *different* events:
frames don't match one to one. What should match is the look and the statistics - how often
each effect shows, how much of the subject it touches, how strongly. This prints those per
effect, times both, and writes a contact sheet (top: glitch.py, bottom: native) of the frames
where each effect is most visible.

    python tools/compare_native.py [FRAMES_DIR] [-n 150] [--out compare_native.png]
    python tools/compare_native.py --scene 3d|roll|parts|plate      (a synthetic figure)

Scenes: 2d (default), 3d (the figure turns 0 -> 90 deg, camera a little above), roll (turned,
rolled 20 deg), parts (head protected: both must leave it untouched), plate (on a background,
spill 0.15: both must leave the plate clean beyond the spill area).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "tests")]
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

import glitch  # noqa: E402
import glitch_native  # noqa: E402

PORTED = tuple(glitch.EFFECT_ORDER)
TINT = 0.5                      # the tint effect only shows with a tint amount


class Scene:
    """How each frame is rendered in a comparison: extra options for glitch.py and the engine,
    the clean reference picture, and what glitch.py's history frames are."""

    def __init__(self, name, frames, n_frames, lm_npz=None, lm_csv=None, plate=None):
        self.name, self.frames, self.n = name, frames, n_frames
        self.lm = glitch.Landmarks(lm_npz) if lm_npz else None
        self.lm_csv, self.plate = lm_csv, plate

    def angles(self, i):
        if self.name == "3d":
            return 90.0 * i / max(1, self.n - 1), 10.0, 0.0
        if self.name == "roll":
            return 30.0, 5.0, 20.0
        return None

    def py_kw(self, i):
        kw = {}
        ang = self.angles(i)
        if ang:
            kw.update(yaw=ang[0], pitch=ang[1], roll=ang[2])
        if self.name == "parts":
            kw.update(landmarks=self.lm.at(i), protect=("head",))
        if self.name == "plate":
            kw.update(plate=self.plate, spill=0.15)
        return kw

    def nat_kw(self, i):
        kw = {}
        ang = self.angles(i)
        if ang:
            kw.update(mode3d=1, yaw=ang[0], pitch=ang[1], roll=ang[2])
        if self.name == "parts":
            kw.update(landmarks_path=self.lm_csv, protect="head")
        if self.name == "plate":
            kw.update(plate=self.plate, spill=0.15)
        return kw

    def ref(self, i):
        f = self.frames[i]
        return np.asarray(glitch.composite_on(f, self.plate) if self.plate is not None else f)

    def py_hist(self, i):
        # with a plate glitch.py's history frames are composites (SeqHistory(plates=...))
        return [glitch.composite_on(self.frames[i - k], self.plate) if self.plate is not None else self.frames[i - k]
                for k in range(1, min(i, glitch.MAX_BACK) + 1)]


def py_render(sc: Scene, i, seed, intensity, fx, gain, fps):
    skip = tuple(e for e in glitch.EFFECT_ORDER if e != fx)
    hist = sc.py_hist(i) if fx in glitch.TIME_FX else []
    out = glitch.glitch(sc.frames[i], t=i / fps, frame=i, seed=seed, intensity=intensity, fps=fps,
                        gains={fx: gain}, skip=skip, history=hist or None, tint=TINT, **sc.py_kw(i))
    return np.asarray(out)


def stats_of(frames, outs):
    """Share of frames the effect changed, and on those: share of the subject's pixels changed
    and the mean colour change there (0..255)."""
    hit, cover, strength = 0, [], []
    for f, o in zip(frames, outs):
        a = np.asarray(f).astype(np.int16)
        a[a[..., 3] == 0] = 0                       # colour under alpha 0 isn't part of the picture
        o = o.astype(np.int16)
        o[o[..., 3] == 0] = 0
        d = np.abs(o - a).max(axis=2)
        ch = d > 2
        if ch.any():
            hit += 1
            subj = max(1, int((a[..., 3] > 0).sum()))
            cover.append(ch.sum() / subj)
            strength.append(float(d[ch].mean()))
    n = len(outs)
    return {"frames_hit": hit / n, "cover": float(np.mean(cover)) if cover else 0.0,
            "strength": float(np.mean(strength)) if strength else 0.0}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("frames", nargs="?")
    ap.add_argument("-n", type=int, default=150)
    ap.add_argument("--seeds", type=int, default=3, help="seeds per effect (default 3)")
    ap.add_argument("--intensity", type=float, default=0.8)
    ap.add_argument("--gain", type=float, default=2.0)
    ap.add_argument("--size", default="480x270")
    ap.add_argument("--out", default="compare_native.png")
    ap.add_argument("--only", help="comma-separated effects (default: all ported)")
    ap.add_argument("--scene", default="2d", choices=("2d", "3d", "roll", "parts", "plate"))
    a = ap.parse_args(argv)
    import synth
    import tempfile
    lm_npz = lm_csv = plate = None
    if a.frames:
        paths = glitch._gather(a.frames)[1][:a.n]
    elif a.scene == "2d":
        w, h = (int(v) for v in a.size.split("x"))
        paths = synth.make_frames(Path(tempfile.mkdtemp()) / "f", a.n, w, h, move=w / 160)
    else:
        paths, lm_npz, lm_csv = synth.make_figure(Path(tempfile.mkdtemp()) / "fig", a.n, 240, 320)
    frames = [Image.open(p).convert("RGBA") for p in paths]
    if a.scene == "plate":
        tmp = Path(tempfile.mkdtemp())
        plate = Image.open(synth.make_plate(tmp / "plate.png", frames[0].width, frames[0].height)).convert("RGBA")
    sc = Scene(a.scene, frames, len(frames), lm_npz, lm_csv, plate)
    refs = [sc.ref(i) for i in range(len(frames))]
    eng = glitch_native.Engine()
    fps = 25.0
    print(f"glitch_core {eng.version}, scene {a.scene}, {len(frames)} frames {frames[0].width}x{frames[0].height}, "
          f"intensity {a.intensity}, gain {a.gain}, {a.seeds} seeds per effect\n")
    heads = ([glitch.part_mask(sc.lm.at(i), ("head",), (frames[0].height, frames[0].width)) >= 0.999
              for i in range(len(frames))] if a.scene == "parts" else None)
    extra = {"parts": "head touched", "plate": "plate dmg / beyond"}.get(a.scene, "")
    print(f"{'effect':<9}{'engine':<9}{'frames hit':>11}{'subject cover':>15}{'strength':>10}{'ms/frame':>10}  {extra}")
    sheet_rows = []
    for fx in (a.only.split(",") if a.only else PORTED):
        res = {}
        best = {}
        for name in ("python", "native"):
            outs_all, frames_all, t_all = [], [], 0.0
            for seed in range(1, a.seeds + 1):
                t0 = time.perf_counter()
                if name == "python":
                    outs = [py_render(sc, i, seed, a.intensity, fx, a.gain, fps) for i in range(len(frames))]
                else:
                    outs = [eng.render(frames, i, seed=seed, intensity=a.intensity, fps=fps, tint=TINT,
                                       gains={**{e: 0.0 for e in glitch.EFFECT_ORDER}, fx: a.gain}, **sc.nat_kw(i))
                            for i in range(len(frames))]
                t_all += time.perf_counter() - t0
                outs_all += outs
                frames_all += refs
            res[name] = stats_of(frames_all, outs_all)
            note = ""
            if heads is not None:           # frames where the protected head changed (by > 2/255)
                hc = sum(int((np.abs(o.astype(int) - f.astype(int)).max(axis=2)[heads[i % len(frames)]] > 2).any())
                         for i, (f, o) in enumerate(zip(frames_all, outs_all)))
                note = f"  {hc} frames"
            if a.scene == "plate":          # plate pixels changed / changed beyond the spill area
                dm, far = 0, 0
                for i, (f, o) in enumerate(zip(frames_all, outs_all)):
                    cut = np.asarray(frames[i % len(frames)])
                    ch = np.abs(o.astype(int) - f.astype(int)).max(axis=2) > 2
                    bg = cut[..., 3] == 0
                    dm += int((ch & bg).sum())
                    box = cut[..., 3] > 0
                    ys, xs = np.nonzero(box)
                    reach = 0.15 * 0.5 * max(np.ptp(xs), np.ptp(ys)) + 16
                    yy, xx = np.mgrid[:cut.shape[0], :cut.shape[1]]
                    far_m = ((xx < xs.min() - reach) | (xx > xs.max() + reach)
                             | (yy < ys.min() - reach) | (yy > ys.max() + reach))
                    far += int((ch & far_m).sum())
                note = f"  {dm / len(outs_all):.0f} px / {far}"
            res[name]["ms"] = t_all / len(outs_all) * 1e3
            diffs = [np.abs(o[..., 3].astype(np.int16) - np.asarray(f)[..., 3].astype(np.int16)).sum()
                     + np.abs(o[..., :3].astype(np.int16) - np.asarray(f)[..., :3].astype(np.int16))[np.asarray(f)[..., 3] > 0].sum()
                     for f, o in zip(frames_all, outs_all)]
            best[name] = [outs_all[i] for i in np.argsort(diffs)[::-1][:4]]
            r = res[name]
            print(f"{fx:<9}{name:<9}{r['frames_hit']:>10.0%}{r['cover']:>14.0%}{r['strength']:>10.1f}{r['ms']:>10.1f}{note}")
        sheet_rows.append((fx, best))
    # contact sheet: per effect two rows (glitch.py, native) of its 4 most visible frames
    tw, th = frames[0].width // 2, frames[0].height // 2
    sheet = Image.new("RGB", (tw * 4 + 90, th * 2 * len(sheet_rows)), (24, 24, 28))
    d = ImageDraw.Draw(sheet)
    for r, (fx, best) in enumerate(sheet_rows):
        for k, name in enumerate(("python", "native")):
            y = (2 * r + k) * th
            d.text((6, y + th // 2 - 10), f"{fx}\n{name}", fill=(220, 220, 220))
            for j, o in enumerate(best[name]):
                im = Image.fromarray(o, "RGBA").resize((tw, th))
                bg = Image.new("RGBA", im.size, (60, 60, 66, 255))
                sheet.paste(Image.alpha_composite(bg, im).convert("RGB"), (90 + j * tw, y))
    sheet.save(a.out)
    print(f"\ncontact sheet: {a.out}")


if __name__ == "__main__":
    main()
