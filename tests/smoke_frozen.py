#!/usr/bin/env python3
"""Check a packaged Glitcher (the output of build_portable.py) really works, on the machine
that built it. Not a pytest file: CI runs it after the build.

    python tests/smoke_frozen.py dist/Glitcher/Glitcher[.exe]      [--pose]
    python tests/smoke_frozen.py dist/Glitcher.app/Contents/MacOS/Glitcher --pose

1. --glitch renders a synthetic sequence (time effects, audio, plate, body parts) with worker
   processes, and the frames are bit-identical to the same render from the source code here.
2. EXR output works (OpenEXR got bundled).
3. --self-test: the editor window opens and its first preview renders.
4. with --pose: --estimate-pose --check runs MediaPipe and finds the model inside the app,
   without downloading it.
Exit code 0 = all passed.
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent), str(HERE)]

import glitch  # noqa: E402
import synth  # noqa: E402


def run(cmd, timeout=600, env=None):
    t = time.time()
    r = subprocess.run(cmd, stdin=subprocess.DEVNULL, capture_output=True, text=True,
                       encoding="utf-8", errors="replace", timeout=timeout, env=env)
    print(f"$ {' '.join(map(str, cmd))}\n  exit {r.returncode} in {time.time() - t:.1f} s")
    for line in (r.stdout + r.stderr).strip().splitlines()[-8:]:
        print("  | " + line)
    return r


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("exe", type=Path, help="the packaged app's executable")
    ap.add_argument("--pose", action="store_true", help="the build includes pose estimation")
    args = ap.parse_args()
    exe = args.exe.resolve()
    if not exe.is_file():
        print(f"no such executable: {exe}", file=sys.stderr)
        return 2
    failures = []

    def check(ok, what):
        print(("PASS  " if ok else "FAIL  ") + what + "\n")
        if not ok:
            failures.append(what)

    work = Path(tempfile.mkdtemp(prefix="glitcher_smoke_"))
    frames = work / "frames"
    synth.make_frames(frames, 12)
    plate = synth.make_plate(work / "plate.png")
    lm = synth.make_landmarks(work / "lm.npz", 12)
    wav = synth.make_clicks_wav(work / "clicks.wav")
    opts = ["-s", "7", "-i", "0.9", "--gain", "echo=2,stutter=2,mosh=2", "--fps", "5",
            "--audio", str(wav), "--plate", str(plate), "--spill", "0.05",
            "--protect", "legs", "--landmarks", str(lm)]

    r = run([exe, "--version"], timeout=120)
    check(r.returncode == 0 and (not r.stdout.strip() or glitch.__version__ in r.stdout),
          f"--version (expect {glitch.__version__})")

    r = run([exe, "--glitch", frames, "-o", work / "app", "-j", "2", *opts])
    ref = work / "ref"
    with open(os.devnull, "w") as null:
        so, sys.stdout = sys.stdout, null
        try:
            glitch.main([str(frames), "-o", str(ref), "-j", "1", *opts])
        finally:
            sys.stdout = so
    got = sorted((work / "app").glob("*.png"))
    same = len(got) == 12 and all(synth.same_images(f, ref / f.name) for f in got)
    check(r.returncode == 0 and same, "--glitch render is bit-identical to the source code's")

    r = run([exe, "--glitch", frames, "-o", work / "exr", "--format", "exr", "--range", "0-1", "-s", "3"])
    check(r.returncode == 0 and len(list((work / "exr").glob("*.exr"))) == 2, "EXR output")

    r = run([exe, "--self-test", frames], timeout=240)
    check(r.returncode == 0, "editor opens and renders its first preview (--self-test)")

    # DaVinci Resolve: a job renders headless, and the plugin installs from the app
    jd = work / "job"
    jd.mkdir()
    (jd / "preset.json").write_text(json.dumps({"seed": 5, "intensity": 0.7}))
    (jd / "job.json").write_text(json.dumps({
        "glitcher_job": 1, "input": str(frames), "range": [2, 5], "output": "out/v001", "plate": "",
        "fps": 25, "format": "exr", "exr_space": "display", "preset": "preset.json"}))
    r = run([exe, "--render-job", jd / "job.json", "-j", "2"])
    out = jd / "out" / "v001"
    check(r.returncode == 0 and len(list(out.glob("*.exr"))) == 4
          and len(list((out / "glitch_pass").glob("*.exr"))) == 4, "--render-job (Resolve plugin)")
    env = {**os.environ, "HOME": str(work), "USERPROFILE": str(work), "APPDATA": str(work)}
    r = run([exe, "--install-resolve", work / "Scripts" / "Utility"], timeout=120, env=env)
    cfg = work / ".glitcher_resolve.json"
    ok = (r.returncode == 0 and (work / "Scripts" / "Utility" / "Glitcher.py").is_file()
          and cfg.is_file() and json.loads(cfg.read_text())["glitcher"] == [str(exe)])
    check(ok, "--install-resolve copies the plugin and records this app")

    if args.pose:
        env = {**os.environ, "HOME": str(work), "USERPROFILE": str(work)}   # hide ~/.cache models
        r = run([exe, "--estimate-pose", "--check"], timeout=300, env=env)
        out = r.stdout + r.stderr
        bundled = "downloading" not in out        # a missing model would be fetched (and say so)
        check(r.returncode == 0 and bundled, "pose estimation works with the bundled model")

    print("all passed" if not failures else f"{len(failures)} FAILED: " + "; ".join(failures))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
