"""Pose estimator: it starts, MediaPipe and the model work, and it fails cleanly without a person."""
import os
from pathlib import Path

import pytest

import estimate_pose
from conftest import skip_or_fail


def find_model():
    """The model CI downloaded (GLITCHER_POSE_MODEL), else one already on this machine. Never
    downloads: tests must not depend on the network."""
    env = os.environ.get("GLITCHER_POSE_MODEL")
    if env:
        return env
    for v in ("heavy", "full", "lite"):
        name = f"pose_landmarker_{v}.task"
        for p in (Path(estimate_pose.__file__).with_name(name),
                  Path.home() / ".cache" / "estimate_pose" / name):
            if p.is_file():
                return str(p)
    return None


def test_version():
    with pytest.raises(SystemExit) as ex:
        estimate_pose.main(["--version"])
    assert ex.value.code == 0


def test_needs_an_input():
    with pytest.raises(SystemExit) as ex:
        estimate_pose.main([])
    assert ex.value.code == 2


def test_pose_check():
    if not estimate_pose._have_mediapipe():
        skip_or_fail("POSE", "mediapipe is not installed")
    model = find_model()
    if not model:
        skip_or_fail("POSE", "no pose model here (set GLITCHER_POSE_MODEL)")
    assert estimate_pose.main(["--check", "--model", model]) == 0


def test_no_person_is_a_clean_error(frames, tmp_path, capsys):
    if not estimate_pose._have_mediapipe():
        skip_or_fail("POSE", "mediapipe is not installed")
    model = find_model()
    if not model:
        skip_or_fail("POSE", "no pose model here (set GLITCHER_POSE_MODEL)")
    with pytest.raises(SystemExit) as ex:
        estimate_pose.main([str(frames), "-o", str(tmp_path / "pose.csv"), "--model", model])
    assert ex.value.code == 2
    assert "no person detected" in capsys.readouterr().err


def test_broken_mediapipe_reports_the_real_error():
    """MediaPipe installed but failing to load must say why, not claim it isn't there."""
    if not estimate_pose._have_mediapipe():
        skip_or_fail("POSE", "mediapipe is not installed")
    import subprocess
    import sys
    code = ("import sys; sys.modules['cv2'] = None; sys.path.insert(0, sys.argv[1]); "
            "import estimate_pose; raise SystemExit(estimate_pose.self_check())")
    r = subprocess.run([sys.executable, "-c", code, str(Path(estimate_pose.__file__).parent)],
                       capture_output=True, text=True, stdin=subprocess.DEVNULL, timeout=120)
    assert r.returncode == 3
    assert "failed to load" in r.stderr and "cv2" in r.stderr


def test_packaged_app_cv2_placeholder():
    """The stand-in build_portable.py gives MediaPipe for OpenCV lets it import and run."""
    import build_portable
    import subprocess
    import sys
    code = (build_portable.CV2_PLACEHOLDER + "\nsys.path.insert(0, sys.argv[1])\n"
            "import mediapipe.tasks.python.vision  # imports cv2\n"
            "import cv2\n"
            "try:\n    cv2.circle\nexcept AttributeError as ex:\n    print(ex)\n")
    r = subprocess.run([sys.executable, "-c", code, str(Path(estimate_pose.__file__).parent)],
                       capture_output=True, text=True, stdin=subprocess.DEVNULL, timeout=120)
    if not estimate_pose._have_mediapipe():
        skip_or_fail("POSE", "mediapipe is not installed")
    assert r.returncode == 0, r.stderr[-500:]
    assert "OpenCV is not included" in r.stdout
