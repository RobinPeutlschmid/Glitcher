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
