import multiprocessing
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import synth  # noqa: E402

# Most of this suite checks glitch.py's own renderer (exact pixels, Numba = NumPy, memory, the
# pool). It runs on that one unless a test asks for the native engine (test_app_native.py
# does); inherited by the subprocesses the tests start.
os.environ.setdefault("GLITCHER_ENGINE", "python")

# Windows and macOS start worker processes with "spawn"; use it everywhere so a pickling or
# import-order bug that only shows up there fails on Linux too.
multiprocessing.set_start_method("spawn", force=True)


def required(flag: str) -> bool:
    """CI sets GLITCHER_REQUIRE_<FLAG>=1 so a missing optional part fails instead of skipping."""
    return os.environ.get(f"GLITCHER_REQUIRE_{flag}", "") == "1"


def skip_or_fail(flag: str, why: str):
    if required(flag):
        pytest.fail(f"{why} (GLITCHER_REQUIRE_{flag}=1)")
    pytest.skip(why)


@pytest.fixture(scope="session")
def frames(tmp_path_factory) -> Path:
    d = tmp_path_factory.mktemp("frames")
    synth.make_frames(d, 12)
    return d


@pytest.fixture(scope="session")
def plate(tmp_path_factory) -> Path:
    return synth.make_plate(tmp_path_factory.mktemp("plate") / "plate.png")


@pytest.fixture(scope="session")
def landmarks(tmp_path_factory) -> Path:
    return synth.make_landmarks(tmp_path_factory.mktemp("lm") / "pose_landmarks.npz", 12)


@pytest.fixture(scope="session")
def clicks_wav(tmp_path_factory) -> Path:
    return synth.make_clicks_wav(tmp_path_factory.mktemp("audio") / "clicks.wav")
