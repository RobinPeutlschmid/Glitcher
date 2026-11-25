"""The pose solver (landmarks -> yaw / pitch / roll / scale) on a synthetic skeleton turned by
known angles. MediaPipe isn't needed: the solver gets exactly what MediaPipe would report for a
person in that pose, so this checks the maths itself (the footage in the other tests has no
person, so this part was never run)."""
import math

import numpy as np
import pytest

import estimate_pose as E

SKELETON = {  # subject space, metres: x = subject's left, y = down, z = front
    "nose": (0, -0.65, 0.10), "ls": (0.18, -0.45, 0), "rs": (-0.18, -0.45, 0),
    "le": (0.25, -0.20, 0.02), "re": (-0.25, -0.20, 0.02), "lh": (0.10, 0, 0), "rh": (-0.10, 0, 0),
    "lk": (0.10, 0.45, 0.02), "rk": (-0.10, 0.45, 0.02), "la": (0.10, 0.85, 0), "ra": (-0.10, 0.85, 0)}


def rotation(yaw, pitch, roll):
    """Subject -> camera, R = Rroll . Rpitch . Ryaw (glitch.Space3D's convention)."""
    y, p, r = map(math.radians, (yaw, pitch, roll))
    Ry = np.array([[math.cos(y), 0, math.sin(y)], [0, 1, 0], [-math.sin(y), 0, math.cos(y)]])
    Rx = np.array([[1, 0, 0], [0, math.cos(p), math.sin(p)], [0, -math.sin(p), math.cos(p)]])
    Rz = np.array([[math.cos(r), -math.sin(r), 0], [math.sin(r), math.cos(r), 0], [0, 0, 1]])
    return Rz @ Rx @ Ry


def features(yaws, pitches, rolls, scales):
    """What MediaPipe reports for that pose: metric world landmarks (its z points away from the
    camera) and image landmarks (weak perspective, s pixels per metre)."""
    F = np.zeros((len(yaws), len(E.FEAT)))
    col = {n: i for i, n in enumerate(E.FEAT)}
    for f, (yw, pt, rl, s) in enumerate(zip(yaws, pitches, rolls, scales)):
        R = rotation(yw, pt, rl)
        F[f, col["ok"]] = 1
        for j, p in SKELETON.items():
            c = R @ np.array(p, float)
            F[f, col[f"wx_{j}"]], F[f, col[f"wy_{j}"]], F[f, col[f"wz_{j}"]] = c[0], c[1], -c[2]
            F[f, col[f"ix_{j}"]], F[f, col[f"iy_{j}"]] = 640 + s * c[0], 400 + s * c[1]
            F[f, col[f"v_{j}"]] = 1.0
    return F


def test_decompose_inverts_the_rotation():
    for yw, pt, rl in ((0, 0, 0), (40, 15, -8), (-120, -20, 12), (170, 5, 3)):
        R = rotation(yw, pt, rl)
        got = E.decompose(R[:, 0], R[:, 1])
        assert np.allclose(got, (yw, pt, rl), atol=1e-6), (yw, pt, rl, got)


def test_a_slow_turn_is_recovered():
    n, fps = 75, 25
    t = np.arange(n) / (n - 1)
    yaws = -60 + 180 * t                                  # turns from 60 deg left to 120 deg right
    pitches = np.full(n, 12.0)                             # camera a little above
    rolls = np.full(n, -6.0)
    scales = 300 * (1 + 0.4 * t)                           # walks 40 % closer
    out = E.estimate(features(yaws, pitches, rolls, scales), fps)
    mid = slice(10, n - 10)                                # smoothing settles at the ends
    assert np.abs(out["yaw"][mid] - yaws[mid]).max() < 8, out["yaw"][mid] - yaws[mid]
    assert np.abs(out["pitch"][mid] - pitches[mid]).max() < 6
    assert np.abs(out["roll"][mid] - rolls[mid]).max() < 3
    ratio = out["scale"] / np.median(out["scale"])         # relative size follows the approach
    want = scales / np.median(scales)
    assert np.abs(ratio[mid] - want[mid]).max() < 0.06


@pytest.mark.parametrize("yaw", [0, 30, 45, 60, 90, 135, 150, 180, -45, -90])
def test_held_poses(yaw):
    """A person who never faces the camera: the width cue used to be calibrated on the clip's
    widest view, so a shot held at 45 deg read as 31 deg."""
    n = 30
    out = E.estimate(features(np.full(n, float(yaw)), np.zeros(n), np.zeros(n), np.full(n, 280.0)), 25)
    err = (out["yaw"][5:-5] - yaw + 180) % 360 - 180      # angles compare modulo a full turn
    assert np.abs(err).max() < 2, (yaw, out["yaw"][5:-5])


def test_csv_round_trip(tmp_path):
    n = 20
    out = E.estimate(features(np.linspace(0, 40, n), np.full(n, 5.0), np.zeros(n), np.full(n, 300.0)), 25)
    p = tmp_path / "pose.csv"
    E.write_csv(p, np.arange(n), out, 25)
    import glitch
    pose = glitch.Pose.from_file(p, 25)
    yaw, pitch, roll = pose.angles(10)
    assert abs(yaw - out["yaw"][10]) < 0.05 and abs(pitch - out["pitch"][10]) < 0.05
