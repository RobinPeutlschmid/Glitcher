"""Convert estimate_pose.py's <pose>_landmarks.npz to the .csv the Fusion node reads.

New runs of estimate_pose.py write both; this is for landmark files made before that.

    python tools/landmarks_csv.py shot_pose_landmarks.npz [more.npz ...]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from estimate_pose import write_landmarks_csv  # noqa: E402


def convert(path: Path) -> Path:
    with np.load(path, allow_pickle=False) as z:
        frames, lm = z["frames"], z["lm"]
        size = z["size"] if "size" in z.files else None
    if size is None:
        raise SystemExit(f"{path}: no image size stored (made by an old estimate_pose.py); run it again")
    out = path.with_suffix(".csv")
    write_landmarks_csv(out, frames, lm, size)
    return out


def main(argv=None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if not args:
        print(__doc__.strip())
        return 2
    for a in args:
        print(f"{a} -> {convert(Path(a))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
