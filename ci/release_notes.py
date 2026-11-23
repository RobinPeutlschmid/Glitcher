#!/usr/bin/env python3
"""Release helper for CI.

    python ci/release_notes.py --check-tag v1.3.0     fail unless the tag is v + glitch.__version__
                                                      and CHANGELOG.md has an entry for it
    python ci/release_notes.py > notes.md             that version's CHANGELOG section (release text)
    python ci/release_notes.py --print-version        just the version, e.g. 1.3.0

Reads glitch.py as text instead of importing it, so it runs on a bare Python with no
packages installed (importing glitch.py needs NumPy and Pillow).
"""
import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def version() -> str:
    m = re.search(r'^__version__\s*=\s*"([^"]+)"', (ROOT / "glitch.py").read_text("utf-8"), re.M)
    if not m:
        raise SystemExit("no __version__ in glitch.py")
    return m.group(1)


def section(v: str) -> str | None:
    text = (ROOT / "CHANGELOG.md").read_text("utf-8")
    m = re.search(rf"^## {re.escape(v)}\b[^\n]*\n(.*?)(?=^## |\Z)", text, re.M | re.S)
    return m.group(1).strip() if m else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check-tag", metavar="TAG")
    ap.add_argument("--print-version", action="store_true")
    a = ap.parse_args()
    v = version()
    if a.print_version:
        print(v)
        return 0
    if a.check_tag is not None:
        tag = a.check_tag.removeprefix("refs/tags/")
        if tag != f"v{v}":
            print(f"error: tag {tag} does not match glitch.__version__ {v} (expected v{v}). "
                  f"Bump __version__ and CHANGELOG.md, commit, then tag.", file=sys.stderr)
            return 1
        if section(v) is None:
            print(f"error: CHANGELOG.md has no '## {v}' entry", file=sys.stderr)
            return 1
        print(f"tag {tag} matches version {v}")
        return 0
    body = section(v)
    if body is None:
        print(f"error: CHANGELOG.md has no '## {v}' entry", file=sys.stderr)
        return 1
    sys.stdout.write(body + "\n\n---\nDownloads: pick the zip for your system. See README.md, "
                     "\"Running the downloaded app\", for the first-start security prompt on "
                     "Windows and macOS. SHA256SUMS.txt lists the checksums.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
