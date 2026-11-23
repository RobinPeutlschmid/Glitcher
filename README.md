# Glitcher

"Digital hallucination" glitch for transparent cut-outs and image sequences (PNG / EXR): a
deterministic, temporally coherent glitch you can re-render frame-exactly, with a GUI
editor, pose-aware 3D layout, body-part targeting, time effects, background-plate spill and
sound-reactive timing.

| File | What it is |
|---|---|
| `glitch.py` | the renderer and its command line (`python glitch.py --help`) |
| `glitch_gui.py` | the editor (`python glitch_gui.py [frames/]`) |
| `estimate_pose.py` | yaw / pitch / roll and body landmarks from footage, via MediaPipe |
| `build_portable.py` | packages everything as an app that needs no Python |
| `CHANGELOG.md` | what changed in each version |

## Run from source

Python 3.10 or newer (CI uses 3.12).

```
python -m pip install -r requirements.txt
python glitch_gui.py
```

Only NumPy and Pillow are required. miniaudio (MP3 + playback), OpenEXR and MediaPipe (pose)
are optional; without them those features say what is missing.

## Tests

```
python -m pip install -r requirements.txt -r requirements-dev.txt
python -m pytest tests
```

The tests make their own synthetic footage, audio and landmarks, so nothing large lives in the
repository. Missing optional packages skip their tests; the GUI test needs a display (on a
Linux server: `xvfb-run -a python -m pytest tests`). Pose tests need the model: set
`GLITCHER_POSE_MODEL=/path/to/pose_landmarker_heavy.task`, or have it in
`~/.cache/estimate_pose/` from an earlier run.

## Build the portable app yourself

```
python build_portable.py            # full build, pose estimation included
python build_portable.py --no-pose  # without MediaPipe: much smaller
```

Build on the system you want the app for (PyInstaller does not cross-compile). The result is
`dist/Glitcher-<version>-<system>-<cpu>.zip`.

## Automatic builds and releases (GitHub Actions)

`.github/workflows/build.yml` runs on every push and pull request:

1. **tests** on Windows, macOS (Apple Silicon and Intel) and Linux;
2. **builds** the portable app on each;
3. **checks the built app**, not only the source: `tests/smoke_frozen.py` renders with the
   packaged app and compares it pixel for pixel with the source code, opens the editor
   (`--self-test`), writes EXR, and runs pose estimation with the bundled model;
4. keeps the zips for 14 days under the run's **Artifacts**.

### Releasing a version

1. Set `__version__` in `glitch.py` and add a `## <version> — <date>` entry at the top of
   `CHANGELOG.md` (the tests check that the two agree).
2. Commit and push, wait for the run to go green.
3. Tag and push the tag:

   ```
   git tag v1.3.0
   git push origin v1.3.0
   ```

The run then also creates the GitHub release **Glitcher 1.3.0** with the four zips,
`SHA256SUMS.txt`, and that version's CHANGELOG entry as the release text. A tag that doesn't
match `__version__` fails within seconds, before anything is built.

| Zip | For |
|---|---|
| `…-windows-x64.zip` | Windows 10/11, 64-bit |
| `…-macos-arm64.zip` | Apple Silicon Macs (M1 and later) |
| `…-macos-x64-nopose.zip` | Intel Macs: everything except pose *estimation* (MediaPipe has no Intel-Mac build). Pose files made elsewhere still work. |
| `…-linux-x64.zip` | Linux with glibc 2.39+ (Ubuntu 24.04 or newer) |

Dependency versions for CI and releases are pinned in `constraints.txt`; change them on
purpose and let CI confirm.

### Running the downloaded app

The apps are not code-signed (that needs paid Apple / Microsoft certificates), so the first
start asks for confirmation:

- **Windows:** SmartScreen says "Windows protected your PC" → **More info** → **Run anyway**.
- **macOS:** the first open is blocked → **System Settings → Privacy & Security** → scroll down
  → **Open Anyway** for Glitcher. Alternatively, in Terminal:
  `xattr -dr com.apple.quarantine /path/to/Glitcher.app`.

### Gitea instead of GitHub

Gitea Actions reads the same workflow syntax, but has no hosted Windows or macOS machines:
you would register your own runner (act_runner) on a Windows PC and a Mac. The artifact
actions also need Gitea's forks of them. GitHub's hosted runners are the simpler route for
these builds.
