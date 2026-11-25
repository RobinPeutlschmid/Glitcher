# Glitcher

[![build](https://github.com/YOUR-NAME/glitcher/actions/workflows/build.yml/badge.svg)](https://github.com/YOUR-NAME/glitcher/actions/workflows/build.yml)
[![latest release](https://img.shields.io/github/v/release/YOUR-NAME/glitcher)](https://github.com/YOUR-NAME/glitcher/releases/latest)

**"Digital hallucination" glitch for VFX: transparent cut-outs and image sequences in, temporally
coherent, re-renderable glitch out.**

Glitcher takes a keyed person (PNG with alpha, or EXR) and breaks them apart with JPEG rot,
block shifts, slices, pixel sorting, smears, RGB splits, echoes, frame holds and datamosh. It
does this in a way that holds up in a shot: the glitch is deterministic (same seed, same
pixels), follows the subject, can turn with them in fake 3D, and can be timed to keyframes, to
the subject's motion or to a soundtrack.

<!-- Add a screenshot or GIF here, e.g.:  ![Glitcher editor](docs/screenshot.png) -->

## Features

- **13 effects**, each with its own strength: `jpeg`, `blocks`, `slices`, `wave`, `sort`,
  `smear`, `split`, `tint`, `scanlines`, `noise`, plus the time effects `echo` (ghost trails),
  `stutter` (frame hold) and `mosh` (datamosh motion bleed).
- **Frame-exact and deterministic.** Rendering frames 40–60 alone gives exactly the same pixels
  as a full render, time effects included, so you can fix one section without re-rendering
  the whole shot.
- **Fake 3D.** The glitch volume turns with the subject (yaw / pitch / roll / camera distance).
  The angles come from `estimate_pose.py` (MediaPipe) or from a CSV.
- **Body-part targeting.** Glitch only the head, keep the face clean, or send datamosh to the
  arms only (`--only`, `--protect`, `--target`).
- **Timing.**
  - Keyframe curves and forced hits.
  - Materialise / dematerialise transitions.
  - Motion-reactive intensity: the glitch calms down when the subject holds still.
- **Sound-reactive.** Load a WAV or MP3 and pick the passage. Loudness drives the glitch and
  detected beats fire bursts.
- **Background plate.** Composite over the clean plate and let the glitch spill into it.
- **Glitch pass.** A matte of where the glitch is, rendered alongside every frame, so the
  comp can glow, grade or regrain just the glitch.
- **DaVinci Resolve plugin.** Send the clip under the playhead to Glitcher. The glitch, its
  pass and a ready Fusion comp come back.
- **Editor (GUI).**
  - Live preview, playback and a timeline.
  - A Pose tab with hint keyframes and an Audio tab.
  - Undo / redo.
  - Presets (Ctrl+S / Ctrl+O, with autosave).
  - Dark and light themes.
- **Pipeline-friendly.**
  - Linear or display EXR in and out, half or float, premultiplied or straight alpha.
  - The editor's **Copy CLI command** button gives the exact command line for what you see, for
    batch renders and render farms.

## Download

Get the app for your system from **[Releases](https://github.com/YOUR-NAME/glitcher/releases/latest)**.
It needs no Python: unzip it and start **Glitcher**.

| Zip | For |
|---|---|
| `Glitcher-<version>-windows-x64.zip` | Windows 10 / 11, 64-bit |
| `Glitcher-<version>-macos-arm64.zip` | Apple Silicon Macs (M1 and later) |
| `Glitcher-<version>-macos-x64-nopose.zip` | Intel Macs. Everything works except pose *estimation*, because MediaPipe has no Intel-Mac build. Pose files made elsewhere still work. |
| `Glitcher-<version>-linux-x64.zip` | Linux with glibc 2.39+ (Ubuntu 24.04 or newer) |

The apps aren't code-signed, so the first start asks you to confirm:

- **Windows:** "Windows protected your PC" → **More info** → **Run anyway**.
- **macOS:** the first open is blocked. Go to **System Settings → Privacy & Security**, scroll
  down and click **Open Anyway**. Or run
  `xattr -dr com.apple.quarantine /path/to/Glitcher.app` in Terminal.

`SHA256SUMS.txt` in each release lets you check that a download is intact.

## Run from source

Python 3.10+ (CI uses 3.12).

```bash
git clone https://github.com/YOUR-NAME/glitcher.git
cd glitcher
python -m pip install -r requirements.txt
python glitch_gui.py                 # the editor
python glitch_gui.py path/to/frames  # ...opened on a sequence
```

Only **NumPy** and **Pillow** are required. The rest is optional, and the app says what's
missing when you use a feature that needs it:

| Package | Adds |
|---|---|
| `miniaudio` | MP3 / FLAC / OGG audio and in-app playback (WAV works without it) |
| `OpenEXR` | EXR input and output |
| `mediapipe` | pose estimation (Pose tab, body-part landmarks) |
| `numba` | compiled versions of the slowest loops: renders ~1.3-1.7x faster, identical frames. The first run on a machine compiles them (~10 s) and caches them. `GLITCHER_NUMBA=0` turns them off. There's no Intel Mac build of Numba |

## DaVinci Resolve plugin

Glitch a clip without leaving Resolve (Resolve **Studio**: since Resolve 21.1, Python scripts
need Studio). You also need Python 3 installed on the machine, which Resolve uses to run the
plugin script; Glitcher itself can be the downloaded app.

**Install:** in the Glitcher editor, open *Presets & output* and click **Install DaVinci Resolve
plugin** (or run `Glitcher --install-resolve`). Restart Resolve.

**Use:** put the playhead over the clip, then **Workspace > Scripts > Glitcher**.

| Panel | What it does |
|---|---|
| Source | **Auto**: the clip's own RGBA image sequence if it has one, else the clip rendered from the timeline with its alpha (a key, Magic Mask or Fusion comp is then included). You can force either. |
| Clean plate | Also render the tracks *below* the clip as a plate: the glitch spills into the real background and comes back as the full picture. |
| Seed / Intensity | Saved in the clip's preset, the same one the editor uses. |
| Open in Glitcher editor | The full editor on this clip; every change is saved to the clip's preset. |
| Render | Renders a new version with Glitcher in the background, then imports it. |
| Import latest | Imports the newest complete version (e.g. one rendered in the editor). |

After import:
- the glitch is on a new **Glitcher** track above the clip, and the original is disabled;
- the glitch and its pass are in the **Glitcher** bin;
- the glitch clip's Fusion comp is ready:

```
MediaIn (glitch) ──> GlitchGlow ──> GlitchGrade ──> MediaOut
                        ▲ Glow Mask       ▲ Effect Mask
GlitchPass (Loader) ──> PassMatte ────────┘
```

Hover a node for a one-line reminder of what it does. Raise *Low* on PassMatte to keep only
the hard glitches. GlitchGlow makes light come from the
glitch only, and the halo can still spread beyond it. GlitchGrade colours the glitch only.

Each new version disables the previous one and copies its comp, tweaks included.

Each clip's work lives in a job folder under `Documents/Glitcher Resolve/<project>/`, holding
`job.json`, `preset.json`, the versions `out/v001`, `v002`, … and the logs. To put it on
another drive, set `"cache"` in `~/.glitcher_resolve.json`.

## Command line

```bash
# glitch a sequence, seed 7, a bit stronger than default
python glitch.py shots/sh010/ -o renders/sh010_glitch/ -s 7 -i 0.7

# re-render only frames 40-60; identical to the same frames of a full render
python glitch.py shots/sh010/ -o renders/sh010_glitch/ -s 7 -i 0.7 --range 40-60

# estimate the subject's turn, then glitch in 3D with the face kept clean
python estimate_pose.py shots/sh010/ -o sh010_pose.csv
python glitch.py shots/sh010/ -o out/ --pose sh010_pose.csv --protect face

# time it to music: the passage from 12.5 s lines up with the first frame
python glitch.py shots/sh010/ -o out/ --audio track.mp3 --audio-start 12.5 --audio-sens 0.6

# materialise at frame 0 over 1.5 s, vanish at 220; composite over the clean plate
python glitch.py shots/sh010/ -o out/ --appear 0:1.5 --vanish 220 --plate plates/sh010/

# also write the glitch matte (out/glitch_pass/, same file names) for the comp
python glitch.py shots/sh010/ -o out/ -s 7 --glitch-pass
```

`python glitch.py --help` lists every option. `estimate_pose.py --help` covers the pose
estimator. `estimate_pose.py --check` tests whether pose estimation works on this machine.

## Project layout

| Path | What it is |
|---|---|
| `glitch.py` | the renderer and its command line |
| `glitch_gui.py` | the editor |
| `glitch_numba.py` | optional compiled (Numba) versions of the hottest loops in glitch.py |
| `estimate_pose.py` | yaw / pitch / roll and body landmarks from footage (MediaPipe) |
| `build_portable.py` | packages everything as an app that needs no Python |
| `resolve/Glitcher.py` | the DaVinci Resolve plugin (installed into Resolve by the editor) |
| `tests/` | pytest suite; makes its own synthetic footage and audio |
| `docs/TESTING.md` | how it's tested, measurements, known gaps |
| `tools/memcheck.py` | memory over a long render, frame by frame |
| `tools/gpu_bench.py` | profiles a render and benchmarks numpy / Numba / GPU versions of the hot spots |
| `ci/release_notes.py` | release helper used by CI |
| `.github/workflows/build.yml` | tests, builds and releases on GitHub Actions |
| `CHANGELOG.md` | what changed in each version |

## Development

```bash
python -m pip install -r requirements.txt -r requirements-dev.txt
python -m pytest tests                # Linux server without a display: xvfb-run -a python -m pytest tests
```

- **Pose tests** need the model. Set `GLITCHER_POSE_MODEL=/path/to/pose_landmarker_heavy.task`,
  or keep it in `~/.cache/estimate_pose/` from an earlier run. Without it those tests are
  skipped.
- **Building the app yourself:** run `python build_portable.py` (or add `--no-pose` for a much
  smaller build without MediaPipe) on the system you're building for. PyInstaller can't
  cross-compile. The zip lands in `dist/`.

What the tests cover, the memory and performance measurements, and what is still not covered:
**[docs/TESTING.md](docs/TESTING.md)**. `tools/memcheck.py` measures memory over a long render.

**Found a Resolve bug?** In the Glitcher panel click **Check this Resolve** and attach the
`resolve_probe.json` it saves. Copied into `tests/resolve_contract/`, it makes the tests hold
the fake Resolve to how your Resolve really behaves.

### Continuous integration

Every push and pull request runs `.github/workflows/build.yml`:

1. **Tests** run on Windows, macOS (Apple Silicon and Intel) and Linux.
2. **The portable app** is built on each system.
3. **The built app is checked**, not just the source:
   - it renders a test sequence, which must be pixel-identical to the source code's render;
   - it opens the editor;
   - it writes EXR;
   - it estimates poses with the bundled model.
4. **The zips** are kept for 14 days under the run's *Artifacts*.

CI installs the exact versions in `constraints.txt`, so a release can't change just because
a dependency did. Update them deliberately and let CI confirm.

### Releasing

1. Set `__version__` in `glitch.py` and add a `## <version> — <date>` entry at the top of
   `CHANGELOG.md`. The tests check that the two match.
2. Commit, push, and wait for a green run.
3. Tag the release and push the tag:

   ```bash
   git tag v1.3.0
   git push origin v1.3.0
   ```

CI then publishes the **Glitcher 1.3.0** release with the four zips, `SHA256SUMS.txt`, and
that CHANGELOG entry as the release notes. A tag that doesn't match `__version__` fails in
seconds, before anything is built.

## Credits

Pose estimation uses Google's [MediaPipe](https://github.com/google-ai-edge/mediapipe) Pose
Landmarker. Audio decoding and playback use [miniaudio](https://github.com/irmen/pyminiaudio).
EXR support comes from the [OpenEXR](https://github.com/AcademySoftwareFoundation/openexr)
Python bindings. The compiled kernels use [Numba](https://numba.pydata.org/).
