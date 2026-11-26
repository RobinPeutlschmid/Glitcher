# Glitcher

[![build](https://github.com/RobinPeutlschmid/Glitcher/actions/workflows/build.yml/badge.svg)](https://github.com/RobinPeutlschmid/Glitcher/actions/workflows/build.yml)
[![latest release](https://img.shields.io/github/v/release/RobinPeutlschmid/Glitcher)](https://github.com/RobinPeutlschmid/Glitcher/releases/latest)

**"Digital hallucination" glitch for VFX: transparent cut-outs and image sequences in, temporally
coherent, re-renderable glitch out.**

Glitcher takes a keyed person (PNG with alpha, or EXR) and breaks them apart with JPEG rot,
block shifts, slices, pixel sorting, smears, RGB splits, echoes, frame holds and datamosh. It
does this in a way that holds up in a shot: the glitch is deterministic (same seed, same
pixels), follows the subject, can turn with them in fake 3D, and can be timed to keyframes, to
the subject's motion or to a soundtrack.

<!-- Add a screenshot or GIF here, e.g.:  ![Glitcher editor](docs/screenshot.png) -->

## Features

- **Native engine (2.0).** Frames are rendered by `glitch_core`, a multithreaded Rust engine,
  in the editor's preview and playback, in renders and on the command line. Measured on 2
  cores: about 3x faster than the Python renderer at 4K, 2x at HD; more cores help it more.
  Without the engine Glitcher falls back to the Python renderer (see *Run from source*).
- **A Fusion node.** The same engine as a node on Resolve's Fusion page and in Fusion Studio
  (`fusion/`): live in the comp, with pose estimation, body parts, plates, hits, appear /
  vanish, a follow box, motion-reactive timing and a glitch matte output.
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

Get the app for your system from the **[latest release](https://github.com/RobinPeutlschmid/Glitcher/releases/latest)**
([all releases](https://github.com/RobinPeutlschmid/Glitcher/releases)). It needs no Python: unzip it and start **Glitcher**.

| Zip (on the [release page](https://github.com/RobinPeutlschmid/Glitcher/releases/latest)) | For |
|---|---|
| `Glitcher-<version>-windows-x64.zip` | Windows 10 / 11, 64-bit |
| `Glitcher-<version>-macos-arm64.zip` | Apple Silicon Macs (M1 and later) |
| `Glitcher-<version>-macos-x64-nopose.zip` | Intel Macs. Everything works except pose *estimation*, because MediaPipe has no Intel-Mac build. Pose files made elsewhere still work. |
| `Glitcher-<version>-linux-x64.zip` | Linux with glibc 2.39+ (Ubuntu 24.04 or newer) |
| `Glitcher-Fusion-<version>-windows-x64.zip` | the Fusion node for Resolve / Fusion Studio on Windows: `Glitcher.fuse` + `glitch_core.dll`; see [fusion/README.md](fusion/README.md) |
| `Glitcher-Fusion-<version>-macos-arm64.zip` | the same for Apple Silicon Macs (`libglitch_core.dylib`) |
| `Glitcher-Fusion-<version>-linux-x64.zip` | the same for Linux (`libglitch_core.so`) |

Builds of the newest code, before it's released: open the newest green run under
**[Actions → build](https://github.com/RobinPeutlschmid/Glitcher/actions/workflows/build.yml)** and download from its *Artifacts* (needs
a GitHub login; kept for 14 days).

The apps aren't code-signed, so the first start asks you to confirm:

- **Windows:** "Windows protected your PC" → **More info** → **Run anyway**.
- **macOS:** the first open is blocked. Go to **System Settings → Privacy & Security**, scroll
  down and click **Open Anyway**. Or run
  `xattr -dr com.apple.quarantine /path/to/Glitcher.app` in Terminal.

`SHA256SUMS.txt` in each release lets you check that a download is intact.

## Run from source

Python 3.10+ (CI uses 3.12). Numba needs a Python version it has a release for: check
[numba.pydata.org](https://numba.pydata.org/) if `pip` can't find one for a brand-new Python.

```bash
git clone https://github.com/RobinPeutlschmid/Glitcher.git
cd Glitcher
python -m pip install -r requirements.txt
python glitch_gui.py                 # the editor
python glitch_gui.py path/to/frames  # ...opened on a sequence
```

Needed: **NumPy** and **Pillow** (it won't run without them) and **Numba** (it runs without
it, but renders are much slower). `requirements.txt` installs all three. Numba compiles the
slowest loops to machine code. The frames are identical with and without it, so it only
changes speed: ~1.3-1.7x per frame at HD, more at 4K. The first run on a machine compiles
them (~10 s) and caches them on disk; later runs load them in under a second. Check it's
active with `python glitch_gui.py --numba-check`, or look for `render code: numba` in the
command line's output. There are no Numba builds for **Intel Macs** (since 0.63), so there
Glitcher renders with plain NumPy: the same frames, just slower. `GLITCHER_NUMBA=0` turns it
off anywhere.

**The native engine** (`engine/`, Rust) renders the frames when its library is there. Build it
once with [Rust](https://rustup.rs) installed:

```bash
cd engine && cargo build --release && cd ..
python glitch_gui.py --engine-check   # "native (glitch_core 2.0.0)" = the app renders with it
```

Without it everything still works on the Python renderer above, ~2-3x slower. The two
renderers make the same kind of glitch from the same settings, but not the same pixels (each
draws its own random events), so a shot started on one should be finished on the same one.
`GLITCHER_ENGINE=python` forces the Python renderer, `GLITCHER_ENGINE=native` refuses to run
without the engine; the command line prints which one it used (`render code: ...`).

The rest is optional, and the app says what's missing when you use a feature that needs it:

| Package | Adds |
|---|---|
| `miniaudio` | MP3 / FLAC / OGG audio and in-app playback (WAV works without it) |
| `OpenEXR` | EXR input and output |
| `mediapipe` | pose estimation (Pose tab, body-part landmarks) |

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
| `glitch.py` | the planner, the Python renderer and the command line |
| `glitch_native.py` | the native engine from Python: `glitch()` with glitch.py's arguments, rendered by `glitch_core` |
| `engine/` | `glitch_core`, the native engine (Rust, C interface) used by the app and the Fusion node |
| `fusion/` | the Fusion node (`Glitcher.fuse`), its docs and its tests against a stand-in of Fusion's API |
| `glitch_gui.py` | the editor |
| `glitch_numba.py` | optional compiled (Numba) versions of the hottest loops in glitch.py |
| `estimate_pose.py` | yaw / pitch / roll and body landmarks from footage (MediaPipe); writes `<pose>.csv`, `<pose>_landmarks.npz` and the same landmarks as `<pose>_landmarks.csv` (for the Fusion node) |
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
- **Engine tests:** `cd engine && cargo test --release`. The Fusion node runs against a
  stand-in of Fusion's API with LuaJIT: `luajit fusion/test_glitcher_mock.lua fusion/Glitcher.fuse
  engine/target/release/libglitch_core.so`.
- **Python tests** run on the Python renderer by default; `tests/test_app_native.py` covers
  the app on the native engine (skipped if it isn't built, unless `GLITCHER_REQUIRE_NATIVE=1`).
- **Building the app yourself:** run `python build_portable.py` (or add `--no-pose` for a much
  smaller build without MediaPipe) on the system you're building for. PyInstaller can't
  cross-compile. It bundles the engine, building it with cargo first if needed
  (`--no-engine` leaves it out). The zip lands in `dist/`.

What the tests cover, the memory and performance measurements, and what is still not covered:
**[docs/TESTING.md](docs/TESTING.md)**. `tools/memcheck.py` measures memory over a long render.

**Found a Resolve bug?** In the Glitcher panel click **Check this Resolve** and attach the
`resolve_probe.json` it saves. Copied into `tests/resolve_contract/`, it makes the tests hold
the fake Resolve to how your Resolve really behaves.

### Continuous integration

Every push and pull request runs `.github/workflows/build.yml`:

1. **Tests** run on Windows, macOS (Apple Silicon and Intel) and Linux, after building the
   native engine there. The engine's own tests and the Fusion node's run in a separate job,
   which also packages the node with its library per system.
2. **The portable app** is built on each system, with the engine inside.
3. **The built app is checked**, not just the source:
   - it renders a test sequence with the engine and with the Python renderer, each of which
     must be pixel-identical to the source code's render;
   - it opens the editor;
   - it writes EXR;
   - it estimates poses with the bundled model.
4. **The zips** are kept for 14 days under the run's *Artifacts* ([runs](https://github.com/RobinPeutlschmid/Glitcher/actions/workflows/build.yml)).

CI installs the exact versions in `constraints.txt`, so a release can't change just because
a dependency did. Update them deliberately and let CI confirm.

### Releasing

1. Set `__version__` in `glitch.py` and add a `## <version> — <date>` entry at the top of
   `CHANGELOG.md`. The tests check that the two match.
2. Commit, push, and wait for a green run.
3. Tag the release and push the tag:

   ```bash
   git tag v2.0.0
   git push origin v2.0.0
   ```

   `engine/Cargo.toml` and `resolve/Glitcher.py` carry the same version; the tests check both.

CI then publishes the **Glitcher 2.0.0** release on the [Releases page](https://github.com/RobinPeutlschmid/Glitcher/releases) with the app zips, the Fusion zips, `SHA256SUMS.txt`, and
that CHANGELOG entry as the release notes. A tag that doesn't match `__version__` fails in
seconds, before anything is built.

## Credits

Pose estimation uses Google's [MediaPipe](https://github.com/google-ai-edge/mediapipe) Pose
Landmarker. Audio decoding and playback use [miniaudio](https://github.com/irmen/pyminiaudio).
EXR support comes from the [OpenEXR](https://github.com/AcademySoftwareFoundation/openexr)
Python bindings. The compiled kernels use [Numba](https://numba.pydata.org/).
