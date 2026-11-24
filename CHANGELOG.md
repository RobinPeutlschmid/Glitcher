# Glitcher changelog

Version numbers: MAJOR.MINOR.PATCH. The version lives in one place, `__version__` in glitch.py;
the GUI title, `--version` of every tool, saved presets and the portable build's zip name read it
from there.

## 1.5.0 — 2026-11-24

### New
- **DaVinci Resolve plugin** (Resolve Studio): Workspace > Scripts > Glitcher. A round trip for
  the clip under the playhead:
  - **Source:** the clip's own RGBA image sequence (only the frames the timeline uses), or the
    clip rendered from the timeline with its alpha, so a key, Magic Mask or Fusion comp made
    in Resolve is included. "Auto" picks for you and says why.
  - **Clean plate (optional):** the tracks below the clip are rendered as the plate, so the
    glitch spills into the real background.
  - **Render** headless from the panel (seed, intensity), or **Open in Glitcher editor** with
    the clip loaded; both use the same per-clip preset, so what you set in the editor is what
    the panel renders.
  - **Import:** the glitch and its glitch pass go into a "Glitcher" bin, the glitch onto a new
    track above the clip (same position; same framing for media files), the original is
    disabled. Its Fusion comp is wired from standard nodes: the pass drives GlitchGlow (through
    the Glow Mask input, so light comes from the glitch only but the halo spreads) and
    GlitchGrade (colour of the glitch only).
  - **Versions:** every render is a new version (v001, v002, ...); importing a new version
    disables the previous one and carries its Fusion comp over, your tweaks included.
  - **Colour:** EXR from Resolve is treated as display-encoded and files from other apps as
    scene-linear (or choose); the result is written back the same way, so it matches the
    clip it replaces.
  - Resolve's Deliver settings are saved before and restored after its renders, and track
    enables are put back.
- Editor button **Install DaVinci Resolve plugin** (or `Glitcher --install-resolve`): copies the
  script into Resolve's Scripts folder and records where this Glitcher is.
- `Glitcher --render-job JOB.json` renders a plugin job headless; `Glitcher --job JOB.json`
  opens it in the editor.

### Changed
- The editor's settings → command-line conversion is now one function shared by the editor,
  `--render-job` and the plugin (checked identical on 120 random editor states).

## 1.4.0 — 2026-11-23

### New
- **Glitch pass**: a matte of where the glitch is, written next to every rendered frame
  (`--glitch-pass`, or "Glitch pass (matte)" under Output format in the editor). Same file
  names, in `glitch_pass/` inside the output folder or a folder you name (`--glitch-pass DIR`).
  - White = the glitch made this pixel: displaced blocks, slices, smears, fills, RGB split,
    echoes, held frames, datamosh, transitions, spill into the plate. Black = the clean subject,
    the clean background (transparency or the plate), and hidden frames.
  - Grey levels say how strongly a pixel changed (linear up to 64 levels of 255), so the comp
    picks the threshold: a Levels / Threshold keeps just the hard glitches and drops subtle
    compression haze.
  - PNG renders: 8-bit greyscale PNG. EXR renders: EXR with R = G = B = A = matte, usable as
    luminance or alpha in Fusion / Nuke. Both give the same matte.
  - Frame-exact like the render: a `--range` re-render gives the same matte as a full one.
  - Refuses to write into the output folder itself (it would replace the frames) or over input
    or plate files; can't be combined with `--trim`.

## 1.3.1 — 2026-11-23

### Fixed
- macOS portable app: pose estimation failed to load ("recursion is detected during loading of
  cv2"): OpenCV's self-loader breaks inside a .app bundle. The portable app now leaves OpenCV out
  (MediaPipe imports it only for drawing helpers Glitcher never uses) and gives MediaPipe a
  placeholder module. This also makes the full build about half the size (Linux zip 272 → 137 MB).
- When MediaPipe is installed but fails to load, the pose estimator now shows the real error
  instead of claiming MediaPipe isn't there.
- CI: the version check ran `glitch.py` on a Python without NumPy and crashed; it now reads the
  version as text. Linux CI installs `libgles2`, which MediaPipe needs.

## 1.3.0 — 2026-11-23

### New
- **Undo / redo** in the GUI: Ctrl+Z undoes, Ctrl+Y or Ctrl+Shift+Z redoes (Cmd on macOS), and
  there are ↶ Undo / ↷ Redo buttons in the top bar. The status bar says what changed
  ("Undid: intensity (step 3 of 7)").
  - Covers every render setting: sliders, effect toggles, paths, timeline keys and hits,
    appear/vanish, body parts, plate, audio passage and sensitivity, and pose hints (the hints
    sidecar is rewritten and the pose re-solved).
  - Not steps: preview scale, compare view, preview background, audio sync — view settings that
    don't change the render.
  - Edits are merged until they settle for 0.4 s, so one slider drag is one step, not fifty.
  - Up to 200 steps are kept; a new edit after undoing drops the redo branch (as in most apps).
  - Cost: one small JSON snapshot per step (a few kB), nothing while idle.
- **Automatic builds and releases** (GitHub Actions, `.github/workflows/build.yml`): every push
  is tested on Windows, macOS (Apple Silicon and Intel) and Linux, the portable app is built on
  each and then checked as a built app; a `v<version>` tag publishes a GitHub release with the
  zips, checksums and this changelog entry. See README.md.
- **Test suite** (`tests/`, pytest): determinism, partial renders == full renders, serial ==
  parallel, overwrite protection, plate, appear, body parts, motion, EXR, audio decoding and beat
  detection, pose estimator, and a GUI run (preview == command line, undo, presets, folding,
  full render). Synthetic footage and audio are generated by the tests themselves.
- `estimate_pose.py --check` (also `Glitcher --estimate-pose --check`): tests that pose
  estimation works on this machine; exit code 0 = yes.
- `Glitcher --self-test FRAMES`: opens the editor, waits for the first preview, closes.

### Changed
- Portable zips are named with the CPU too: `Glitcher-1.3.0-windows-x64.zip`,
  `…-macos-arm64.zip`; `--no-pose` builds end in `-nopose`.
- `requirements.txt` (runtime) and `requirements-dev.txt` (tests, packaging); CI installs the
  exact versions in `constraints.txt`.

### Fixed
- Windows portable app: `Glitcher --glitch …` / `--estimate-pose …` started from a shortcut or
  script without a console could crash on its first progress message (a windowed app has no
  stdout/stderr). They now print to nowhere instead.
- Child processes (pose estimator, audio players, ffmpeg) get an explicit empty stdin, which a
  windowed Windows app needs to start them reliably.

## 1.2.1 — 2026-11-23

### Fixed
- Folding sections in a scrolled panel left the content drawn at the bottom under empty space
  (the old scroll position was kept and the scroll area never shrank). The panel now fits its
  content, returns to the top when everything fits, and hides the scrollbar when there is
  nothing to scroll.

## 1.2.0 — 2026-11-23

First versioned release. The project is now called **Glitcher** (the files keep their names:
glitch.py, glitch_gui.py, estimate_pose.py, build_portable.py).

### New
- **Sound-reactive glitch** (Audio tab, `glitch.py --audio`): load a WAV or MP3, pick the passage on
  the waveform (its start lines up with the first frame), set strength and sensitivity. Loudness
  scales the glitch; beats, found as spectral-flux onsets, force short bursts. The preview's
  Play follows the audio clock so picture and sound stay in sync. MP3/FLAC/OGG need the
  `miniaudio` package or ffmpeg; WAV needs nothing extra.
- **Collapsible sections**: click any section header to fold it (remembered between sessions);
  Ctrl+B hides the whole side panel for a full-width preview.
- `--version` on glitch.py, estimate_pose.py and the GUI/portable app; presets record the
  version that saved them.

### Changed
- App name in the window title, portable build (`Glitcher`, zip `Glitcher-1.2.0-<os>.zip`) and
  settings file (`~/.glitcher.json`; the old `~/.glitch_gui.json` is picked up once).

### Earlier (unversioned, now part of 1.2)
Fake-3D glitch with yaw/pitch/roll/distance, MediaPipe pose estimator with hints, Tkinter GUI,
16-bit half-float EXR, timing curves and hits, time effects (echo, frame hold, datamosh),
materialise/vanish transitions, body-part targeting, motion-reactive intensity, background plate
with spill, dark/light theme, Ctrl+S/Ctrl+O presets with autosave and paths, portable build, and
the bug/security sweep (output may never overwrite the input; private temp folder; https-only
model download; defaults 25 fps, padding 0, spill 0.05 everywhere).
