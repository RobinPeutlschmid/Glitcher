# Glitcher changelog

Version numbers: MAJOR.MINOR.PATCH. The version lives in one place, `__version__` in glitch.py;
the GUI title, `--version` of every tool, saved presets and the portable build's zip name read it
from there.

## 1.7.0 — 2026-11-25

### Faster
- **Compiled kernels (Numba), with identical frames.** The three slowest loops are now
  compiled when Numba is installed: echo's compositing (`_under`, also used by mosh and by the
  3-D front/back layering), the noise grain, and the mosh replay. Each is the numpy code
  rewritten as one loop over the pixels, with the same float32 operations in the same order.
  The frames are byte-identical, and the tests compare them on edge-case pixels and whole
  renders (PNG, EXR, 2-D, 3-D). Estimated on a real preset at 720x480: 1.34x per frame. On
  synthetic 1080p frames: 1.3-1.7x.
  - The first run on a machine compiles the kernels (about 10 s) into a disk cache. After
    that they load in under a second. Renders compile once in the main process, so the
    workers only load. The editor starts this in its preview worker when it opens.
  - Memory: about 35 MB more per process that loads them. The editor's own process and its
    playback workers stay on numpy, so the editor doesn't get heavier. The live preview and
    the render workers use the kernels. The render worker memory estimate includes them.
  - `GLITCHER_NUMBA=0` turns them off. `Glitcher --numba-check` says which code runs. The
    CLI prints `render code: ...`.
  - The mosh motion search stays numpy on purpose: numpy's float32 summation order can't be
    reproduced exactly in a loop, and a last-bit difference could change a motion vector. It
    is about 1 % of a frame.
  - Packaging: the builds include Numba (+~50-60 MB zipped). The Intel Mac build doesn't,
    because Numba has no Intel Mac wheels since 0.63, so it renders with numpy, with the same
    frames. The frozen-app check now compares the app's compiled render with a numpy render
    from source, and checks that the disk cache works inside the app.
- `tools/gpu_bench.py` measures whether a GPU (CuPy) or compiled CPU code (Numba) would speed
  up renders on the machine it runs on. It profiles a real render, then benchmarks ports of
  the hot spots, each first checked bit for bit against glitch.py. Each variant then reports
  whether its frames come out identical. It also measures CUDA start-up, VRAM, transfer
  costs, and how throughput holds up with several workers. Writes `gpu_bench_result.json`.
  Glitcher itself doesn't use it.
  On an RTX 4060 laptop GPU, the GPU versions were fast, and their frames were identical.
  They still lost overall: uploading and downloading frames ran at ~1.3 GB/s, CUDA took
  4.7 s to start in each process, and the motion search was slower on the GPU (many small
  arrays). Numba came out ahead (1.34x against 1.22-1.27x), needs no particular GPU, and is
  bit-identical. That's why 1.7.0 uses Numba.

## 1.6.0 — 2026-11-25

### New
- Resolve panel: **Check this Resolve**. On a throwaway timeline it asks Resolve every question
  the plugin depends on: how sequences import, how clips are placed and trimmed, which inputs
  Fusion's nodes have, how renders are named. It saves the answers to `resolve_probe.json`
  and removes everything it made. Recorded reports go in `tests/resolve_contract/`, and the
  tests then fail if the fake Resolve they use answers differently from the real one.
- Testing: 157 tests, up from 73. See docs/TESTING.md.

### Fixed
- Pose estimation: a person who never faces the camera got the wrong yaw (held at 45 deg read
  as 31 deg, at 135 deg as 149 deg). One cue was calibrated on the clip's widest view, assuming
  a frontal moment. It now uses the body's own proportions from MediaPipe's metric skeleton.
  Held poses are now exact; on a clip that does face the camera, results move by ≤ 3 deg.

### Faster
- Echo / datamosh compositing blends whole regions in one pass: renders are ~8 % faster with
  default settings and ~10 % with heavy time effects, with bit-identical output.
- The command line (and the Resolve plugin) now uses every CPU by default instead of at most 8;
  the memory-aware pool lowers the count when RAM is short.

### Removed
- Dead code: an unused pixel helper, two attributes nothing read, and an unused plugin
  function.

## 1.5.3 — 2026-11-24

### Fixed
- Resolve plugin: Import latest still made a 1-frame clip (1.5.2 didn't fix it) and could jump
  the playhead to ~18 hours. The plugin no longer depends on how Resolve numbers frames:
  - Resolve's renders (named by timeline timecode, e.g. src_00090000.exr) are renamed to
    src_0001.exr, src_0002.exr, ... right after rendering;
  - after importing, the clip's frame count is checked; a wrong result (e.g. a single still)
    is removed and the frames imported again as a list of files;
  - the clip goes on the timeline whole (no start / end frame), checked for length and
    position; only if that fails are explicit frame ranges tried, each wrong try removed;
  - the playhead is put back where it was;
  - every step is logged in the panel and in `panel.log` in the Glitcher Resolve folder.

## 1.5.2 — 2026-11-24

### Fixed
- Resolve plugin: Import latest made a 1-frame, empty clip. The plugin asked Resolve for frames
  0..n-1 of the imported sequence, but Resolve numbers an image sequence by its file numbers
  (a render of the timeline is named after timeline frames, e.g. src_00086523.exr), so the
  request fell outside the clip. It now uses the sequence's own numbers and checks that the clip
  really is full length at the right spot. If not, it removes that placement and tries Resolve's
  other conventions. If nothing fits, it stops with a message instead of leaving a broken clip.
- A previous version's Fusion comp is only carried over if that version's clip was healthy.
- The comp no longer gets a note that covered PassMatte and the GlitchPass Loader; each node
  has a one-line comment instead (hover it).

## 1.5.1 — 2026-11-24

### Fixed
- Editor: "Render sequence" could fail with "A process in the process pool was terminated
  abruptly" or "Unable to allocate … MiB": the render ran out of memory.
  - It started one render worker per CPU thread while the preview workers kept running.
  - Each worker kept up to 650 MB of cached frames for the time effects. Measured, the cache
    made rendering no faster at all; it only multiplied memory by the number of workers. The
    caches are now 260 MB, and a long 1080p run peaks at ~650 MB per worker instead of 900+ MB
    and still climbing.

  Now:
  - the number of workers is sized to the free memory, from measured long-run peaks (the
    status bar says when it uses fewer);
  - the preview workers are freed during a full render and come back after it;
  - if a worker still dies or runs out of memory, the unfinished frames are rendered again
    with half the workers,
    down to one frame at a time. Same pixels, only slower. Only a frame that crashes a lone
    worker stops the render, and the message names that frame and shows the worker's trace.
- The command line and the Resolve plugin's renders use the same memory-aware, self-healing
  pool.
- Transitions (appear / vanish) with the time effects could give different pixels depending on
  how frames were split between workers (1 worker ≠ 4 workers), because a per-worker cache mixed
  up the dissolving frame with the clean one. Renders are now identical whatever the worker
  count, and a partial re-render matches the full one again. (Transitions rendered before this
  fix may differ slightly from new renders; everything else is bit-identical.)
- `--install-resolve` from the packaged Windows app no longer waits on a message box when its
  output is captured (it hung CI).

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
