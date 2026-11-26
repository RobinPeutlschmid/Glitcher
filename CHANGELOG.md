# Glitcher changelog

Version numbers: MAJOR.MINOR.PATCH. The version lives in one place, `__version__` in glitch.py;
the GUI title, `--version` of every tool, saved presets and the portable build's zip name read it
from there.

## 2.0.0 — 2026-11-26

**Glitcher 2.0: one native engine for the app and for Fusion.** The editor, its renders and the
command line now render with `glitch_core` (Rust, multithreaded), the engine the new Fusion node
uses. The node gains a follow box, motion-reactive timing and a glitch matte output, which
completes what it needed from the app.

### The app renders with the native engine
- **Preview, playback, renders, the command line and Resolve jobs** go through `glitch_core`
  when its library is there, and through the Python renderer otherwise. The planner (layout
  boxes, follow, curves, hits, motion, audio, appear / vanish, pose) stays in glitch.py and hands
  each frame's settings to the engine, so every option works as before.
- **Speed**, measured on 2 cores, whole renders including file reading and writing: a 16-frame
  4K shot in 9.7 s instead of 31.4 s (3.2x); 24 frames of HD in one process 4.7 s instead of
  9.5 s (2x). The engine is ~45 ms of an HD frame; the rest is PNG decoding and encoding, which
  the render workers overlap. Not measured on more cores yet.
- **Different look, same kind.** The engine draws its own random events, so the same seed gives
  a different (equally coherent) glitch than 1.x, and the time effects run on the subject before
  a plate is composited (as in the node). Old projects open and render, but not pixel for pixel
  as before. `GLITCHER_ENGINE=python` renders with the 1.x renderer.
- **Render workers:** the engine is multithreaded itself, so a render runs a quarter as many
  worker processes as there are cores, each with its share of threads, rather than one per
  core. Workers are started fresh (not forked) when the engine is used. Playback workers use
  one engine thread each.
- **Conversions in the engine.** Straight 8-bit / float frames to premultiplied float and back
  are done on the engine's threads (`gc_premultiply`, `gc_unpremultiply`), with the same
  arithmetic as before; in numpy they took longer than the render itself.
- `glitch_native.glitch()` takes glitch.py's `glitch()` arguments and frame types (PIL in, PIL
  out; float 0..255 in, float out; padding, trim, bbox, shift, 3D angles, reveal, landmarks,
  body parts, plate) and can also return the glitch matte.
- `Glitcher --engine-check` (and `python glitch_gui.py --engine-check`) says whether the native
  engine is used; exit code 0 = yes.
- **Packaging and CI:** `build_portable.py` bundles the library (building it with cargo if
  missing; `--no-engine` leaves it out). CI builds the engine before the app's tests, which now
  require it, and the frozen-app check compares the app's native render and its Python render
  with the source code's, pixel for pixel. Releases also carry `Glitcher-Fusion-<version>-<os>.zip`
  (the node and its library).
- Versions: the app, the engine (`engine/Cargo.toml`) and the Resolve plugin are all 2.0.0, and
  a test keeps the engine's version equal to the app's.

### Fusion node: follow box, motion-reactive, glitch matte
- **Follow the subject.** A new layout box mode, and the default in 3D (*Automatic* does what the
  app does: whole shot in 2D, follow in 3D). The box's centre is the subject's, smoothed over
  *Follow smoothing* seconds; its size is the shot's typical size (90th percentile), times the
  pose's scale unless *Ignore pose scale* is ticked. It comes from the shot scan, like the
  whole-shot box.
- **Motion-reactive** (0..1). The shot scan also measures how much the subject changes from frame
  to frame (on a ~160 px grey copy of brightness and alpha), smoothed and scaled to the shot's
  95th percentile, as the app's motion curve. Calm frames get less glitch, fast ones more, and
  the sharpest a burst.
- **Glitch matte**, a second output: where the output differs from both the clean subject (on the
  plate) and the plate, ramped over 2..64 levels as the app's glitch pass, in R, G, B and alpha.
  *Advanced → Glitch matte output* turns it off.
- C interface ABI 8: `GcParams` gains `follow`, `motion`, and `reveal_set` / `reveal_idx` /
  `reveal_set_q` / `landmarks` (a frame planned by the host: the app hands over its own reveal
  and landmarks); `GcStats` gains `motion` and `layout_from`; `gc_render` takes a matte image.

### Earlier steps of the native engine and Fusion node (0.1-0.9, unreleased)
- **`engine/` (glitch_core, Rust): the start of the native engine.**
  - Contains the scheduler and the time effects: echo, stutter and mosh. The other effects
    come next.
  - Works in premultiplied float, and renders the same frames on every OS and CPU (no
    platform maths library in pixel code).
  - Has a C interface that the Fusion node, Python and later OpenFX share.
  - The same statistics as glitch.py (how often each effect shows and how much it covers),
    with different random draws.
  - At 4K on 2 cores: echo ~70 ms, stutter ~30 ms, mosh ~125 ms per active frame.
- **`fusion/Glitcher.fuse`: the Glitcher node** for Resolve's Fusion page and Fusion Studio.
  - Controls: seed, intensity, tempo, density, bursts, and an Echo / Stutter / Mosh strength.
  - Fetches only the earlier frames the engine asks for, and hands Fusion's float images to
    the engine in place.
  - `fusion/GlitcherProbe.fuse` and `fusion/NOTES.md` record what Fusion exposes to a node,
    including the pixel layout.
- `glitch_native.py`: the engine from Python (ctypes).
- **Engine 0.9.1: pose estimation fixed in Resolve** (from the first real run).
  - **Range.** The node couldn't read the clip's frame range while rendering, so it stored
    and scanned only frame 0. It now reads the range from the UI side, where the comp is
    reachable: on any control change or button press. It writes the range into a new
    **Shot frames** field (editable), and the render uses that. If no range can be read, the
    Console says so and the layout uses this frame's box.
  - **Duplicates.** Resolve runs several copies of a node's script side by side, each with its
    own engine, so every copy started its own pose run and the estimator ran again and again.
    Pose runs (keyed by folder) and shot scans (keyed by a hidden per-node id) are now shared by
    every engine in the process, so there is one run and one estimator process. A finished-run
    marker now includes the range.
  - Tests: the Fusion stand-in now refuses comp attributes while rendering, as Resolve seems to.
    It runs two copies of the node side by side and checks one estimator run and one shot scan.
  - C interface: `gc_shot_begin` is replaced by `gc_shot_open(engine, key, generation, ...)`.
- **Engine 0.9.0: appear and vanish** - the last part of glitch.py's look the engine lacked.
  - **The transition.** The subject builds up out of glitch structure (appear) or breaks apart
    and goes (vanish). This uses glitch.py's reveal map: arrival times swept over the layout
    box, blocks and row bands, ranked so progress q reveals a q share. Pixels in transit are
    shoved sideways in slices, become mosaic blocks and get a glow colour.
  - **Around it.** The glitch peaks mid-transition (0.9 x sin(pi q), as a hit) and is masked
    to where the subject exists, a little ahead of the front. Before the first appear or
    after a vanish the subject is hidden; with a plate the result is then the clean plate.
  - The timeline rules (which transition is running, progress, hidden / visible) match
    `Timing.reveal`, checked on the same cases. The pattern is drawn from the engine's own
    random numbers, so it looks like the app's in kind, not pixel for pixel.
  - Node: *Appear at* / *Vanish at* lists, *Appear / vanish length*, and *Appear here*,
    *Vanish here*, *Remove here* at the playhead. Show pose and the Debug log show the state.
  - C interface ABI 7: `GcParams` gains `appear`, `vanish` and `trans_len`; `GcStats` gains
    `reveal` and `reveal_q`.
- **Engine 0.8.0: the app's layout box, and hits.**
  - **Layout box.** The app lays effects out in the union of the subject's boxes over the whole
    shot (`_plan_boxes`), but the node used only the current frame's subject box, so its
    glitches stayed closer to the body. Measured on a subject drifting across the frame (16
    seeds): glitch.py's whole-shot path puts 5,692 px per frame outside the subject; the node
    put 4,811, and now puts 5,677 with the new default **Whole shot**. The node scans the
    render range once (`gc_shot_*`; the scan is kept while Resolve runs, *Rescan shot*
    redoes it). *This frame's subject* keeps the old layout, and *Box grow* enlarges either box.
    The earlier first measurement (half as much outside the subject) was seed noise: with
    enough seeds the engine matched glitch.py for the same box.
  - **Hits**, as glitch.py's `--hits` and the app's timing: frame[:strength[:seconds]]. Activity
    jumps to full, intensity rises to at least the strength, effects get up to 2x the events,
    and it all decays over the length. The envelope matches `Timing.hit` to 5 decimals. In the
    node: a *Hits* list, *Add hit here* / *Remove hit here* at the playhead, *Hit strength*
    and *Hit length*. A malformed list is ignored, with a Console line (warn bit 16).
  - The Debug log also shows the input and output images' domain of definition (`dod=`), to
    check whether Fusion crops anything the glitch draws outside the subject. With Show pose
    on, the overlay shows the hit level and the layout box.
  - C interface ABI 6: `GcParams` gains `layout_mode`, `layout`, `box_grow`, `hits` and
    `hit_len`; `GcStats` gains `hit`.
- **Engine 0.7.0: see the pose the node uses.** The Yaw / Pitch / Roll sliders are offsets on
  top of the pose file and never show its values, so nothing in the Inspector showed whether
  a pose was used. New **3D → Show pose** draws over the output:
  - the landmark skeleton (subject's left orange, right blue);
  - the 3D camera as a ring around the body, with a facing arrow and an up line;
  - the angles used on that frame, and whether the pose and landmark files were read.

  The Debug log prints the same angles (`view=`). C interface ABI 5: `GcParams._pad` becomes
  `overlay`, and `GcStats` gains `has_view` and `view` (yaw × lock, pitch, roll, camera
  distance). The node's FFI types are renamed (`GcParams5`, `GcStats5`) so a reloaded node
  can't clash with the old layout.
- **Engine 0.6.0: pose estimation in the node.** Pressing *Estimate pose* starts a run, and
  playing or scrubbing the shot once does the rest. The node stores its input frames as
  PNGs; each render also fetches a few missing ones, within 250 ms, so Fusion stays
  responsive. Then it runs `estimate_pose.py`, or the Glitcher app's `--estimate-pose`, in the
  background with the comp's frame numbers. The pose and landmark fields point at the result,
  and *Reload pose* re-renders with it. Yaw / pitch / roll hints re-solve in seconds from the
  landmark cache, and *Start over* re-captures. Progress and errors go to the Console.
  - Engine: `posejob` (capture with conversion to straight 8-bit sRGB, box
    downscale, PNG writing on worker threads, launching, progress from the estimator's
    `--progress` lines, a done marker so a reopened comp doesn't run it again), and the C
    functions `gc_pose_request`, `gc_pose_feed`, `gc_pose_missing`, `gc_pose_status`,
    `gc_pose_sizes`. New dependency: the `png` crate (pure Rust).
  - The Glitcher app: started without console handles (windowed build), it writes its output
    to `GLITCHER_LOG` when set, so the node can show the estimator's progress and errors.
  - Tested: a stand-in estimator run end to end from the Rust tests and from the Fusion
    stand-in. `estimate_pose.py` with MediaPipe also ran on frames the engine stored; it
    accepted every argument and read every frame.
- **Engine 0.5.0: 3D mode, body parts and plates** — everything of glitch.py's look except
  reveals.
  - **3D mode**: each frame effect's event is a plane in a volume around the body
    (glitch.py's `Space3D` / `_place`), three times as many are spawned, and each plane effect
    runs twice, once for the planes in front of the body and once for those behind it. What the
    behind pass changes goes *under* the front result. RGB split gets its depth parallax. With
    a roll the planes are laid out in an upright working frame (premultiplied bicubic) and
    only the pixels the glitch changed are turned back, so the rest of the frame stays exact.
  - **Angles** from estimate_pose.py's pose CSV (read by the engine: interpolated, optional
    Gaussian smoothing, optional yaw unwrap), plus yaw / pitch / roll values that are added to
    it; lock, camera distance (or the file's `cam_dist`), depth and tilt as in glitch.py.
  - **Body parts**: *only*, *protect* and per-effect *targets*, as capsules around the
    MediaPipe landmarks at quarter resolution (glitch.py's `part_mask`). In a rolled shot the
    per-effect masks turn with the working frame.
  - **Plates**: the subject over the plate, a spill mask (glitch.py's `_spill_mask`: Chebyshev
    distance on an 8 px grid, smoothstep falloff) limits the damage, and the result goes over
    the plate again so there are no holes. One deliberate difference: the time effects run on
    the subject before compositing (see `fusion/README.md`).
  - **Landmarks as text**: estimate_pose.py now also writes `<pose>_landmarks.csv` next to the
    `.npz`. The engine reads it without numpy, and glitch.py reads either.
    `tools/landmarks_csv.py` converts older `.npz` files.
  - **Faster frame effects**: each effect used to copy the whole content area before it
    started. Now the picture "before" is copy-on-write in 64 px tiles: an effect saves only
    the tiles it is about to write, and reads everything else live. Writers borrow only their
    own pixels of a row, so the reads stay sound. At 4K on 2 cores this took the 2-D
    median from 339 to 175 ms, the rolled 3D one from 625 to 316 ms and plates from 571 to
    303 ms.
  - Checked against glitch.py with `tools/compare_native.py --scene 3d|roll|parts|plate` on a
    synthetic figure: the frame effects' statistics agree within a few percent in every
    scene. The protected head stays untouched in both, and neither damages the plate beyond the
    spill area.
  - New tests: every pixel an effect changes lies in the area it reports (2-D and both 3-D
    passes; the layering relies on it). A rolled frame with the effects off comes out
    bit-identical. *Protect* and *only* keep their areas exactly clean in 2-D, 3-D and rolled.
    A plate is never touched beyond the spill mask. In-place rendering equals rendering to a
    separate output.
  - C interface **ABI 4**: `GcParams` gains the 3D, track-file, body-part and spill fields.
    `gc_render` gains a plate image, and `GcStats` gains `flags` and `warn`. New functions:
    `gc_last_error`, `gc_parse_parts`, `gc_parse_targets`, and `gc_sizes`, which the node and
    `glitch_native.py` use to check their struct layouts against the library.
  - Node: a **Plate** input and collapsible **Effects / 3D / Body parts / Plate / Advanced**
    groups. Pose and landmark files are picked with a file browser, and there is a Track
    frame offset. Warnings go to the Console, and the debug log shows the track frame and mode.
    Usage: `fusion/README.md`.
- **Engine 0.4.0: all thirteen effects.** The ten frame effects are ported in their 2-D form:
  JPEG, blocks, slices, wave, pixel sort, smear, RGB split, tint, scanlines and noise.
  - Colour changes are computed on the straight colour and premultiplied again, so they look
    like glitch.py's. Effects that only move pixels move premultiplied ones.
  - Against glitch.py, `tools/compare_native.py` measured each effect alone on 4 seeds × 100
    frames: frames hit, share of the subject covered and strength all agree within a few
    percent. JPEG is slightly weaker (99 against 116): its noise is a hash-based
    near-Gaussian, with lighter tails than numpy's.
  - The engine tracks the area that holds content as effects write. So no effect scans or
    copies the whole frame to find the subject, and each one snapshots only that area.
    A test checks the tracked area never misses a pixel.
  - Speed at 4K on 2 cores, all effects: median 210 ms per frame at the defaults, 530 ms at
    intensity 1 with every effect at 2.
  - The node exposes a strength for every effect, plus Tint amount and JPEG quality.
  - C interface ABI 3: `GcParams` gains tint, quality and bleed; `GcStats` gains a count per
    frame effect.
  - Not ported yet in 0.4.0: 3-D mode, body parts, plates and reveals (all but reveals came in
    0.5.0).
- **Engine 0.3.0, so the node works during playback.** Resolve's Fusion page hands a node
  32-bit float frames when scrubbing (Depth 8) and 16-bit float ones during playback
  (Depth 7). The node only took 32-bit, so in playback it passed every frame through. The
  engine now takes float32 in place and converts float16, uint16 and uint8 in and out
  (exact half-float conversion, tested on all 65536 values), and the history can mix formats
  frame by frame.
  - The node tells float16 from uint16 (both 2-byte channels) by checking against
    `GetPixel`, once per Depth code.
  - C interface ABI 2: `GcImage` gains `format`, and its stride counts channel values.
- Node: a **Debug log** option (one line per frame: what Fusion asked for, which earlier
  frames arrived and in what format, what the engine did). It is also marked time-variant now,
  and answers pre-calc requests without rendering.
- `tools/compare_native.py`: glitch.py and the engine side by side, with statistics and a
  contact sheet.
- CI builds and tests the engine on Windows, macOS and Linux, tests the node against a
  stand-in of Fusion's API, and uploads `glitcher-fusion-<os>` (the library plus the .fuse
  files).

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
