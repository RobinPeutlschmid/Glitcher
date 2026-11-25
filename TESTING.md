# Testing Glitcher

How Glitcher is tested, what the deep dive of 1.6.0 found, and what is still not covered.
Run everything with:

```bash
python -m pip install -r requirements.txt -r requirements-dev.txt
python -m pytest tests                      # Linux without a display: xvfb-run -a python -m pytest tests
```

157 tests, about a minute on two cores. CI runs them on Windows, macOS and Linux, then builds the
portable app and checks the built app (`tests/smoke_frozen.py`).

## Why the tests missed the Resolve 1-frame bug

The plugin was tested against `tests/fake_resolve.py`, a stand-in for Resolve's scripting
API. I wrote the fake from what I *believed* Resolve does. When a belief was wrong (Resolve
numbers an image sequence's frames by its file numbers, not from 0), the plugin and the fake
were wrong in the same way, so the tests passed. A fake can only test code against its own
assumptions. More fake-based tests would not have helped.

The fix is to check the fake against the real thing:

1. **Check this Resolve** (button in the panel) runs `probe()`. On a throwaway timeline it asks
   Resolve every question the plugin depends on:
   - importing a sequence numbered from 101 and one numbered like a timecode (90000), as a
     pattern and as a file list;
   - placing each clip whole, with 0-based frames, with file-number frames, and trimmed;
   - what a new Fusion comp contains and which inputs Loader, Bitmap mask, Glow and Color
     Corrector have;
   - how an EXR render of 3 frames is named and counted.

   It saves the answers to `resolve_probe.json` and deletes everything it made.
2. Put that file in `tests/resolve_contract/`.
   `test_fake_matches_recorded_resolve` runs the same probe on the fake and fails on any
   difference: "the fake Resolve disagrees with 21.1-windows.json: …".
3. Fix the fake until it agrees. From then on, every plugin test runs against a fake that
   behaves like your Resolve, and a new Resolve version can be checked with one click.

I checked that the comparison really fails: a made-up recording in which Resolve makes the
1-frame clip makes the test fail and names the difference.

**Still open:** no real recording exists yet. Until one is added, the two contract tests
skip, and say so in the test output.

## What each test file covers

| File | Tests | What |
|---|---|---|
| `test_render.py` | 24 | determinism, partial == full renders, 1 worker == 3 workers, overwrite protection, plate, appear, body parts, motion, EXR, glitch pass |
| `test_combinations.py` | 14 | features in combination (below) |
| `test_sensitivity.py` | 54 cases | every option changes the output; neutral options don't (below) |
| `test_memory.py` | 2 | no growth over a long render once the caches are full |
| `test_pool.py` | 7 | worker pool: sized to memory; killed and out-of-memory workers are retried; a real crash is named |
| `test_pose_solver.py` | 13 cases | landmarks → yaw / pitch / roll / scale on a synthetic skeleton |
| `test_pose.py` | 6 | estimator CLI, `--check`, error messages |
| `test_audio.py` | 6 | WAV at every bit depth, float WAV via miniaudio, beat detection, passage, sensitivity |
| `test_gui.py` | 9 | editor: preview == CLI, undo/redo, presets, folding, full render, Resolve job mode |
| `test_resolve_plugin.py` | 22 | the round trip against the fake Resolve, the probe, the contract |
| `smoke_frozen.py` | — | the *built* app: render == source code, EXR, editor, `--render-job`, `--install-resolve`, pose with the bundled model |

## Memory

`tools/memcheck.py` renders frame after frame in one process and records memory after each
frame:

```bash
python tools/memcheck.py FRAMES -n 144 -- --glitch-pass --format exr
```

### Per feature

Real 900×1300 clip, 144 frames:

| Settings | Peak MB | Growth per 10 frames at frame 144 |
|---|---|---|
| time effects off | 111 | 0.0 |
| defaults (time effects on) | 426 | 5.9 |
| strong time effects | 463 | 8.1 |
| + plate | 482 | 2.9 |
| + body parts | 410 | 1.3 |
| + 3D pose | 433 | 3.5 |
| + audio | 445 | 7.0 |
| + motion | 445 | 3.9 |
| + transitions | 450 | 9.0 |
| + glitch pass, EXR | 512 | 3.9 |
| EXR input | 573 | 3.4 |
| follow + pad 0.2 | 456 | 4.0 |
| everything at once | 641 | 11.1 |

### Is the growth a leak?

**No.** A 400-frame render levels off at 434 MB from frame 150 to frame 400. The growth seen
at frame 144 is the last two caches filling up: they hold a fixed number of motion fields (96)
and matching images (64). The time effects' frame caches are capped at 160 MB and 100 MB.

**Leak test:** `test_memory.py` renders 320 frames and measures Python's own allocation
counter (NumPy arrays included) between frame 160 and frame 320; growth must stay under 4 MB.
An injected leak of one frame-sized array per frame grew 30 MB and failed the test, so the
test does catch leaks.

### The editor

After scrubbing all 144 frames at 100%, the editor uses 97 MB and its preview workers 114 MB.
Forty edits add 1 MB; the undo stack holds a few kB per step. Preview workers render at
reduced size, so their caches stay small.

### The out-of-memory renders of 1.5.0

The cause was 650 MB of frame caches per worker multiplied by one worker per CPU thread.
Measured, those caches made rendering no faster at all: 1106 ms per frame with them,
1121 ms without. They were cut to 260 MB, and the worker count is now fitted to free memory.

## Combinations

There are 12 features with 2–3 settings each: plate, body parts, 3D pose, time effects,
transitions, motion, audio, glitch pass, EXR, follow, pad, timing. All combinations would be
10,368 renders. `test_combinations.py` instead builds 13 combinations (all-pairs) in which
every pair of settings of every two features occurs at least once. Bugs that live between two
features, like the 1-vs-4-worker transition bug, show up there.

Each combination must keep four promises:

1. **Determinism:** 1 worker and 3 workers give the same pixels.
2. **Partial renders:** `--range` gives the same frames as a full render.
3. **Alpha:** opaque over a plate; empty before `--appear`.
4. **Glitch pass:** black on every unchanged pixel.

**Result:** all 13 pass. No new combination bug was found. The transition bug of 1.5.1 had
already been fixed, and its own test guards it.

## Settings with no effect

`test_sensitivity.py` changes every render option alone, in a context where it applies:
`--spill` needs a plate, `--lock` needs a pose, `--hit-len` needs hits, and so on. The output
must change. Options that are *meant* to leave pixels alone (worker count, lossless EXR
compression) are checked the other way round: they must change nothing. A further test
fails if a new option is added without a case.

**Result:** every option does something.
- `--pose-unwrap` changed nothing with one pose key per frame. That's correct: 185° and −175°
  point the same way, so unwrapping only matters where angles are blended (`--pose-smooth`,
  or pose keys not on every frame). Its help text now says so.
- `--jobs` and `--exr-compression piz` / `none` change no pixels, as intended.

## Coverage and dead code

Line coverage of the whole suite:

| File | Covered | Notes |
|---|---|---|
| `glitch.py` | 89% | uncovered: mostly Windows/macOS-only branches; code that runs in worker processes isn't counted |
| `resolve/Glitcher.py` | 80% | uncovered: the panel's file dialog and starting the editor |
| `estimate_pose.py` | 61% (was 31%) | the solver is now tested (below); uncovered: model download, plotting |
| `glitch_gui.py` | 53% | the editor's drawing and event code; the end-to-end editor checks cover the workflows |

**Dead code found and removed** (vulture plus coverage): an unused pixel helper (`_fin`), two
pose attributes that were set but never read, an unused plugin function (`save_config`), and
a value recorded for nothing. `_bold` in the editor looks unused but keeps Tk from deleting a
font, so it stays, with a comment.

## The pose solver: a real bug found

The test footage has no person in it, so MediaPipe never found one, and the solver that turns
landmarks into yaw, pitch, roll and scale never ran in any test. `test_pose_solver.py` now
builds a skeleton, turns it by known angles, and gives the solver exactly what MediaPipe would
report. A slow 180° turn while walking 40% closer is recovered within tolerance for yaw,
pitch, roll and scale.

It also found a bug: **a person who never faces the camera got the wrong yaw.**
- One of the three yaw cues was calibrated on the clip's widest view of the shoulders,
  assuming the person faces the camera at some point.
- Held at 45° for the whole shot, the person read as 31°; held at 135°, as 149°.
- It is now calibrated from the body's own proportions in MediaPipe's metric skeleton:
  30 / 45 / 60 / 135 / 150° are recovered exactly.
- On a real clip that does face the camera, results move by 0.8° on average, 2.8° at most.

## Performance

Measured on the real 900×1300 clip, one worker. Where a frame's time goes with default
settings (330 ms in total):

| | ms |
|---|---|
| echo (ghost trails) | 83 |
| saving the PNG | 50 |
| fx_jpeg | 47 |
| datamosh | 39 |
| noise | 37 |
| loading history frames | 25 |
| everything else | ~50 |

### What was done

All of it is bit-identical: 6,000 random blending cases, 96 JPEG-effect renders, 60 real
frames, and the 1.2 regression set (2D, 3D, time effects, plate, parts, motion, EXR) are
unchanged.

- **Echo / datamosh compositing:** blends the covered region in one pass instead of picking
  pixels out and writing them back. The blend itself is 1.7× faster; renders are 8% faster with
  defaults (320 → 295 ms) and 10% with heavy time effects (940 → 856 ms).
- **fx_jpeg:** no full-frame 64-bit temporary array, and the spill area is computed only when
  needed (4%).
- **Workers:** the command line and the Resolve plugin use every CPU instead of at most 8,
  fitted to free memory. 1 → 2 workers measured 1.62× here. On a 16-thread PC, 8 → 16 workers
  is my estimate of the largest gain, but I couldn't measure it on a 2-core machine.

### Looked at, and left alone (with reasons)

- **PNG compression level:** levels 1–3 take the same 45 ms per frame and give the same file
  size, so there's nothing to win.
- **The frame caches:** bigger caches made no difference to speed (above). They are now smaller.
- **Noise:** draws a random value per pixel of the full frame so the grain doesn't change when
  the layout box moves. Drawing less would change the look.
- **Reusing the current frame as the next frame's history:** saves one load per frame (about
  3%). But the current frame can already be altered (plate, transition), and handing it on
  safely is exactly what the 1.5.1 bug got wrong. Not worth the risk.
- **Echo compositing beyond this:** what's left is several passes per layer over the image.
  Removing it would take restructuring the echo, which is riskier for a few percent.

**Where the time really goes:** the time effects (echo, stutter, datamosh) are on by default
and take about 60% of a frame (183 ms without them, 449 ms with). That's a creative choice
rather than an optimisation: switch them off, or lower their gains, for shots that don't need
them.

## What is still not covered

- **Real Resolve:** only through the contract once a recording exists (above). The panel's
  progress timer, its file dialog and starting the editor from Resolve run only in Resolve.
- **The editor on Windows/macOS:** CI runs the editor tests there, but things like DPI
  scaling and fonts are only checked by eye.
- **Pose estimation on real people:** MediaPipe itself is checked to load and run (`--check`),
  and the solver is tested on synthetic skeletons; accuracy on real footage is only checked by
  eye.
- **Memory on Windows:** the leak test runs on every OS in CI. The per-feature numbers above
  are Linux; Windows reports memory differently but should behave the same.
