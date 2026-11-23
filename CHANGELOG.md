# Glitcher changelog

Version numbers: MAJOR.MINOR.PATCH. The version lives in one place, `__version__` in glitch.py;
the GUI title, `--version` of every tool, saved presets and the portable build's zip name read it
from there.

## 1.2.1 — 2026-11-23
Fixed
Folding sections in a scrolled panel left the content drawn at the bottom under empty space (the old scroll position was kept and the scroll area never shrank). The panel now fits its content, returns to the top when everything fits, and hides the scrollbar when there is nothing to scroll.

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
