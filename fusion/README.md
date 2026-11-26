# The Glitcher node for Fusion

`Glitcher.fuse` puts Glitcher into Resolve's Fusion page (or Fusion Studio) as a node. The pixels
are processed by `glitch_core`, the native engine (`engine/`); the node only reads its controls,
fetches the earlier frames the engine asks for and hands everything over.

## Install

Copy `Glitcher.fuse` and the library — `glitch_core.dll` on Windows, `libglitch_core.dylib` on
macOS — into Fusion's Fuses folder and restart Resolve. On Windows that is
`%APPDATA%\Blackmagic Design\DaVinci Resolve\Support\Fusion\Fuses`. Each [release](https://github.com/RobinPeutlschmid/Glitcher/releases/latest) has them as
`Glitcher-Fusion-<version>-<os>.zip` (Windows: `Glitcher-Fusion-<version>-windows-x64.zip`); from source, `cd engine && cargo build --release` makes
the library in `engine/target/release/`. The library and the node must come from the same
build: the node checks this and says so in the Console otherwise.

## Inputs and outputs

- **Input** (yellow): the keyed subject, with alpha.
- **Plate** (green, optional): a background. The subject is composited onto it, the glitch may
  damage the plate around the subject (**Plate → Spill onto the plate**, 0 = only the subject),
  and the output is the composite. A plate of another size is resized to the input.
- **Output**: the glitched subject (or composite).
- **Glitch matte** (second output): where the glitch is, see *Glitch matte* below.

## Controls

| Group | What |
|---|---|
| top | Seed, Intensity, Tempo, Density, Bursts |
| Timing and layout | Shot frames, **Layout box** (Automatic / This frame's subject / Whole shot / Follow the subject), Follow smoothing, Box grow, **Motion-reactive**, Rescan shot, **Hits**, Hit strength, Hit length, **Add hit here**, **Remove hit here**, **Appear at**, **Vanish at**, Appear / vanish length, **Appear here**, **Vanish here**, **Remove here** |
| Effects | a strength per effect (1 default, 0 off, up to 3), Tint amount, JPEG quality |
| 3D | **3D mode**, Yaw / Pitch / Roll (degrees; animatable), Lock to the turn, Camera distance, Volume depth, Tilted planes, **Pose file** (from `estimate_pose.py`), Pose smoothing, ignore the file's pitch / roll, unwrap yaw |
| Body parts | **Landmarks file** (`<pose>_landmarks.csv`), *Glitch only* these parts (none ticked = everywhere), *Keep clean* these, *Per effect* (`mosh=head; jpeg=torso+arms`), edge softness |
| Plate | Spill onto the plate |
| Pose estimation | Estimator, Pose folder, Model, Yaw / Pitch / Roll hints, lens focal length and sensor width, stored frame size, Input is linear light, **Estimate pose**, **Reload pose**, **Start over** |
| Advanced | Track frame offset, Frame rate (0 = the comp's), **Glitch matte output**, Debug log, Library path |

With a pose file the sliders are **added** to it: leave them at 0 to follow the file, or use
them to correct it. Without one they are the angles. The sliders never show the file's values:
the file is read inside the engine each frame. To see what is used, see below.

## Checking that the pose is used

Tick **3D → Show pose**. The output then shows:
- the landmarks as a skeleton: the subject's own left side orange, right side blue. On a
  person facing the camera, orange is on screen right; if it isn't, the fit is mirrored there;
- a ring around the body seen through the glitch's 3D camera, with a magenta arrow where the
  subject faces and a green line for its up direction. It should turn, tilt and roll with the
  person. The front half of the ring is bright, the back half dim;
- top left: the frame and track frame, the **yaw / pitch / roll / camera distance actually used
  on this frame** (file + sliders, yaw times Lock), and whether the pose and landmark files were
  read.

Turn it off before rendering: it is drawn into the output. The **Debug log** line has the same
numbers (`view=yaw … pitch … roll … cam …`) without drawing anything. In 3D mode the frame effects become planes
in a volume around the body — some in front of it, some behind (those only show where the
body doesn't cover them) — seen through the camera; with a roll they are laid out upright and
turned with the subject.

## Shot frames

The node needs the shot's frame range for the whole-shot layout box and for pose estimation.
It reads the clip's In / Out (the yellow render range marks on the Fusion page) from the comp
when you change any of its controls or press a button, and writes it into **Timing and layout
→ Shot frames** (e.g. `0-249`). That's because Resolve doesn't let the node read the comp
while it renders. You can also type the range there; an empty field is filled in again the
next time you change a control. If the Console says it couldn't read the range, touch any
control or type it in.

## Layout box

The effects are placed inside a layout box. **Automatic** (the default) does what the Glitcher
app does: in 2D, **the whole shot's box**; in 3D mode, **following the subject**.

- **Whole shot**: the union of where the subject is across the shot. Glitches land wherever the
  subject goes during the shot, including where it isn't on this frame, so they spread off the
  body. Right for a subject that stays roughly in place.
- **Follow the subject**: a box that moves with the subject. Its centre is the subject's,
  smoothed over **Follow smoothing** seconds (0.4 by default; 0 = no smoothing), so it glides
  instead of jittering with the key. Its size is held steady at a typical size for the shot
  (the 90th percentile of the subject's boxes), so the glitch layout doesn't breathe when an
  arm moves out. With a pose file, the size also follows the pose's scale (the subject walking
  towards or away from the camera); tick **3D → Ignore pose scale** to keep it fixed. Right for
  a subject that walks across the frame, where the whole-shot box would be most of the frame.
- **This frame's subject**: only where the subject is on this frame. No scan needed.

Every mode but the last needs the **shot scan**: on the first render the node reads every frame
of *Shot frames* once, which pauses Fusion for a moment (about the time it takes to play the
range). The Console says how long it took. The scan is kept while Resolve runs; press **Rescan
shot** after changing the key or the footage. Ranges over 3000 frames fall back to this
frame's box. **Box grow** enlarges any of the boxes on every side (0.25 = a quarter of its
size), so effects spread further still. With a plate, the box also sets how far *Spill*
reaches. Show pose prints the box used and which kind it is (`LAYOUT BOX … (WHOLE SHOT)`,
`(FOLLOW)` or `(THIS FRAME)`); the Debug log has it as `layout=`.

## Motion-reactive

**Motion-reactive** (0..1, default 0 = off) ties the glitch to how fast the subject moves, as
the app's *motion* setting does: calmer while the subject holds still, stronger in fast moves,
and a burst on the sharpest ones. The shot scan measures the motion: for every frame, how much
a small grey copy of the subject (its brightness and its alpha) changes from the frame before,
lightly smoothed and scaled so the shot's fast moments (its 95th percentile) are 1. At 1, a
still subject gets 15 % of the glitch and the fastest moves up to about 2x; anything faster
than 0.55 of the shot's fast moments also adds a hit-like boost. The measure is relative to
the shot, so a slow shot still has its own peaks. The Debug log shows it as `motion=`.

## Glitch matte

The second output is a matte of where the glitch changed the picture, like the app's *glitch
pass*: white where the output differs from both the clean subject (or subject over plate) and
the plate, black where it is untouched, a soft ramp between (differences from 2 to 64 levels in
8-bit terms). R, G, B and alpha are all the matte, so it works as a mask in any tool. Use it to
glow, grade or regrain only the glitch, or to drive a Lens Blur. It costs one comparison pass;
untick **Advanced → Glitch matte output** if you don't use it (the output is then black).

## Hits

A hit is a forced burst at a frame, as in the app: activity jumps to full, intensity rises to at
least the hit's strength and every effect gets up to twice as many events, then it all
decays over the hit's length. **Add hit here** adds one at the playhead with *Hit strength*,
and **Remove hit here** removes the one at or next to the playhead (within 3 frames). The
**Hits** field holds the list and can be typed in directly: `48, 112:0.6, 200:1:0.8` means
frame, then optionally strength (0..1) and length in seconds (default: *Hit length*). With
Show pose on, the overlay shows the hit level; the Debug log has it as `hit=`.

## Appear and vanish

As in the app: at an **appear** the subject builds up out of glitch structure, and at a
**vanish** it breaks apart and is gone. Pixels arriving or leaving are shoved sideways in
slices, turned into mosaic blocks and lit with a glow, in an order that sweeps over the layout
box. The glitch peaks mid-transition like a hit, and only shows where the subject already
exists. Before the first appear, and after a vanish (until the next appear), the subject is
invisible: on a plate you see the clean plate.

**Appear here** / **Vanish here** add one at the playhead (one transition per frame: the new
one replaces whatever was there), and **Remove here** removes the one at or next to it. The
**Appear at** / **Vanish at** fields hold the lists and can be typed in: `0, 120:1.5` means
frame, then optionally its length in seconds (default: *Appear / vanish length*). Each
transition gets its own build-up pattern, from the Seed. Show pose and the Debug log
(`reveal=`) show where a frame is.

## Pose estimation in the node

The node can make the pose and landmark files itself, from whatever reaches its input (a keyed
MediaIn, a Magic Mask, any comp):

1. Set **Estimator** once: the Glitcher app (a build with pose estimation) or `estimate_pose.py`.
   Left empty, the node tries the `GLITCHER_APP` environment variable, then an
   `estimate_pose.py` or `Glitcher.exe` next to `Glitcher.fuse`. For `estimate_pose.py` it uses
   the Python in `GLITCHER_PYTHON`, else a `venv` / `.venv` next to the script, else `py -3`
   (Windows) / `python3`; that Python needs MediaPipe.
2. Press **Estimate pose**. The shot's frames are the clip's In / Out (see *Shot frames*). The node fills in
   the Pose and Landmarks file fields and turns on 3D mode.
3. **Play or scrub through the range once.** Every frame the node renders is stored (as a PNG,
   at most *Stored frame size* on its longest side). Each render also fetches up to a quarter of
   a second's worth of missing frames, so Fusion stays responsive. The Console counts the frames.
4. When all frames are stored, the estimator runs in the background. The Console shows its
   progress, then `pose ready (person found in …)`.
5. Press **Reload pose**. Fusion only re-renders frames it hasn't cached, and frames rendered
   while estimating didn't have the pose yet.

Wrong way round for a stretch (pose models confuse front and back)? Add **Yaw hints** such as
`0:0, 120:180` (frame:degrees, in the comp's frame numbers) and press **Estimate pose** again.
The stored frames and landmarks are reused, so this re-solve takes seconds instead of re-running
MediaPipe. **Pitch / Roll hints** shift those curves through known values. **Start over** deletes
the stored frames first; use it when the footage or key changed.

The files go to the **Pose folder**, by default
`%LOCALAPPDATA%\Glitcher\pose\<date>-<node name>` (macOS `~/Library/Caches/...`, Linux
`~/.cache/...`). It holds the frames, `pose.csv`, `pose_landmarks.csv`, `pose.png` (a plot of
the curves: check it), the landmark cache and `estimate.log`. A finished run is not repeated
when the comp is reopened. The frames are kept for re-solving; delete the folder when the shot
is done.

The frames are numbered from the render range's start, so the pose's frame numbers are the
comp's and **Track frame offset** stays 0. If the comp works in linear light, tick **Input is
linear light** so the pose model sees normal-looking pictures. **Lens focal length** (with the
sensor width) adds camera distance to the pose, as `estimate_pose.py --focal-mm` does.

## Track files and frame numbers

`estimate_pose.py` writes the pose (`shot_pose.csv`) and the landmarks (`shot_pose_landmarks.npz`
for glitch.py, `shot_pose_landmarks.csv` for this node). For landmark files made before that,
convert: `python tools/landmarks_csv.py shot_pose_landmarks.npz`.

Both are looked up by frame number = the comp's frame + **Track frame offset**. If the frames
estimate_pose.py saw start where the comp starts, leave it at 0. If the shot starts later in
that footage, enter how many frames later. The **Debug log** prints the track frame it used
(`track=`) for every frame, so you can check. Landmarks are scaled from the size they were
found at to the input's size, so proxies and resized footage line up.

Problems with a file (missing, unreadable, no person in it) are printed once in the Console;
the frame still renders, without that file.

## Differences from glitch.py

- With a plate, the time effects (Echo, Stutter, Mosh) run on the subject before it is
  composited, not on the composite. So the node never needs earlier plate frames (half the
  fetching), and echo trails show on the plate; in glitch.py, behind an opaque composite,
  they are mostly hidden.
- The same seed gives different events (hash-based draws instead of numpy's), with the same
  statistics: `tools/compare_native.py --scene 2d|3d|roll|parts|plate` compares them.
- Motion-reactive measures the motion on the node's input (a small grey copy of each frame),
  so its curve is close to the app's but not identical.
- Not in the node: timing curves and audio. Animate the node's sliders instead of curves, or
  add hits at the beats.

## Speed

Measured at 3840x2160 on 2 cores, a figure filling the frame's height, intensity 0.8:

| Mode | Median per frame |
|---|---|
| 2D | 175 ms |
| 3D | 96 ms |
| 3D with roll | 316 ms |
| Plate | 303 ms |
| 3D + roll + plate + a protected part | 585 ms |

Most of it scales with cores, so a 16-thread machine should be several times faster (not
measured). Set the environment variable `GLITCH_CORE_PROFILE=1` before starting Resolve to get
a per-stage breakdown on stderr.
