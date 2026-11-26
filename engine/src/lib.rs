//! glitch_core: Glitcher's native engine, with a C interface that LuaJIT's FFI (the Fusion
//! node), Python's ctypes and C++ (OpenFX) can all call. All of glitch.py's look: the 13 effects,
//! 3D mode (planes around the body from a pose file or angles), body-part masks (landmarks
//! file) and background plates.
//!
//! Images: RGBA, premultiplied, nominally 0..1, given as a pointer to the TOP row and a stride
//! in channel values (negative when the host stores rows bottom-up, like Fusion). float32 is
//! used in place; float16, uint16 and uint8 are converted in and out. Every
//! function catches panics and returns an error code: a crash would take the host (Resolve)
//! down with it.

pub mod detmath;
pub mod engine;
pub mod fx_frame;
pub mod fx_time;
pub mod image;
pub mod layers;
pub mod masks;
pub mod motion;
pub mod overlay;
pub mod pixfmt;
pub mod reveal;
pub mod posejob;
pub mod sched;
pub mod shotbox;
pub mod spike;
pub mod tracks;

pub use spike::{gc_copy_rows, gc_spike};

use engine::{Engine, Scene};
use image::{Frame, View, resample};
use pixfmt::{PixFmt, input_view, write_out};
use sched::{EFFECTS, MAX_BACK, Params};
use std::ffi::{CStr, CString, c_char};
use std::path::Path;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::OnceLock;
use std::time::Instant;

static VERSION: &[u8] = concat!(env!("CARGO_PKG_VERSION"), "\0").as_bytes();

/// Version of the library, e.g. "0.2.0".
#[unsafe(no_mangle)]
pub extern "C" fn gc_version() -> *const c_char {
    VERSION.as_ptr() as *const c_char
}

/// Worker threads the engine uses.
#[unsafe(no_mangle)]
pub extern "C" fn gc_threads() -> i32 {
    rayon::current_num_threads() as i32
}

/// Milliseconds on a monotonic clock (Lua's os.clock is CPU time, not wall time).
#[unsafe(no_mangle)]
pub extern "C" fn gc_now_ms() -> f64 {
    static T0: OnceLock<Instant> = OnceLock::new();
    T0.get_or_init(Instant::now).elapsed().as_secs_f64() * 1e3
}

/// Version of the structs below: a host built against another layout gets an error, not junk.
pub const GC_ABI: u32 = 8;

/// The look: same meaning and ranges as glitch.py's options.
#[repr(C)]
pub struct GcParams {
    pub abi: u32,          // = GC_ABI
    pub seed: u32,
    pub intensity: f64,    // 0..1
    pub fps: f64,
    pub tempo: f64,        // > 1 shorter, snappier events
    pub density: f64,      // event count multiplier
    pub bursts: f64,       // 0..1 calm/burst contrast
    pub boost: f64,        // 0..1 activity floor right now
    pub gains: [f64; 13],  // per effect, glitch.py's order; 1 = default, 0 = off
    pub tint: f64,         // 0..1 colour wash
    pub quality: i32,      // JPEG quality 1..100, 0 = from the intensity
    pub bleed: i32,        // px JPEG junk may spill past the silhouette, < 0 = 3 % of the width
    // ---- 3D (ABI 4)
    pub mode3d: i32,       // 0 = flat, 1 = planes in a volume around the body
    pub pose_flags: i32,   // 1 = ignore the pose file's pitch, 2 = its roll, 4 = unwrap its yaw,
                           // 8 = ignore its scale (the follow box keeps one size)
    pub yaw: f64,          // degrees, added to the pose file's (if any); + = front turns right
    pub pitch: f64,        // + = camera above
    pub roll: f64,         // + = subject rotated clockwise on screen
    pub lock: f64,         // share of the yaw the volume follows (1 glued, 0 ignores)
    pub cam_dist: f64,     // camera distance in box heights; <= 0 = the pose file's, else 3
    pub depth: f64,        // half-thickness of the volume in box widths (0.35)
    pub tilt: f64,         // share of planes tilted toward horizontal (0.25)
    pub pose_smooth: f64,  // Gaussian smoothing of the pose curves, seconds
    pub track_offset: f64, // added to the frame number to look frames up in the pose / landmarks
    pub pose_path: *const c_char,       // estimate_pose.py's CSV, NULL or "" = none
    pub landmarks_path: *const c_char,  // its _landmarks.csv, needed by the part options
    // ---- body parts: bits face 1, head 2, torso 4, arms 8, hands 16, legs 32, feet 64
    pub only: u32,         // the glitch may only touch these (0 = everywhere)
    pub protect: u32,      // these stay clean
    pub targets: [u32; 13], // per effect: only these parts
    pub overlay: u32,      // 1 = draw the pose check over the output ("Show pose")
    pub feather: f64,      // softness of the part edges (1)
    // ---- plate
    pub spill: f64,        // how far the glitch may damage the plate around the subject (0.05)
    // ---- layout and hits (ABI 6)
    pub layout_mode: i32,  // 0 = this frame's subject box, 1 = `layout`, 2 = from the shot scan
                           // (whole shot or follow, see `follow`)
    pub _pad2: i32,
    pub layout: [f64; 4],  // x0, y0, x1, y1 (y down, may extend past the canvas)
    pub box_grow: f64,     // grow the layout box by this share of its size on each side
    pub hits: *const c_char, // "48, 112:0.6, 200:1:0.8" = frame[:strength[:seconds]]
    pub hit_len: f64,      // seconds of a hit without its own (0.5)
    // ---- appear / vanish (ABI 7)
    pub appear: *const c_char,  // "0, 120:1.5" = frame[:seconds]: the subject builds up (invisible
                                // before the first appear)
    pub vanish: *const c_char,  // the subject breaks apart and is gone afterwards
    pub trans_len: f64,         // seconds of a transition without its own (1.0)
    // ---- from the shot scan (ABI 8): see gc_shot_open
    pub follow: f64,       // layout_mode 2: 0 = the whole shot's box, > 0 = a box following the
                           // subject, its centre smoothed over this many seconds
    pub motion: f64,       // 0..1: the glitch settles when the subject holds still, spikes on fast moves
    // ---- for a host that has planned the frame itself (glitch.py's planner: the app)
    pub reveal_set: i32,   // 0 = from appear / vanish, else 1 visible, 2 appearing, 3 vanishing, 4 hidden
    pub reveal_idx: i32,   // the transition's index (its pattern), with 2 / 3
    pub reveal_set_q: f64, // its progress 0..1, with 2 / 3
    pub landmarks: *const f64, // this frame's 33 x (x, y, visibility) in pixels, NULL = from the file
}

impl Default for GcParams {
    /// glitch.py's defaults (flat, no tracks, no parts).
    fn default() -> Self {
        GcParams { abi: GC_ABI, seed: 0, intensity: 0.5, fps: 25.0, tempo: 1.0, density: 1.0, bursts: 0.5, boost: 0.0,
                   gains: [1.0; 13], tint: 0.0, quality: 0, bleed: -1, mode3d: 0, pose_flags: 0, yaw: 0.0, pitch: 0.0,
                   roll: 0.0, lock: 1.0, cam_dist: 0.0, depth: 0.35, tilt: 0.25, pose_smooth: 0.0, track_offset: 0.0,
                   pose_path: std::ptr::null(), landmarks_path: std::ptr::null(), only: 0, protect: 0, targets: [0; 13],
                   overlay: 0, feather: 1.0, spill: 0.05, layout_mode: 0, _pad2: 0,
                   layout: [0.0; 4], box_grow: 0.0, hits: std::ptr::null(), hit_len: 0.5, appear: std::ptr::null(),
                   vanish: std::ptr::null(), trans_len: 1.0, follow: 0.0, motion: 0.0,
                   reveal_set: 0, reveal_idx: 0, reveal_set_q: 0.0, landmarks: std::ptr::null() }
    }
}

/// An image the host owns: `ptr` = top row, `stride` = channel values from one row to the next
/// (may be negative), `format` = 0 float32, 1 float16, 2 uint16, 3 uint8. NULL ptr = no image.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct GcImage {
    pub ptr: *mut std::ffi::c_void,
    pub width: i32,
    pub height: i32,
    pub stride: i64,
    pub format: i32,
    pub _reserved: i32,
}

#[repr(C)]
#[derive(Default)]
pub struct GcStats {
    pub ms: f64,
    pub echo: i32,
    pub stutter: i32,
    pub mosh: i32,
    pub has_box: i32,
    pub bbox: [i32; 4],     // x0, y0, x1, y1 (y down)
    pub frame_fx: [i32; 10], // jpeg, blocks, slices, wave, sort, smear, split, tint, scanlines, noise
    pub flags: i32,          // 1 = 3D, 2 = rolled, 4 = plate used, 8 = body parts used
    pub warn: i32,           // 1 = pose file unusable, 2 = landmarks file unusable, 4 = part options
                             // without landmarks, 8 = plate of another size (resized), 16 = hits
                             // or appear / vanish malformed (ignored); gc_last_error
    pub has_view: i32,       // 1 = 3D: `view` holds what this frame used
    pub _pad: i32,
    pub view: [f64; 4],      // yaw (x lock), pitch, roll in degrees, camera distance
    pub hit: f64,            // this frame's hit level 0..1 (ABI 6)
    pub reveal: i32,         // 0 visible, 1 appearing, 2 vanishing, 3 hidden (ABI 7)
    pub _pad3: i32,
    pub reveal_q: f64,       // progress of the transition 0..1
    pub motion: f64,         // the subject's motion at this frame (1 = the shot's fast moments), -1 = not known
    pub layout_from: i32,    // 0 this frame's subject, 1 given box, 2 whole shot, 3 follow box (ABI 8)
    pub _pad4: i32,
}

impl GcImage {
    fn fmt(&self) -> Option<PixFmt> {
        PixFmt::from_i32(self.format)
    }
    /// The image as a float32 view: in place for float32, else converted into `keep`.
    ///
    /// # Safety
    /// As `View::from_raw`, for the image's format.
    unsafe fn view(&self, keep: &mut Option<Frame>) -> Option<View> {
        if self.width <= 0 || self.height <= 0 || self.ptr.is_null() {
            return None;
        }
        unsafe {
            input_view(self.ptr as *mut u8, self.width as usize, self.height as usize, self.stride as isize,
                       self.fmt()?, keep)
        }
    }
}

fn params(p: &GcParams) -> Option<Params> {
    if p.abi != GC_ABI {
        return None;
    }
    Some(Params { seed: p.seed, intensity: p.intensity, fps: p.fps, tempo: p.tempo, density: p.density,
                  bursts: p.bursts, boost: p.boost, gains: p.gains, tint: p.tint, quality: p.quality,
                  bleed: p.bleed }.sane())
}

/// The look at `frame`: the params with that frame's hit applied. A malformed hit list is
/// ignored (the node shows the message).
unsafe fn params_at(e: Option<&Engine>, p: &GcParams, frame: i64) -> Option<(Params, f64, bool, reveal::Reveal)> {
    let par = params(p)?;
    let mut ok = true;
    let mut msgs = Vec::new();
    let text = unsafe { cstr(p.hits) };
    let mut hit = 0.0;
    if !text.trim().is_empty() {
        match sched::parse_hits(&text, fin(p.hit_len, 0.5).max(0.01)) {
            Ok(h) => hit = sched::hit_env(&h, frame as f64, par.fps),
            Err(m) => {
                ok = false;
                msgs.push(format!("hits: {m}"));
            }
        }
    }
    // appear / vanish: where the subject is, and the glitch's peak mid-transition
    let len = fin(p.trans_len, 1.0).max(0.01);
    let mut trans = Vec::new();
    for (txt, appear) in [(unsafe { cstr(p.appear) }, true), (unsafe { cstr(p.vanish) }, false)] {
        match reveal::parse(&txt, appear, len) {
            Ok(t) => trans.extend(t),
            Err(m) => {
                ok = false;
                msgs.push(m);
            }
        }
    }
    let rv = match p.reveal_set {
        1 => reveal::Reveal::Visible,
        2 | 3 => reveal::Reveal::Moving(p.reveal_set == 2, fin(p.reveal_set_q, 1.0).clamp(0.0, 1.0), p.reveal_idx.max(0) as usize),
        4 => reveal::Reveal::Hidden,
        _ => {
            let rv = reveal::at(&trans, frame as f64, par.fps);
            hit = f64::max(hit, reveal::boost(rv));         // a planned frame has its peak in its look already
            rv
        }
    };
    if let (Some(e), false) = (e, msgs.is_empty()) {
        e.set_error(&msgs.join("; "));
    }
    Some((par.with_hit(hit), hit, ok, rv))
}

/// glitch.py's apply_motion: with amount 1 the glitch drops to a sixth when the subject is still
/// and goes past full on fast moves (and forces activity).
fn with_motion(mut par: Params, v: f64, amount: f64) -> Params {
    if amount > 0.0 && v.is_finite() {
        let k = 1.0 - amount + amount * (0.15 + 1.6 * v.min(1.2));
        par.intensity = (par.intensity * k).clamp(0.0, 1.0);
        par.boost = par.boost.max(amount * ((v - 0.55) * 2.2).clamp(0.0, 1.0));
    }
    par
}

/// The engine's shot scan (if the host opened one).
fn shot_of(e: &Engine) -> Option<Shared<shotbox::ShotBox>> {
    lock(&e.shot).clone()
}

/// The subject's motion at `frame` from the shot scan, if it is complete.
fn motion_at(e: &Engine, frame: i64) -> Option<f64> {
    shot_of(e).and_then(|s| lock(&s).motion(frame))
}

/// A C string field as a path; None for NULL or empty.
unsafe fn path<'a>(s: *const c_char) -> Option<&'a Path> {
    if s.is_null() {
        return None;
    }
    let t = unsafe { CStr::from_ptr(s) }.to_str().ok()?.trim().trim_matches('"');
    (!t.is_empty()).then(|| Path::new(t))
}

fn fin(v: f64, d: f64) -> f64 {
    if v.is_finite() { v } else { d }
}

/// The frame's scene from the 3D / part / plate fields, loading the track files (cached).
/// Problems go into `warn` and the engine's last error; the frame still renders.
/// What the scene was made from, for the "Show pose" overlay.
#[derive(Default)]
struct SceneInfo {
    track_frame: f64,
    pose_track: Option<std::sync::Arc<tracks::Pose>>,
    layout_from: i32,
    pose: Option<bool>,      // None = no pose file, Some(read?)
    lm: Option<bool>,
}

unsafe fn scene(e: &Engine, p: &GcParams, frame: i64, w: usize, h: usize, warn: &mut i32, info: &mut SceneInfo) -> Scene {
    let mut msgs: Vec<String> = Vec::new();
    let tf = frame as f64 + fin(p.track_offset, 0.0);
    info.track_frame = tf;
    let mut sc = Scene { only: p.only & 127, protect: p.protect & 127, feather: fin(p.feather, 1.0).max(0.0),
                         spill: fin(p.spill, 0.05).clamp(0.0, 1.0), box_grow: fin(p.box_grow, 0.0).clamp(0.0, 5.0),
                         ..Default::default() };
    if p.layout_mode == 1 && p.layout.iter().all(|v| v.is_finite()) && p.layout[2] > p.layout[0] && p.layout[3] > p.layout[1] {
        sc.layout = Some((p.layout[0], p.layout[1], p.layout[2], p.layout[3]));
    }
    for (t, v) in sc.targets.iter_mut().zip(p.targets.iter()) {
        *t = v & 127;
    }
    if p.mode3d != 0 {
        let (mut ang, mut cam) = ([fin(p.yaw, 0.0), fin(p.pitch, 0.0), fin(p.roll, 0.0)], None);
        if let Some(pp) = unsafe { path(p.pose_path) } {
            let sig = fin(p.pose_smooth, 0.0).max(0.0) * fin(p.fps, 25.0).max(1.0);
            match e.tracks.pose(pp, sig, p.pose_flags & 4 != 0) {
                Ok(pose) => {
                    info.pose = Some(true);
                    info.pose_track = Some(pose.clone());
                    let (a, c) = pose.at(tf);
                    ang[0] += a[0];
                    if p.pose_flags & 1 == 0 {
                        ang[1] += a[1];
                    }
                    if p.pose_flags & 2 == 0 {
                        ang[2] += a[2];
                    }
                    cam = c;
                }
                Err(m) => {
                    info.pose = Some(false);
                    *warn |= 1;
                    msgs.push(format!("pose: {m}"));
                }
            }
        }
        let cam = if fin(p.cam_dist, 0.0) > 0.0 { p.cam_dist } else { cam.unwrap_or(3.0) };
        sc.view3d = Some([ang[0] * fin(p.lock, 1.0), ang[1], ang[2], cam, fin(p.depth, 0.35), fin(p.tilt, 0.25)]);
    }
    let wants_parts = sc.only | sc.protect | sc.targets.iter().fold(0, |a, b| a | b) != 0;
    let lpath = unsafe { path(p.landmarks_path) };
    if !p.landmarks.is_null() {
        // given for this frame (in canvas pixels)
        let v = unsafe { std::slice::from_raw_parts(p.landmarks, 99) };
        let mut lm = [[0.0; 3]; 33];
        for (i, q) in lm.iter_mut().enumerate() {
            *q = [v[3 * i], v[3 * i + 1], v[3 * i + 2]];
        }
        // no person (NaN): empty part masks, as glitch.py - "only head" then glitches nothing
        sc.lm = Some(lm);
        info.lm = Some(lm.iter().all(|q| q[0].is_finite() && q[1].is_finite()));
    } else if wants_parts || (p.overlay != 0 && lpath.is_some()) {
        match lpath {
            None => {
                *warn |= 4;
                msgs.push("body parts need a landmarks file (estimate_pose.py writes <pose>_landmarks.csv)".into());
            }
            Some(lp) => match e.tracks.landmarks(lp) {
                Ok(t) => {
                    info.lm = Some(true);
                    sc.lm = Some(t.at(tf, w, h));
                }
                Err(m) => {
                    info.lm = Some(false);
                    *warn |= 2;
                    msgs.push(format!("landmarks: {m}"));
                }
            },
        }
    }
    if !msgs.is_empty() {
        if *warn & 16 != 0 {
            let prev = e.last_error.lock().unwrap_or_else(|e| e.into_inner()).to_string_lossy().into_owned();
            msgs.insert(0, prev);
        }
        e.set_error(&msgs.join("; "));
    }
    sc
}

impl Engine {
    fn set_error(&self, m: &str) {
        let mut g = self.last_error.lock().unwrap_or_else(|e| e.into_inner());
        *g = CString::new(m.replace('\0', " ")).unwrap_or_default();
    }
}

/// The most recent problem (a track file that could not be read, ...), "" if none. Valid until
/// the next call on this engine.
///
/// # Safety
/// `e` from gc_engine_new.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gc_last_error(e: *const Engine) -> *const c_char {
    static EMPTY: &[u8] = b"\0";
    if e.is_null() {
        return EMPTY.as_ptr() as *const c_char;
    }
    let g = unsafe { &*e }.last_error.lock().unwrap_or_else(|e| e.into_inner());
    g.as_ptr()
}

/// "head,hands" / "upper" -> part bits, or -1 if a name is unknown (message: gc_last_error).
///
/// # Safety
/// `e` from gc_engine_new (or NULL), `text` a C string (or NULL = no parts).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gc_parse_parts(e: *const Engine, text: *const c_char) -> i32 {
    if text.is_null() {
        return 0;
    }
    let t = unsafe { CStr::from_ptr(text) }.to_string_lossy();
    match masks::parse_parts(&t) {
        Ok(b) => b as i32,
        Err(m) => {
            if !e.is_null() {
                unsafe { &*e }.set_error(&m);
            }
            -1
        }
    }
}

/// "mosh=head; jpeg=torso+arms" -> 13 part masks (glitch.py's effect order) into `out`;
/// 0, or -1 if malformed (message: gc_last_error).
///
/// # Safety
/// `e` from gc_engine_new (or NULL), `text` a C string (or NULL), `out` room for 13 uint32.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gc_parse_targets(e: *const Engine, text: *const c_char, out: *mut u32) -> i32 {
    if out.is_null() {
        return -1;
    }
    let o = unsafe { std::slice::from_raw_parts_mut(out, 13) };
    if text.is_null() {
        o.fill(0);
        return 0;
    }
    let t = unsafe { CStr::from_ptr(text) }.to_string_lossy();
    match masks::parse_targets(&t, &EFFECTS) {
        Ok(v) => {
            o.copy_from_slice(&v);
            0
        }
        Err(m) => {
            if !e.is_null() {
                unsafe { &*e }.set_error(&m);
            }
            -1
        }
    }
}

/// Sizes in bytes of GcParams, GcImage and GcStats, into `out[0..3]`: a host can check its
/// own declarations of the structs against them.
///
/// # Safety
/// `out` has room for 3 ints.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gc_sizes(out: *mut i32) {
    if !out.is_null() {
        let o = unsafe { std::slice::from_raw_parts_mut(out, 3) };
        o[0] = std::mem::size_of::<GcParams>() as i32;
        o[1] = std::mem::size_of::<GcImage>() as i32;
        o[2] = std::mem::size_of::<GcStats>() as i32;
    }
}

// ----------------------------------------------------------------------- shared jobs
// A host can run several copies of one node at once (Resolve does: separate script states
// render frames side by side), each with its own engine. Work that must happen once per node -
// a pose run, a shot scan - is kept here, once per process, and every engine of that node uses
// the same one.
type Shared<T> = std::sync::Arc<std::sync::Mutex<T>>;
static POSE_JOBS: std::sync::LazyLock<std::sync::Mutex<std::collections::HashMap<std::path::PathBuf, Shared<posejob::PoseJob>>>> =
    std::sync::LazyLock::new(Default::default);
static SHOT_SCANS: std::sync::LazyLock<std::sync::Mutex<std::collections::HashMap<String, Shared<shotbox::ShotBox>>>> =
    std::sync::LazyLock::new(Default::default);

fn lock<T>(m: &std::sync::Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

// ----------------------------------------------------------------------- pose estimation
/// A pose estimation run (see posejob): the host's frames `first..=last` are stored in `dir`,
/// then `estimator` (the Glitcher app, or estimate_pose.py run with Python: GLITCHER_PYTHON, else
/// `py -3` on Windows / `python3`) writes dir/pose.csv and dir/pose_landmarks.csv. `run` names
/// the request: a new number starts a new run (frames already stored are reused), the same one
/// continues it, and a run that finished before is not repeated.
#[repr(C)]
pub struct GcPoseJob {
    pub abi: u32,            // = GC_ABI
    pub run: i32,
    pub restart: i32,        // 1 = delete stored frames and landmarks first
    pub max_side: i32,       // longest side of the stored frames (e.g. 1280)
    pub linear: i32,         // 1 = the host's pixels are linear light
    pub _pad: i32,
    pub first: i64,
    pub last: i64,
    pub fps: f64,
    pub focal_mm: f64,       // lens, 0 = unknown (then no camera distance)
    pub sensor_mm: f64,
    pub dir: *const c_char,
    pub estimator: *const c_char,
    pub variant: *const c_char,   // "heavy" / "full" / "lite"
    pub yaw_hints: *const c_char, // "0:0, 120:180"
    pub pitch_hints: *const c_char,
    pub roll_hints: *const c_char,
}

/// Where a pose run is: state 0 none, 1 collecting frames, 2 estimating, 3 done, 4 failed.
#[repr(C)]
pub struct GcPoseStatus {
    pub state: i32,
    pub captured: i32,
    pub total: i32,
    pub progress: i32,       // frames the estimator has done
    pub message: [u8; 512],  // C string
}

unsafe fn cstr(s: *const c_char) -> String {
    if s.is_null() { String::new() } else { unsafe { CStr::from_ptr(s) }.to_string_lossy().into_owned() }
}

/// Sizes in bytes of GcPoseJob and GcPoseStatus into `out[0..2]`.
///
/// # Safety
/// `out` has room for 2 ints.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gc_pose_sizes(out: *mut i32) {
    if !out.is_null() {
        let o = unsafe { std::slice::from_raw_parts_mut(out, 2) };
        o[0] = std::mem::size_of::<GcPoseJob>() as i32;
        o[1] = std::mem::size_of::<GcPoseStatus>() as i32;
    }
}

/// Start or continue the run `j.run`. Returns its state, or -1 (message: gc_last_error), -3 ABI.
///
/// # Safety
/// `e` from gc_engine_new, `j` valid with NULL or C-string fields.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gc_pose_request(e: *const Engine, j: *const GcPoseJob) -> i32 {
    if e.is_null() || j.is_null() {
        return -1;
    }
    let r = catch_unwind(AssertUnwindSafe(|| {
        let (eng, j) = unsafe { (&*e, &*j) };
        if j.abi != GC_ABI {
            return -3;
        }
        let spec = posejob::PoseSpec {
            dir: std::path::PathBuf::from(unsafe { cstr(j.dir) }.trim().trim_matches('"')),
            first: j.first, last: j.last, max_side: j.max_side.clamp(64, 8192) as usize, linear: j.linear != 0,
            estimator: unsafe { cstr(j.estimator) }, fps: fin(j.fps, 25.0).max(1.0),
            variant: match unsafe { cstr(j.variant) }.as_str() { v @ ("full" | "lite") => v.to_string(), _ => "heavy".into() },
            focal_mm: fin(j.focal_mm, 0.0).max(0.0), sensor_mm: fin(j.sensor_mm, 36.0).max(1.0),
            yaw_hints: unsafe { cstr(j.yaw_hints) }, pitch_hints: unsafe { cstr(j.pitch_hints) },
            roll_hints: unsafe { cstr(j.roll_hints) },
        };
        if spec.dir.as_os_str().is_empty() {
            eng.set_error("pose estimation: no folder");
            return -1;
        }
        // the node's run: the one already going for this folder (from this or another copy of
        // the node), else a new one
        let mut reg = lock(&POSE_JOBS);
        if let Some(job) = reg.get(&spec.dir).cloned() {
            let mut j2 = lock(&job);
            if j2.run == j.run as i64 && (j2.spec.first, j2.spec.last) == (spec.first, spec.last) {
                let st = j2.poll().state as i32;
                drop(j2);
                *lock(&eng.pose) = Some(job);
                return st;
            }
        }
        let dir = spec.dir.clone();
        match posejob::PoseJob::new(spec, j.run as i64, j.restart != 0) {
            Ok(mut job) => {
                let st = job.poll().state as i32;
                let job = std::sync::Arc::new(std::sync::Mutex::new(job));
                reg.insert(dir, job.clone());
                *lock(&eng.pose) = Some(job);
                st
            }
            Err(m) => {
                eng.set_error(&format!("pose estimation: {m}"));
                -1
            }
        }
    }));
    r.unwrap_or(-2)
}

/// Frame numbers the run still needs (ascending, at most `max`) into `out`; returns how many.
///
/// # Safety
/// `e` from gc_engine_new, `out` room for `max` int64.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gc_pose_missing(e: *const Engine, out: *mut i64, max: i32) -> i32 {
    if e.is_null() || out.is_null() || max <= 0 {
        return 0;
    }
    let Some(job) = lock(&unsafe { &*e }.pose).clone() else { return 0 };
    let m = lock(&job).missing(max as usize);
    let o = unsafe { std::slice::from_raw_parts_mut(out, m.len()) };
    o.copy_from_slice(&m);
    m.len() as i32
}

/// Give the run frame `frame` (NULL image = the host has nothing there). 1 = stored, 0 = not
/// needed, negative = error.
///
/// # Safety
/// `e` from gc_engine_new; `img` NULL or valid as described on GcImage.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gc_pose_feed(e: *const Engine, frame: i64, img: *const GcImage) -> i32 {
    if e.is_null() {
        return -1;
    }
    let r = catch_unwind(AssertUnwindSafe(|| {
        let Some(job) = lock(&unsafe { &*e }.pose).clone() else { return 0 };
        let mut keep = None;
        let v = if img.is_null() { None } else { unsafe { (*img).view(&mut keep) } };
        lock(&job).feed(frame, v.as_ref()) as i32
    }));
    r.unwrap_or(-2)
}

/// The run's state (also starts the estimator once all frames are stored), into `*st`.
///
/// # Safety
/// `e` from gc_engine_new, `st` valid.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gc_pose_status(e: *const Engine, st: *mut GcPoseStatus) -> i32 {
    if e.is_null() || st.is_null() {
        return -1;
    }
    let r = catch_unwind(AssertUnwindSafe(|| {
        let s = unsafe { &mut *st };
        let Some(job) = lock(&unsafe { &*e }.pose).clone() else {
            s.state = 0;
            s.message[0] = 0;
            return 0;
        };
        let p = lock(&job).poll();
        s.state = p.state as i32;
        s.captured = p.captured as i32;
        s.total = p.total as i32;
        s.progress = p.progress as i32;
        let b = p.message.as_bytes();
        let n = b.len().min(s.message.len() - 1);
        s.message[..n].copy_from_slice(&b[..n]);
        s.message[n] = 0;
        s.state
    }));
    r.unwrap_or(-2)
}

/// glitch.py's glitch_pass: where the glitch made the pixel. A pixel counts when the output is
/// neither the clean subject (over the plate) nor the clean background (nothing, or the plate);
/// grey says how strongly (8-bit premultiplied difference, 2 = nothing .. 64 = white). R = G = B
/// = A = the matte, so it works as luminance or alpha.
const PASS_LO: f32 = 2.0;
const PASS_HI: f32 = 64.0;

fn glitch_matte(out: &View, clean: &View, plate: Option<&View>) -> Frame {
    use rayon::prelude::*;
    let (w, h) = (out.w, out.h);
    let mut m = Frame::new(w, h);
    m.data.par_chunks_mut(4 * w).enumerate().for_each(|(y, row)| {
        for x in 0..w {
            let q = |p: [f32; 4]| p.map(|v| (v.clamp(0.0, 1.0) * 255.0).round());
            let o = q(out.px(x as i64, y as i64));
            let s = clean.px(x as i64, y as i64);
            let (cl, bg) = match plate {
                Some(pv) => {
                    let b = pv.px(x as i64, y as i64);
                    (q(image::over(s, b)), q(b))
                }
                None => (q(s), [0.0; 4]),
            };
            let d = |a: [f32; 4], b: [f32; 4]| (0..4).map(|c| (a[c] - b[c]).abs()).fold(0.0, f32::max);
            let v = ((d(o, cl).min(d(o, bg)) - PASS_LO) / (PASS_HI - PASS_LO)).clamp(0.0, 1.0);
            row[4 * x..4 * x + 4].copy_from_slice(&[v, v, v, v]);
        }
    });
    m
}

/// "Show pose": the landmarks, the 3D camera as a ring with a facing arrow, and the values used.
fn draw_overlay(o: &View, sc: &Scene, info: &SceneInfo, st: &engine::Stats, frame: i64, p: &GcParams, hit: f64) {
    if let Some(lm) = &sc.lm {
        overlay::skeleton(o, lm);
    }
    if let (Some(v), Some(bb)) = (sc.view3d, st.bbox) {
        let sp = sched::Space3D::new(v[0].to_radians(), v[3], v[4], v[1].to_radians(), v[2].to_radians(), v[5]);
        overlay::gizmo(o, &sp, bb);
    }
    let mut lines = vec![format!("FRAME {frame}  TRACK FRAME {}", info.track_frame)];
    match sc.reveal {
        reveal::Reveal::Moving(ap, q, _) => lines.push(format!("{} {:.0}%", if ap { "APPEARING" } else { "VANISHING" }, q * 100.0)),
        reveal::Reveal::Hidden => lines.push("SUBJECT HIDDEN (BEFORE APPEAR / AFTER VANISH)".into()),
        reveal::Reveal::Visible => {}
    }
    if hit > 0.0 {
        lines.push(format!("HIT {hit:.2}"));
    }
    if let Some((x0, y0, x1, y1)) = st.bbox {
        let from = ["THIS FRAME", "GIVEN", "WHOLE SHOT", "FOLLOW"][info.layout_from.clamp(0, 3) as usize];
        lines.push(format!("LAYOUT BOX {x0} {y0} - {x1} {y1} ({from})"));
    }
    match sc.view3d {
        Some(v) => {
            let lock = fin(p.lock, 1.0);
            lines.push(format!("YAW {:.1}  PITCH {:.1}  ROLL {:.1}  CAM {:.1}", v[0], v[1], v[2], v[3]));
            if (lock - 1.0).abs() > 1e-9 {
                lines.push(format!("(YAW X LOCK {lock:.2})"));
            }
        }
        None => lines.push("3D MODE OFF".into()),
    }
    lines.push(match info.pose {
        None => "POSE FILE: NONE (SLIDERS ONLY)".into(),
        Some(true) => "POSE FILE: READ".into(),
        Some(false) => "POSE FILE: NOT READ (SEE CONSOLE)".into(),
    });
    lines.push(match info.lm {
        None => "LANDMARKS: NONE".into(),
        Some(true) => "LANDMARKS: READ".into(),
        Some(false) => "LANDMARKS: NOT READ (SEE CONSOLE)".into(),
    });
    lines.push("SHOW POSE IS ON: TURN IT OFF TO RENDER".into());
    overlay::text(o, &lines, (o.h / 360).max(1));
}

// ----------------------------------------------------------------------- whole-shot layout box
/// Start (or keep) the scan of frames first..=last of a w x h input for node `key`; returns how
/// many frames are still to be fed. Every copy of the node shares it; a different range, size
/// or `generation` (the node's Rescan count) starts over.
///
/// # Safety
/// `e` from gc_engine_new, `key` a C string (NULL = this engine only).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gc_shot_open(e: *const Engine, key: *const c_char, generation: i64, first: i64, last: i64,
                                      w: i32, h: i32) -> i32 {
    if e.is_null() || w <= 0 || h <= 0 || last < first {
        return -1;
    }
    let key = if key.is_null() { format!("engine {:p}", e) } else { unsafe { cstr(key) } };
    let mut reg = lock(&SHOT_SCANS);
    let fresh = || {
        let mut s = shotbox::ShotBox::new(first, last, w as usize, h as usize);
        s.generation = generation;
        std::sync::Arc::new(std::sync::Mutex::new(s))
    };
    let job = match reg.get(&key) {
        Some(j) if { let s = lock(j); s.same(first, last, w as usize, h as usize) && s.generation == generation } => j.clone(),
        _ => {
            let j = fresh();
            reg.insert(key, j.clone());
            j
        }
    };
    let left = lock(&job).remaining() as i32;
    *lock(&unsafe { &*e }.shot) = Some(job);
    left
}

/// Frames the scan still needs (ascending, at most `max`) into `out`; returns how many.
///
/// # Safety
/// `e` from gc_engine_new, `out` room for `max` int64.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gc_shot_missing(e: *const Engine, out: *mut i64, max: i32) -> i32 {
    if e.is_null() || out.is_null() || max <= 0 {
        return 0;
    }
    let Some(job) = lock(&unsafe { &*e }.shot).clone() else { return 0 };
    let m = lock(&job).missing(max as usize);
    unsafe { std::slice::from_raw_parts_mut(out, m.len()) }.copy_from_slice(&m);
    m.len() as i32
}

/// Frame `frame` for the scan (NULL = the host has no picture there).
///
/// # Safety
/// `e` from gc_engine_new; `img` NULL or valid as described on GcImage.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gc_shot_feed(e: *const Engine, frame: i64, img: *const GcImage) -> i32 {
    if e.is_null() {
        return -1;
    }
    let r = catch_unwind(AssertUnwindSafe(|| {
        let Some(job) = lock(&unsafe { &*e }.shot).clone() else { return 0 };
        let mut keep = None;
        let v = if img.is_null() { None } else { unsafe { (*img).view(&mut keep) } };
        lock(&job).feed(frame, v.as_ref());
        1
    }));
    r.unwrap_or(-2)
}

/// The whole shot's box into `out` (x0, y0, x1, y1): 1 = ready, 0 = still scanning, 2 = the
/// shot has no subject anywhere.
///
/// # Safety
/// `e` from gc_engine_new, `out` room for 4 doubles.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gc_shot_result(e: *const Engine, out: *mut f64) -> i32 {
    if e.is_null() || out.is_null() {
        return -1;
    }
    let Some(job) = lock(&unsafe { &*e }.shot).clone() else { return 0 };
    let res = lock(&job).result();
    match res {
        None => 0,
        Some(None) => 2,
        Some(Some((x0, y0, x1, y1))) => {
            unsafe { std::slice::from_raw_parts_mut(out, 4) }.copy_from_slice(&[x0 as f64, y0 as f64, x1 as f64, y1 as f64]);
            1
        }
    }
}

/// A new engine (holds caches between frames). Free it with gc_engine_free.
#[unsafe(no_mangle)]
pub extern "C" fn gc_engine_new() -> *mut Engine {
    Box::into_raw(Box::new(Engine::new()))
}

/// # Safety
/// `e` from gc_engine_new, not used afterwards (NULL is ignored).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gc_engine_free(e: *mut Engine) {
    if !e.is_null() {
        drop(unsafe { Box::from_raw(e) });
    }
}

/// Which earlier frames rendering `frame` will read: writes k (frames back, ascending) into
/// `lags` (room for 24) and returns how many, or a negative error. `max_back` = how many frames
/// exist before this one (e.g. frame - first frame of the clip).
///
/// # Safety
/// Pointers valid; `lags` has room for 24 ints.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gc_needed(e: *const Engine, p: *const GcParams, frame: i64, width: i32, height: i32,
                                   max_back: i32, lags: *mut i32) -> i32 {
    if e.is_null() || p.is_null() || lags.is_null() || width <= 0 || height <= 0 {
        return -1;
    }
    let r = catch_unwind(AssertUnwindSafe(|| {
        let Some((par, _, _, _)) = (unsafe { params_at(None, &*p, frame) }) else { return -3 };
        let mo = fin(unsafe { &*p }.motion, 0.0).clamp(0.0, 1.0);
        let par = match (mo > 0.0, motion_at(unsafe { &*e }, frame)) {
            (true, Some(v)) => with_motion(par, v, mo),
            _ => par,
        };
        let ks = unsafe { &*e }.needed(&par, frame.max(0), width as usize, height as usize, max_back.max(0) as usize);
        let out = unsafe { std::slice::from_raw_parts_mut(lags, MAX_BACK) };
        for (o, k) in out.iter_mut().zip(ks.iter()) {
            *o = *k as i32;
        }
        ks.len() as i32
    }));
    r.unwrap_or(-2)
}

/// Render `frame`: `*cur` -> `*out` (same size; may be the same image). `back` = 24 images,
/// back[k-1] = the source k frames earlier (ptr NULL where missing; NULL array = none).
/// `plate` = background (NULL or ptr NULL = none; another size is resized): the subject is
/// composited onto it and the result is the composite. Returns 0, or negative: -1 bad
/// arguments, -2 internal error (what was written then may be partial), -3 ABI mismatch.
///
/// # Safety
/// Every non-NULL image must be valid as described on GcImage for the duration of the call;
/// the path fields of `p` NULL or C strings.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gc_render(e: *const Engine, p: *const GcParams, frame: i64, cur: *const GcImage,
                                   back: *const GcImage, plate: *const GcImage, out: *const GcImage,
                                   matte: *const GcImage, stats: *mut GcStats) -> i32 {
    // images by pointer, not by value: every FFI (LuaJIT, ctypes) passes pointers the same way
    if e.is_null() || p.is_null() || cur.is_null() || out.is_null() {
        return -1;
    }
    let (cur, out) = unsafe { (*cur, *out) };
    let r = catch_unwind(AssertUnwindSafe(|| {
        let Some((par, hit, hits_ok, rv)) = (unsafe { params_at(Some(&*e), &*p, frame) }) else { return -3 };
        let mo = fin(unsafe { &*p }.motion, 0.0).clamp(0.0, 1.0);
        let mv = motion_at(unsafe { &*e }, frame);
        let par = match (mo > 0.0, mv) {
            (true, Some(v)) => with_motion(par, v, mo),
            _ => par,
        };
        let Some(ofmt) = out.fmt() else { return -1 };
        if out.ptr.is_null() || out.width != cur.width || out.height != cur.height
            || out.stride.unsigned_abs() < 4 * out.width.max(0) as u64 {
            return -1;
        }
        let mut keep_cur = None;
        let Some(c) = (unsafe { cur.view(&mut keep_cur) }) else { return -1 };
        let mut keeps: Vec<Option<Frame>> = (0..MAX_BACK).map(|_| None).collect();
        let hist: Vec<Option<View>> = if back.is_null() {
            vec![None; MAX_BACK]
        } else {
            let b = unsafe { std::slice::from_raw_parts(back, MAX_BACK) };
            b.iter().zip(keeps.iter_mut()).map(|(img, k)| unsafe { img.view(k) }).collect()
        };
        // float32 output: render in place; other formats: render to float, then convert out
        let mut tmp = None;
        let o = if ofmt == PixFmt::F32 {
            unsafe { View::from_raw(out.ptr as *mut f32, out.width as usize, out.height as usize, out.stride as isize) }
        } else {
            tmp = Some(Frame::new(c.w, c.h));
            Some(tmp.as_mut().unwrap().view())
        };
        let Some(o) = o else { return -1 };
        let eng = unsafe { &*e };
        let mut warn = if hits_ok { 0 } else { 16 };
        let mut info = SceneInfo::default();
        let mut sc = unsafe { scene(eng, &*p, frame.max(0), c.w, c.h, &mut warn, &mut info) };
        sc.reveal = rv;
        info.layout_from = if sc.layout.is_some() { 1 } else { 0 };
        // the layout box from the shot scan: the whole shot's, or one that follows the subject
        if unsafe { &*p }.layout_mode == 2 {
            if let Some(scan) = shot_of(eng) {
                let mut s = lock(&scan);
                let follow = fin(unsafe { &*p }.follow, 0.0).max(0.0);
                if follow > 0.0 {
                    let off = fin(unsafe { &*p }.track_offset, 0.0);
                    let scales: Option<Vec<f64>> = info.pose_track.as_ref()
                        .filter(|_| unsafe { &*p }.pose_flags & 8 == 0)
                        .and_then(|pt| (s.first..=s.last).map(|f| pt.scale(f as f64 + off)).collect());
                    if let Some(b) = s.follow(frame.max(0), follow * par.fps, scales.as_deref()) {
                        sc.layout = Some(b);
                        info.layout_from = 3;
                    }
                } else if let Some(Some((x0, y0, x1, y1))) = s.result() {
                    sc.layout = Some((x0 as f64, y0 as f64, x1 as f64, y1 as f64));
                    info.layout_from = 2;
                }
            }
        }
        // the plate, converted / resized to the frame if need be
        let mut keep_plate = None;
        let mut sized = None;
        let pv = if plate.is_null() { None } else { unsafe { (*plate).view(&mut keep_plate) } };
        let pv = match pv {
            Some(v) if !v.same_shape(&c) => {
                warn |= 8;
                sized = Some(resample(&v, c.w, c.h));
                Some(sized.as_mut().unwrap().view())
            }
            v => v,
        };
        // the glitch matte compares with the untouched input: keep a copy if it is rendered in place
        let want_matte = !matte.is_null() && unsafe { !(*matte).ptr.is_null() };
        let clean_copy = (want_matte && std::ptr::eq(c.ptr(), o.ptr())).then(|| Frame::from_view(&c));
        // in place for float32 needs the clean frame to be a separate copy (render handles
        // identical pointers); converted inputs are copies already
        let st = eng.render_scene(&par, frame.max(0), c, &hist, o, pv, &sc);
        if want_matte {
            let mut cc = clean_copy;
            let cv = cc.as_mut().map(|f| f.view()).unwrap_or(c);
            let m = glitch_matte(&o, &cv, pv.as_ref());
            let mi = unsafe { *matte };
            if mi.width as usize == c.w && mi.height as usize == c.h {
                if let Some(mf) = mi.fmt() {
                    unsafe { write_out(&m.view_const(), mi.ptr as *mut u8, mi.stride as isize, mf) };
                }
            }
        }
        if unsafe { &*p }.overlay != 0 {
            draw_overlay(&o, &sc, &info, &st, frame.max(0), unsafe { &*p }, hit);
        }
        drop(sized);
        drop(keep_plate);
        if ofmt != PixFmt::F32 {
            unsafe { write_out(&o, out.ptr as *mut u8, out.stride as isize, ofmt) };
        }
        drop(tmp);
        if !stats.is_null() {
            let s = unsafe { &mut *stats };
            s.ms = st.ms;
            s.echo = st.events[0] as i32;
            s.stutter = st.events[1] as i32;
            s.mosh = st.events[2] as i32;
            s.has_box = st.bbox.is_some() as i32;
            for (o, v) in s.frame_fx.iter_mut().zip(st.frame_fx.iter()) {
                *o = *v as i32;
            }
            if let Some((x0, y0, x1, y1)) = st.bbox {
                s.bbox = [x0 as i32, y0 as i32, x1 as i32, y1 as i32];
            }
            s.flags = st.flags as i32;
            s.warn = warn;
            s.has_view = sc.view3d.is_some() as i32;
            s.hit = hit;
            s.motion = mv.unwrap_or(-1.0);
            s.layout_from = info.layout_from;
            (s.reveal, s.reveal_q) = match rv {
                reveal::Reveal::Visible => (0, 0.0),
                reveal::Reveal::Moving(true, q, _) => (1, q),
                reveal::Reveal::Moving(false, q, _) => (2, q),
                reveal::Reveal::Hidden => (3, 0.0),
            };
            if let Some(v) = sc.view3d {
                s.view = [v[0], v[1], v[2], v[3]];
            }
        }
        0
    }));
    r.unwrap_or(-2)
}

// --------------------------------------------------------------------------- host conversions
// glitch.py's frames are straight alpha, 8-bit or float on a 0..255 scale; the engine works in
// premultiplied float 0..1. Converting in numpy cost more than the render itself (~50 ms a
// 1080p frame on 2 cores, and each frame is converted again for every frame that looks back
// at it), so the host conversions live here, on the engine's threads. The arithmetic is the
// same as glitch_native.to_premul / from_premul (f32, divide by 255, round half up).

/// Straight RGBA -> premultiplied float32: `src` (w x h, `stride` elements per row; format 3 =
/// 8-bit, 0 = float32 on a 0..255 scale, alpha clipped to 0..1) into `dst` at (x0, y0). The
/// rest of `dst` is left as it is (zeros, for a padded canvas). 0 = ok, -1 = bad arguments.
///
/// # Safety
/// `src` addresses `h` rows of `4 * w` elements of its format; `dst` is a valid float32 image.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gc_premultiply(src: *const u8, w: i32, h: i32, stride: i64, format: i32,
                                        dst: *const GcImage, x0: i32, y0: i32) -> i32 {
    let Some(d) = (unsafe { dst.as_ref() }) else { return -1 };
    let fmt = PixFmt::from_i32(format);
    if src.is_null() || d.ptr.is_null() || d.format != 0 || w <= 0 || h <= 0 || x0 < 0 || y0 < 0
        || x0 + w > d.width || y0 + h > d.height || (stride as i128) < 4 * w as i128
        || !matches!(fmt, Some(PixFmt::U8) | Some(PixFmt::F32)) {
        return -1;
    }
    let (w, h, x0, y0) = (w as usize, h as usize, x0 as usize, y0 as usize);
    let (sb, db) = (src as usize, d.ptr as usize);
    let (ss, ds) = (stride as usize, d.stride as usize);
    let u8in = fmt == Some(PixFmt::U8);
    use rayon::prelude::*;
    (0..h).into_par_iter().for_each(|y| {
        let out = unsafe { std::slice::from_raw_parts_mut((db as *mut f32).add((y0 + y) * ds + 4 * x0), 4 * w) };
        if u8in {
            let s = unsafe { std::slice::from_raw_parts((sb as *const u8).add(y * ss), 4 * w) };
            for (o, p) in out.chunks_exact_mut(4).zip(s.chunks_exact(4)) {
                let a = p[3] as f32 / 255.0;
                o[0] = p[0] as f32 / 255.0 * a;
                o[1] = p[1] as f32 / 255.0 * a;
                o[2] = p[2] as f32 / 255.0 * a;
                o[3] = a;
            }
        } else {
            let s = unsafe { std::slice::from_raw_parts((sb as *const f32).add(y * ss), 4 * w) };
            for (o, p) in out.chunks_exact_mut(4).zip(s.chunks_exact(4)) {
                let a = (p[3] / 255.0).clamp(0.0, 1.0);
                o[0] = p[0] / 255.0 * a;
                o[1] = p[1] / 255.0 * a;
                o[2] = p[2] / 255.0 * a;
                o[3] = a;
            }
        }
    });
    0
}

/// Premultiplied float32 -> straight RGBA: the (x0, y0, w, h) part of `src` into `dst` (`stride`
/// elements per row; format 3 = 8-bit rounded and clipped, 0 = float32 on a 0..255 scale, HDR
/// kept). 0 = ok, -1 = bad arguments.
///
/// # Safety
/// `src` is a valid float32 image; `dst` addresses `h` rows of `4 * w` elements of its format.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gc_unpremultiply(src: *const GcImage, x0: i32, y0: i32, w: i32, h: i32,
                                          dst: *mut u8, stride: i64, format: i32) -> i32 {
    let Some(s) = (unsafe { src.as_ref() }) else { return -1 };
    let fmt = PixFmt::from_i32(format);
    if dst.is_null() || s.ptr.is_null() || s.format != 0 || w <= 0 || h <= 0 || x0 < 0 || y0 < 0
        || x0 + w > s.width || y0 + h > s.height || (stride as i128) < 4 * w as i128
        || !matches!(fmt, Some(PixFmt::U8) | Some(PixFmt::F32)) {
        return -1;
    }
    let (w, h, x0, y0) = (w as usize, h as usize, x0 as usize, y0 as usize);
    let (sb, db) = (s.ptr as usize, dst as usize);
    let (ss, ds) = (s.stride as usize, stride as usize);
    let u8out = fmt == Some(PixFmt::U8);
    use rayon::prelude::*;
    (0..h).into_par_iter().for_each(|y| {
        let p = unsafe { std::slice::from_raw_parts((sb as *const f32).add((y0 + y) * ss + 4 * x0), 4 * w) };
        let straight = |q: &[f32]| -> [f32; 4] {
            let a = q[3];
            let k = a.max(1e-8);
            let c = |v: f32| if a > 0.0 { v / k } else { 0.0 };
            [c(q[0]) * 255.0, c(q[1]) * 255.0, c(q[2]) * 255.0, a * 255.0]
        };
        if u8out {
            let o = unsafe { std::slice::from_raw_parts_mut((db as *mut u8).add(y * ds), 4 * w) };
            for (o, q) in o.chunks_exact_mut(4).zip(p.chunks_exact(4)) {
                for (oc, v) in o.iter_mut().zip(straight(q)) {
                    *oc = (v + 0.5).clamp(0.0, 255.0) as u8;
                }
            }
        } else {
            let o = unsafe { std::slice::from_raw_parts_mut((db as *mut f32).add(y * ds), 4 * w) };
            for (o, q) in o.chunks_exact_mut(4).zip(p.chunks_exact(4)) {
                o.copy_from_slice(&straight(q));
            }
        }
    });
    0
}
