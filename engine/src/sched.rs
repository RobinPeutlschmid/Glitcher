//! When glitches happen: the calm/burst activity curve and the events of each effect at a
//! moment - a port of glitch.py's Params / _activity / _bucket / Ctx.events. The same formulas,
//! with deterministic hash draws instead of numpy's generators (so the events are different
//! ones than in the Python version, with the same statistics). In 3D mode each event of a frame
//! effect is also a plane in the subject's glitch volume (Space3D / Pl, glitch.py's _place).

use crate::detmath::{Stream, cos, sin, ss, u, vnoise};
use std::f64::consts::{PI, TAU};

/// Effect ids, in glitch.py's EFFECT_ORDER (the id is part of every random key).
pub const EFFECTS: [&str; 13] = ["jpeg", "blocks", "slices", "wave", "sort", "smear", "split", "tint",
                                 "scanlines", "noise", "echo", "stutter", "mosh"];
pub const ECHO: usize = 10;
pub const STUTTER: usize = 11;
pub const MOSH: usize = 12;
pub const JPEG: usize = 0;
pub const BLOCKS: usize = 1;
pub const SLICES: usize = 2;
pub const WAVE: usize = 3;
pub const SORT: usize = 4;
pub const SMEAR: usize = 5;
pub const SPLIT: usize = 6;
pub const TINT: usize = 7;
pub const SCANLINES: usize = 8;
pub const NOISE: usize = 9;
/// Deepest look-back, in frames.
pub const MAX_BACK: usize = 24;
/// In 3D roughly half the planes face away and half of the rest sit behind the body, so more
/// are spawned to keep a similar amount of visible glitch.
pub const PLANE_BOOST: f64 = 3.0;
/// The effects that are planes in 3D, drawn in front of and behind the body.
pub const LAYERED: [usize; 6] = [JPEG, BLOCKS, SLICES, WAVE, SORT, SMEAR];

/// Events per second at full activity, shortest and longest life [s].
pub fn schedule(fx: usize) -> (f64, f64, f64) {
    match fx {
        0 => (2.5, 0.25, 1.1),
        1 => (16.0, 0.12, 0.6),
        2 => (30.0, 0.08, 0.5),
        3 => (3.5, 0.5, 1.8),
        4 => (8.0, 0.3, 1.2),
        5 => (40.0, 0.25, 1.3),
        ECHO => (1.2, 0.4, 1.4),
        STUTTER => (3.0, 0.15, 0.7),
        MOSH => (0.8, 0.35, 1.0),
        _ => (0.0, 0.1, 0.2),                                  // the continuous effects have no events
    }
}

#[derive(Clone, Debug)]
pub struct Params {
    pub seed: u32,
    pub intensity: f64,  // 0..1 overall strength
    pub fps: f64,
    pub tempo: f64,      // > 1 = shorter, snappier events
    pub density: f64,    // multiplier on event counts
    pub bursts: f64,     // 0 = constant activity, 1 = strong calm/burst cycles
    pub boost: f64,      // 0..1 floor on activity at this moment
    pub gains: [f64; 13], // per effect: 1 default, 0 off, up to ~3
    pub tint: f64,       // 0..1 colour wash
    pub quality: i32,    // JPEG quality 1..100, 0 = from the intensity
    pub bleed: i32,      // px JPEG junk may spill past the silhouette, < 0 = 3 % of the width
}

impl Default for Params {
    fn default() -> Self {
        Params { seed: 0, intensity: 0.5, fps: 25.0, tempo: 1.0, density: 1.0, bursts: 0.5, boost: 0.0,
                 gains: [1.0; 13], tint: 0.0, quality: 0, bleed: -1 }
    }
}

impl Params {
    /// Clamped to the ranges glitch.py accepts.
    pub fn sane(mut self) -> Self {
        let f = |v: f64, d: f64| if v.is_finite() { v } else { d };
        self.intensity = f(self.intensity, 0.5).clamp(0.0, 1.0);
        self.fps = f(self.fps, 25.0).max(1.0);
        self.tempo = f(self.tempo, 1.0).max(0.1);
        self.density = f(self.density, 1.0).max(0.0);
        self.bursts = f(self.bursts, 0.5).clamp(0.0, 1.0);
        self.boost = f(self.boost, 0.0).clamp(0.0, 1.0);
        self.tint = f(self.tint, 0.0).clamp(0.0, 1.0);
        self.quality = self.quality.clamp(0, 100);
        for g in self.gains.iter_mut() {
            *g = f(*g, 1.0).max(0.0);
        }
        self
    }
}

/// "48, 112:0.6, 200:1:0.8" -> hits (frame, strength 0..1, seconds); `len` = default seconds
/// (glitch.py's Timing.parse_hits). Separators: commas, semicolons, spaces.
pub fn parse_hits(text: &str, len: f64) -> Result<Vec<(f64, f64, f64)>, String> {
    let mut out = Vec::new();
    for part in text.split([',', ';', ' ', '\t', '\n']).map(str::trim).filter(|s| !s.is_empty()) {
        let b: Vec<&str> = part.split(':').collect();
        let num = |i: usize, d: f64| -> Result<f64, String> {
            match b.get(i).map(|s| s.trim()).filter(|s| !s.is_empty()) {
                None => Ok(d),
                Some(s) => s.parse::<f64>().map_err(|_| format!("hit '{part}': expected FRAME[:STRENGTH[:SECONDS]]")),
            }
        };
        let (f, st, sec) = (num(0, f64::NAN)?, num(1, 1.0)?, num(2, len)?);
        if !f.is_finite() || st < 0.0 || sec <= 0.0 || b.len() > 3 {
            return Err(format!("hit '{part}': expected FRAME[:STRENGTH[:SECONDS]], strength >= 0, seconds > 0"));
        }
        out.push((f, st, sec));
    }
    out.sort_by(|a, b| a.0.total_cmp(&b.0));
    Ok(out)
}

/// Hit envelope at `frame` (glitch.py's Timing.hit): a quick rise over 0.08 s before the hit,
/// then a smooth fall over its length; the strongest hit wins.
pub fn hit_env(hits: &[(f64, f64, f64)], frame: f64, fps: f64) -> f64 {
    let mut e: f64 = 0.0;
    for &(hf, st, sec) in hits {
        let dt = (frame - hf) / fps;
        if !(-0.08..=sec).contains(&dt) {
            continue;
        }
        let env = if dt < 0.0 { ss(dt, -0.08, 0.0) } else { ss(dt, sec.max(1e-3), 0.0) };
        e = e.max(env * st);
    }
    e
}

impl Params {
    /// The look at a frame with hits (glitch.py's Timing.overrides): activity jumps to full,
    /// intensity rises to at least the hit, every effect gets up to twice the events.
    pub fn with_hit(mut self, e: f64) -> Self {
        let e = e.clamp(0.0, 1.0);
        if e > 0.0 {
            self.intensity = self.intensity.max(e);
            for g in self.gains.iter_mut() {
                *g *= 1.0 + e;
            }
            self.boost = self.boost.max(e);
        }
        self
    }
}

/// How busy the glitch is at time t (0..1), before the intensity fade-in.
pub fn activity(p: &Params, t: f64) -> f64 {
    let s = p.seed as u64;
    let n = 0.6 * vnoise(s, 0, t * 0.9) + 0.4 * vnoise(s, 1, t * 2.3);
    let n = ((n - 0.5) * 1.6 + 0.5).clamp(0.0, 1.0);
    (p.intensity + p.bursts * (n - 0.5) * 1.2).max(p.boost).clamp(0.0, 1.0)
}

/// The camera as seen from inside the subject's glitch volume (glitch.py's Space3D).
///
/// Subject -> camera rotation R = Rpitch . Ryaw (radians): yaw = the subject turned about its
/// vertical axis (+ = its front turns toward screen right; already times `lock`), pitch = camera
/// elevation (+ = above). Roll is a pure 2-D rotation of the picture: the planes are laid out in an
/// upright working frame that is rotated back afterwards. Rows of R are the camera's axes in
/// subject space: `right` (screen x), `up` (screen y, which points down), `cam` (toward the camera).
#[derive(Clone, Debug)]
pub struct Space3D {
    pub yaw: f64,
    pub pitch: f64,
    pub roll: f64,
    pub cam_dist: f64,   // in layout-box heights
    pub depth: f64,      // half-thickness of the volume in box widths
    pub tilt: f64,       // share of planes tilted toward horizontal
    pub right: [f64; 3],
    pub up: [f64; 3],
    pub cam: [f64; 3],
}

impl Space3D {
    pub fn new(yaw: f64, cam_dist: f64, depth: f64, pitch: f64, roll: f64, tilt: f64) -> Self {
        let (c, s) = (cos(yaw), sin(yaw));
        let (cp, sp) = (cos(pitch), sin(pitch));
        let ry = [[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]];
        let rx = [[1.0, 0.0, 0.0], [0.0, cp, sp], [0.0, -sp, cp]];
        let mut r = [[0.0; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                r[i][j] = (0..3).map(|k| rx[i][k] * ry[k][j]).sum();
            }
        }
        let f = |v: f64, d: f64| if v.is_finite() { v } else { d };
        Space3D { yaw, pitch, roll, cam_dist: f(cam_dist, 3.0).max(0.5), depth: f(depth, 0.35).max(0.0),
                  tilt: f(tilt, 0.25).clamp(0.0, 1.0), right: r[0], up: r[1], cam: r[2] }
    }
}

fn dot(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Screen placement of one 3-D plane event (glitch.py's Pl).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Pl {
    pub sx: f64,   // projected centre x
    pub sy: f64,   // projected centre y
    pub s: f64,    // perspective scale at the centre
    pub ax: f64,   // in-plane horizontal axis projected on screen x (foreshortening, signed)
    pub nx: f64,   // plane normal projected on screen x
    pub vis: f64,  // facing weight 0..1 (back faces culled, fades near edge-on)
    pub wf: f64,   // 1 = fully in front of the body, 0 = fully behind
    pub xr: f64,   // projected half-width on screen
    pub rho: f64,  // out-of-plane share of displacements (-0.5..0.5)
    pub z: f64,    // depth of the centre toward the camera (px)
    pub vy: f64,   // vertical foreshortening (1 = seen square-on vertically)
}

/// Which side of the body a pass of a layered effect draws.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Layer {
    Front,
    Back,
}

/// One event alive at this moment.
#[derive(Clone, Debug)]
pub struct Ev {
    pub key: [u64; 4],   // (seed, stream, bucket, j): unique id, the root of its random values
    pub u: f64,          // progress through its life, 0..1
    pub g: f64,          // activity gate 0..1 (times the strength's amplitude; in 3D x facing x layer)
    pub p: [f64; 8],     // 8 fixed uniforms describing it
    pub t0: f64,         // start time [s]
    pub pl: Option<Pl>,  // its plane in 3D mode
}

/// Everything an effect needs to know about this frame (except pixels).
#[derive(Clone)]
pub struct Ctx {
    pub p: Params,
    pub t: f64,
    pub fi: i64,
    /// Layout box (x0, y0, x1, y1), y down, may extend past the canvas.
    pub bbox: (i64, i64, i64, i64),
    pub act: f64,
    /// 3D mode: the camera; None = flat (2-D) placement.
    pub space: Option<Space3D>,
    pub layer: Layer,
}

impl Ctx {
    pub fn new(p: Params, t: f64, fi: i64, bbox: (i64, i64, i64, i64)) -> Self {
        let act = activity(&p, t);
        Ctx { p, t, fi, bbox, act, space: None, layer: Layer::Front }
    }

    pub fn with_space(mut self, space: Option<Space3D>) -> Self {
        self.space = space;
        self
    }

    pub fn layered(&self, layer: Layer) -> Ctx {
        let mut c = self.clone();
        c.layer = layer;
        c
    }

    /// Put an event in subject space and project it through the camera (glitch.py's _place).
    pub fn place(&self, ev: &Ev) -> Option<Pl> {
        let sp = self.space.as_ref()?;
        let (x0, y0, x1, y1) = self.bbox;
        let (bw, bh) = ((x1 - x0) as f64, (y1 - y0) as f64);
        let (cx, cy) = ((x0 + x1) as f64 / 2.0, (y0 + y1) as f64 / 2.0);
        let k = &ev.key;
        let q = |i: u64| u(&[k[0], k[1], k[2], k[3], 50, i]);       // apart from the 2-D genome
        let qs: Vec<f64> = (0..6).map(q).collect();
        let p = [(qs[0] - 0.5) * bw, (qs[1] - 0.5) * bh, (2.0 * qs[2] - 1.0) * sp.depth * bw];
        let al = qs[3] * TAU;
        // tilted planes: the normal's elevation, up to horizontal (floor / ceiling-like slabs)
        let el = if q(6) < sp.tilt { (2.0 * q(7) - 1.0) * (PI / 2.0) } else { 0.0 };
        let (ce, se) = (cos(el), sin(el));
        let (ca, sa) = (cos(al), sin(al));
        let a = [ca, 0.0, -sa];                          // in-plane horizontal axis
        let n = [sa * ce, -se, ca * ce];                 // plane normal (y is down)
        let b = [sa * se, ce, ca * se];                  // in-plane 'vertical' axis
        let (xx, yy, zz) = (dot(&p, &sp.right), dot(&p, &sp.up), dot(&p, &sp.cam));
        let d = sp.cam_dist * bh;
        let s = d / (d - zz).max(0.25 * d);
        let ax = dot(&a, &sp.right);
        let eps = 0.15 * bw;
        Some(Pl { sx: cx + xx * s, sy: cy + yy * s, s, ax, nx: dot(&n, &sp.right),
                  vis: ss(dot(&n, &sp.cam), 0.05, 0.35), wf: ss(zz, -eps, eps),
                  xr: ax.abs() * (0.2 + 0.3 * qs[5]) * bw * s, rho: (2.0 * qs[4] - 1.0) * 0.5, z: zz,
                  vy: dot(&b, &sp.up).abs() })
    }

    pub fn gain(&self, fx: usize) -> f64 {
        self.p.gains[fx]
    }

    /// The frame an event started on.
    pub fn start_frame(&self, ev: &Ev) -> i64 {
        (ev.t0 * self.p.fps - 1e-6).ceil() as i64
    }

    /// A uniform that stays constant for `frames` frames, then jumps (stepped jitter).
    pub fn held(&self, ev: &Ev, ch: u64, frames: i64) -> f64 {
        let k = &ev.key;
        u(&[k[0], k[1], k[2], k[3], ch, (self.fi.div_euclid(frames.max(1))) as u64])
    }

    /// The events of effect `fx` alive at this frame's time, in a fixed order.
    ///
    /// Strength (gain) means more AND bigger: below 1 each event survives with probability =
    /// gain; above 1 a second stream (2x the rate) adds events with probability (gain-1)/2, so
    /// moving a slider adds or removes single events instead of reshuffling them. Amplitude
    /// scales with sqrt(gain).
    ///
    /// In 3D mode the frame effects' events are planes: three times as many are spawned, and
    /// only those facing the camera on this pass's side of the body are kept, their gate times
    /// how much they face it and how far they are on this side. Time effects stay on screen.
    pub fn events(&self, fx: usize) -> Vec<Ev> {
        let raw = self.events_raw(fx);
        if self.space.is_none() || fx >= ECHO {
            return raw;
        }
        raw.into_iter()
            .filter_map(|mut ev| {
                let pl = self.place(&ev)?;
                let w = if self.layer == Layer::Front { pl.wf } else { 1.0 - pl.wf };
                ev.g *= pl.vis * w;
                ev.pl = Some(pl);
                (ev.g > 0.01).then_some(ev)
            })
            .collect()
    }

    fn events_raw(&self, fx: usize) -> Vec<Ev> {
        let p = &self.p;
        let (rate, lo, hi) = schedule(fx);
        let (lo, hi, mut rate) = (lo / p.tempo, hi / p.tempo, rate * p.density);
        if self.space.is_some() && fx < ECHO {
            rate *= PLANE_BOOST;
        }
        let gf = self.gain(fx);
        let amp_g = if gf != 1.0 { gf.sqrt() } else { 1.0 };
        let fid = fx as u64 + 1;
        let mut streams = vec![(fid, rate, if gf < 1.0 { Some(gf) } else { None })];
        if gf > 1.0 {
            streams.push((fid + 100, rate * 2.0, Some(((gf - 1.0) / 2.0).min(1.0))));
        }
        let t = self.t;
        let seed = p.seed as u64;
        let mut out = Vec::new();
        for (sid, rt, keep) in streams {
            let k0 = ((t - hi).floor() as i64).max(0);
            for k in k0..=(t.floor() as i64) {
                let mut s = Stream::new(&[seed, sid, k as u64]);
                let n = s.poisson(rt);
                for j in 0..n as u64 {
                    let start = k as f64 + s.next_f64();
                    let life = lo + (hi - lo) * s.next_f64();
                    let thr = s.next_f64();
                    let mut pp = [0.0; 8];
                    for v in pp.iter_mut() {
                        *v = s.next_f64();
                    }
                    if !(start <= t && t < start + life) {
                        continue;
                    }
                    if let Some(kp) = keep {
                        if u(&[seed, sid, k as u64, j, 60]) >= kp {
                            continue;
                        }
                    }
                    let g = ss(self.act, thr - 0.12, thr + 0.12) * (p.intensity * 5.0).min(1.0) * amp_g;
                    if g > 0.01 {
                        out.push(Ev { key: [seed, sid, k as u64, j], u: (t - start) / life, g, p: pp, t0: start, pl: None });
                    }
                }
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ctx(p: &Params, fi: i64) -> Ctx {
        Ctx::new(p.clone(), fi as f64 / p.fps, fi, (100, 50, 300, 450))
    }

    #[test]
    fn event_rate_matches_the_schedule() {
        // at full, flat activity every event passes the gate: the live count averages rate x life
        let p = Params { intensity: 1.0, bursts: 0.0, ..Default::default() }.sane();
        let (rate, lo, hi) = schedule(ECHO);
        let frames = 25 * 400;
        let n: usize = (0..frames).map(|f| ctx(&p, f).events(ECHO).len()).sum();
        let mean = n as f64 / frames as f64;
        let want = rate * (lo + hi) / 2.0;
        assert!((mean - want).abs() < 0.12 * want, "{mean} vs {want}");
    }

    #[test]
    fn events_are_stable_across_frames_and_calls() {
        let p = Params { seed: 42, intensity: 0.8, ..Default::default() }.sane();
        let a = ctx(&p, 100).events(MOSH);
        let b = ctx(&p, 100).events(MOSH);
        assert_eq!(format!("{a:?}"), format!("{b:?}"));
        // an event alive on two frames keeps its id and genome, only its progress moves
        for f in 0..2000 {
            let (x, y) = (ctx(&p, f).events(STUTTER), ctx(&p, f + 1).events(STUTTER));
            for e in &x {
                if let Some(e2) = y.iter().find(|e2| e2.key == e.key) {
                    assert_eq!(e.p, e2.p);
                    assert!(e2.u > e.u);
                }
            }
        }
    }

    #[test]
    fn gain_adds_and_removes_events_without_reshuffling() {
        let base = Params { seed: 7, intensity: 1.0, bursts: 0.0, ..Default::default() };
        let mut lo = base.clone();
        lo.gains[ECHO] = 0.5;
        let mut hi = base.clone();
        hi.gains[ECHO] = 3.0;
        let mut off = base.clone();
        off.gains[ECHO] = 0.0;
        for f in (0..5000).step_by(7) {
            let keys = |p: &Params| ctx(&p.clone().sane(), f).events(ECHO).iter().map(|e| e.key).collect::<Vec<_>>();
            let (k1, kl, kh, k0) = (keys(&base), keys(&lo), keys(&hi), keys(&off));
            assert!(kl.iter().all(|k| k1.contains(k)));           // fewer: a subset
            assert!(k1.iter().all(|k| kh.contains(k)));           // more: a superset
            assert!(k0.is_empty());
        }
    }

    #[test]
    fn planes_split_into_front_and_back_and_face_the_camera() {
        let p = Params { seed: 3, intensity: 1.0, bursts: 0.0, ..Default::default() }.sane();
        let mut front = 0usize;
        let mut back = 0usize;
        let mut flat = 0usize;
        for f in (0..3000).step_by(5) {
            let c = ctx(&p, f).with_space(Some(Space3D::new(0.6, 3.0, 0.35, 0.2, 0.0, 0.25)));
            let (fe, be) = (c.layered(Layer::Front).events(SLICES), c.layered(Layer::Back).events(SLICES));
            for e in fe.iter().chain(be.iter()) {
                let pl = e.pl.unwrap();
                assert!(pl.vis > 0.0 && pl.s > 0.0 && pl.xr >= 0.0 && pl.vy <= 1.0 + 1e-9);
            }
            front += fe.len();
            back += be.len();
            flat += ctx(&p, f).events(SLICES).len();
            // the same event on both sides only where it straddles the body
            for e in &fe {
                if let Some(b) = be.iter().find(|b| b.key == e.key) {
                    let w = e.pl.unwrap().wf;
                    assert!(w > 0.0 && w < 1.0);
                    let _ = b;
                }
            }
            // time effects are not placed
            assert!(c.events(ECHO).iter().all(|e| e.pl.is_none()));
        }
        // 3x the planes, about half facing, split over two sides: a similar visible count
        let ratio = (front + back) as f64 / flat as f64;
        assert!(ratio > 0.8 && ratio < 2.2, "{ratio}");
        assert!(front > 0 && back > 0);
    }

    #[test]
    fn camera_axes_are_orthonormal() {
        let s = Space3D::new(1.1, 3.0, 0.35, -0.4, 0.3, 0.25);
        for (a, b) in [(&s.right, &s.up), (&s.up, &s.cam), (&s.right, &s.cam)] {
            assert!(dot(a, b).abs() < 1e-9);
        }
        for a in [&s.right, &s.up, &s.cam] {
            assert!((dot(a, a) - 1.0).abs() < 1e-9);
        }
    }

    #[test]
    fn hits_parse_and_shape_like_glitch_py() {
        let h = parse_hits("48, 112:0.6;200:1:0.8", 0.5).unwrap();
        assert_eq!(h, vec![(48.0, 1.0, 0.5), (112.0, 0.6, 0.5), (200.0, 1.0, 0.8)]);
        assert!(parse_hits("4x", 0.5).is_err() && parse_hits("4:1:0", 0.5).is_err());
        // glitch.py: Timing(hits=[(48, 1, 0.5)], fps=25).hit(f)
        let want = [(46.0, 0.0), (47.0, 0.5), (48.0, 1.0), (50.0, 0.93139), (55.0, 0.41043), (60.0, 0.00467), (61.0, 0.0)];
        for (f, v) in want {
            let e = hit_env(&h[..1], f, 25.0);
            assert!((e - v).abs() < 1e-3, "frame {f}: {e} vs {v}");
        }
        let p = Params { intensity: 0.3, ..Default::default() }.with_hit(0.8);
        assert!((p.intensity - 0.8).abs() < 1e-12 && (p.gains[0] - 1.8).abs() < 1e-12 && (p.boost - 0.8).abs() < 1e-12);
    }

    #[test]
    fn zero_intensity_is_calm() {
        let p = Params { intensity: 0.0, ..Default::default() }.sane();
        assert!((0..500).all(|f| ctx(&p, f).events(ECHO).is_empty()));
    }
}
