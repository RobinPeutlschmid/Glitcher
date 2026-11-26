//! One engine per node / renderer: the caches (motion between frames, track files) plus the
//! render call, which runs glitch.py's `glitch()` pipeline on a frame:
//!
//!   subject (cutout) -> time effects (echo, stutter, mosh; on the subject, from earlier frames)
//!   -> over the plate, if any -> frame effects (2-D; or in 3D: the plane effects in front of /
//!   behind the body in an upright working frame, then split, tint, scanlines, noise on screen)
//!   -> limited to the allowed body parts -> limited to near the subject on a plate -> over the
//!   plate again (no holes).
//!
//! One difference to glitch.py with a plate: the time effects run on the subject, not on the
//! composite. Then the Fusion node needs no earlier plate frames (half the fetching), and echo
//! trails show on the plate (behind an opaque composite they are hidden).

use crate::fx_frame::{Area, Before};
use std::sync::Arc;
use crate::fx_time::{Hist, echo, mosh, needed, stutter};
use crate::image::{Frame, Rect, View, alpha_box, boxes, copy, over};
use crate::layers::{Mask, Roll, Tiles, mix, rotate_back, rotate_in, run_layered, run_masked};
use crate::masks::{Allow, Landmarks, PartMask, Spill};
use crate::motion::Caches;
use crate::sched::{Ctx, ECHO, LAYERED, MOSH, Params, STUTTER, Space3D};
use crate::reveal::{self, Map, Reveal};
use crate::tracks::TrackCache;
use rayon::prelude::*;
use std::sync::Mutex;
use std::time::Instant;

#[derive(Default)]
pub struct Engine {
    cache: Mutex<Caches>,
    pub tracks: TrackCache,
    pub last_error: Mutex<std::ffi::CString>,
    /// The pose estimation run this node works on (see posejob). Shared, by folder, with every
    /// other engine in the process: a host may run several copies of a node (Resolve does).
    pub pose: Mutex<Option<std::sync::Arc<Mutex<crate::posejob::PoseJob>>>>,
    /// The whole-shot layout box scan (see shotbox), shared the same way, by node.
    pub shot: Mutex<Option<std::sync::Arc<Mutex<crate::shotbox::ShotBox>>>>,
}

/// What, besides the look, shapes this frame: the 3D camera, body parts, the plate's spill.
#[derive(Clone, Debug)]
pub struct Scene {
    /// 3D mode: yaw (x lock), pitch, roll in degrees, then cam_dist, depth, tilt.
    pub view3d: Option<[f64; 6]>,
    /// This frame's body landmarks in canvas pixels (needed for any part option).
    pub lm: Option<Landmarks>,
    /// Part bits the glitch may touch (0 = everywhere) / must leave clean / per effect.
    pub only: u32,
    pub protect: u32,
    pub targets: [u32; 13],
    pub feather: f64,
    /// With a plate: how far the glitch may damage it around the subject (x subject size).
    pub spill: f64,
    /// The layout box to use instead of this frame's subject box (the whole shot's).
    pub layout: Option<(f64, f64, f64, f64)>,
    /// Grow the layout box by this share of its size on each side.
    pub box_grow: f64,
    /// Appear / vanish at this frame (seed of the transition patterns: the look's seed).
    pub reveal: crate::reveal::Reveal,
}

impl Default for Scene {
    fn default() -> Self {
        Scene { view3d: None, lm: None, only: 0, protect: 0, targets: [0; 13], feather: 1.0, spill: 0.05, layout: None,
                box_grow: 0.0, reveal: crate::reveal::Reveal::Visible }
    }
}

pub const FLAG_3D: u32 = 1;
pub const FLAG_ROLLED: u32 = 2;
pub const FLAG_PLATE: u32 = 4;
pub const FLAG_PARTS: u32 = 8;

#[derive(Debug, Default, Clone, Copy)]
pub struct Stats {
    pub ms: f64,
    pub events: [usize; 3],              // echo, stutter, mosh events applied
    pub frame_fx: [usize; 10],           // jpeg .. noise: events / effects applied
    pub bbox: Option<(i64, i64, i64, i64)>,
    pub flags: u32,
}

/// The layout box the effects are placed in: the subject (alpha above 8/255, or any alpha if
/// it is all faint, e.g. motion blur), at least 2 px each way. None: nothing visible.
pub fn layout_box(cur: &View) -> Option<(i64, i64, i64, i64)> {
    let (x0, y0, x1, y1) = alpha_box(cur, 8.0 / 255.0).or_else(|| alpha_box(cur, 0.0))?;
    Some((x0, y0, x1.max(x0 + 2), y1.max(y0 + 2)))
}

fn tm(v: &[Option<PartMask>], i: usize) -> Option<&dyn Mask> {
    v[i].as_ref().map(|m| m as &dyn Mask)
}

/// Every pixel of `r` in `out` through `f(x, y, pixel)`, rows in parallel.
fn each_px(out: &View, r: Rect, f: impl Fn(usize, usize, [f32; 4]) -> [f32; 4] + Sync) {
    (r.y0..r.y1).into_par_iter().for_each(|y| {
        // SAFETY: one thread per row
        let row = unsafe { out.row_mut(y) };
        for x in r.x0..r.x1 {
            let o = [row[4 * x], row[4 * x + 1], row[4 * x + 2], row[4 * x + 3]];
            row[4 * x..4 * x + 4].copy_from_slice(&f(x, y, o));
        }
    });
}

impl Engine {
    pub fn new() -> Self {
        Self::default()
    }

    /// Earlier frames (k back) this frame will read, given `max_back` frames exist before it.
    pub fn needed(&self, p: &Params, frame: i64, w: usize, h: usize, max_back: usize) -> Vec<usize> {
        let c = Ctx::new(p.clone(), frame as f64 / p.fps, frame, (0, 0, w as i64, h as i64));
        needed(&c, w, h, max_back)
    }

    /// Render frame `frame` of `cur` into `out` (same size) in 2-D, without plate or parts.
    pub fn render(&self, p: &Params, frame: i64, cur: View, back: &[Option<View>], out: View) -> Stats {
        self.render_scene(p, frame, cur, back, out, None, &Scene::default())
    }

    /// Render frame `frame` of `cur` into `out` (same size; may be `cur` itself). `back[k-1]` =
    /// the *source* frame k frames earlier (None if not available; those of another size are
    /// ignored). `plate` = background, same size (the result is then the composite).
    #[allow(clippy::too_many_arguments)]
    pub fn render_scene(&self, p: &Params, frame: i64, cur: View, back: &[Option<View>], out: View,
                        plate: Option<View>, sc: &Scene) -> Stats {
        let t0 = Instant::now();
        let prof = std::env::var_os("GLITCH_CORE_PROFILE").is_some();
        let lap = |what: &str| if prof { eprintln!("  {:>7.1} ms  {what}", t0.elapsed().as_secs_f64() * 1e3) };
        let mut stats = Stats::default();
        let (w, h) = (out.w, out.h);
        // out may be cur itself: then keep a clean copy, the effects read the untouched frame
        let mut keep = None;
        let cur = if std::ptr::eq(cur.ptr(), out.ptr()) {
            keep = Some(Frame::from_view(&cur));
            keep.as_mut().unwrap().view()
        } else {
            copy(&cur, &out);
            cur
        };
        let plate = plate.filter(|v| v.same_shape(&cur));
        let back: Vec<Option<View>> = back.iter().map(|b| b.filter(|v| v.same_shape(&cur))).collect();
        let (lb, cut_box) = boxes(&cur, 8.0 / 255.0);
        // before an appear / after a vanish the subject isn't there: as an empty frame
        let hidden = sc.reveal == Reveal::Hidden;
        let Some((x0, y0, x1, y1)) = lb.or_else(|| alpha_box(&cur, 0.0)).filter(|_| !hidden) else {
            if hidden {
                each_px(&out, Rect::new(0, h, 0, w), |_, _, _| [0.0; 4]);
            }
            if let Some(pv) = plate {                       // subject out of shot: the clean plate
                copy(&pv, &out);
                stats.flags |= FLAG_PLATE;
            }
            stats.ms = t0.elapsed().as_secs_f64() * 1e3;
            return stats;
        };
        // the layout box: this frame's subject, or the one given (the whole shot's), grown
        let (x0, y0, x1, y1) = match sc.layout {
            Some((a, b, c, d)) => (a.round() as i64, b.round() as i64, c.round() as i64, d.round() as i64),
            None => (x0, y0, x1, y1),
        };
        let (gx, gy) = (((x1 - x0) as f64 * sc.box_grow).round() as i64, ((y1 - y0) as f64 * sc.box_grow).round() as i64);
        let (x0, y0, x1, y1) = (x0 - gx, y0 - gy, x1 + gx, y1 + gy);
        let bbox = (x0, y0, x1.max(x0 + 2), y1.max(y0 + 2));
        stats.bbox = Some(bbox);
        // appear / vanish: the subject partly built or gone, laid out over the layout box; from
        // here on it is "the subject" (history frames stay as they were, as in glitch.py)
        let mut revealed = None;
        let mut reveal_map = None;
        let (cur, cut_box) = match sc.reveal {
            Reveal::Moving(appear, q, idx) => {
                let map = Map::new(p.seed, idx);
                let mut f = reveal::start(&cur, cut_box, appear, q, &map, bbox);
                reveal_map = Some((appear, q, map));
                let v = f.view();
                copy(&v, &out);
                let cb = boxes(&v, 0.0).1;
                revealed = Some(f);
                (v, cb)
            }
            _ => (cur, cut_box),
        };
        let space = sc.view3d.map(|v| Space3D::new(v[0].to_radians(), v[3], v[4], v[1].to_radians(),
                                                   v[2].to_radians(), v[5]));
        let c = Ctx::new(p.clone(), frame as f64 / p.fps, frame, bbox).with_space(space.clone());

        // time effects first, on the clean subject (glitch.py's order: echo, stutter, mosh)
        let hist = Hist { cur, back: &back };
        let mut wrote = Rect::new(0, 0, 0, 0);
        if c.gain(ECHO) > 0.0 {
            stats.events[0] = echo(&c, &hist, &out, &mut wrote);
        }
        if c.gain(STUTTER) > 0.0 {
            stats.events[1] = stutter(&c, &hist, &out, &mut wrote);
        }
        if c.gain(MOSH) > 0.0 {
            stats.events[2] = mosh(&c, &hist, &out, &self.cache, &mut wrote);
        }
        let mut cbox = cut_box.union(wrote);
        lap("copy, boxes, time effects");

        // the plate: everything from here on works on the composite, but only near the subject
        // (further out the spill mask puts the clean plate back anyway)
        let spill = plate.map(|_| Spill::new(cur, sc.spill, bbox));
        let mut limit = None;
        if let (Some(pv), Some(sp)) = (plate, &spill) {
            stats.flags |= FLAG_PLATE;
            each_px(&out, Rect::new(0, h, 0, w), |x, y, o| over(o, pv.px(x as i64, y as i64)));
            let (bw, bh) = ((bbox.2 - bbox.0) as usize, (bbox.3 - bbox.1) as usize);
            let bleed = if p.bleed < 0 { w * 3 / 100 } else { p.bleed as usize };
            // effects read up to ~a quarter box past where they write
            cbox = cut_box.grow(sp.reach() + bw.max(bh) / 4 + bleed + 8, w, h);
            limit = Some(cut_box.grow(sp.reach() + 2, w, h));
            lap("plate composite, spill mask");
        }

        // body parts
        let lm = sc.lm.filter(|_| sc.only | sc.protect | sc.targets.iter().fold(0, |a, b| a | b) != 0);
        if lm.is_some() {
            stats.flags |= FLAG_PARTS;
        }
        let pmask = |l: &Landmarks, parts: u32| PartMask::new(l, parts, w, h, sc.feather);
        let screen_targets: Vec<Option<PartMask>> =
            (0..13).map(|i| lm.as_ref().filter(|_| sc.targets[i] != 0).map(|l| pmask(l, sc.targets[i]))).collect();

        match &space {
            None => {
                let mut area = Area::new(cbox);
                area.limit = limit;
                for fx in 0..10 {
                    stats.frame_fx[fx] = run_masked(fx, &c, &out, &mut area, tm(&screen_targets, fx));
                    lap(crate::sched::EFFECTS[fx]);
                }
                cbox = area.cbox;
            }
            Some(sp) => {
                stats.flags |= FLAG_3D;
                let roll_deg = sp.roll.to_degrees();
                if roll_deg.abs() > 1e-3 {
                    // the planes live in an upright working frame; only what they change is
                    // turned back, everything else stays exact
                    stats.flags |= FLAG_ROLLED;
                    let roll = Roll::new(roll_deg, (bbox.0 + bbox.2) as f64 / 2.0, (bbox.1 + bbox.3) as f64 / 2.0);
                    let mut up = rotate_in(&out, cbox, &roll);
                    lap("rotate in");
                    let wv = up.frame.view();
                    let wbox = up.area;
                    let base = Arc::new(Before::new(&wv));
                    // part masks for the work frame: the landmarks turned with it
                    let work_targets: Vec<Option<PartMask>> = (0..13)
                        .map(|i| {
                            lm.as_ref().filter(|_| sc.targets[i] != 0).map(|l| {
                                let mut t = *l;
                                for q in t.iter_mut() {
                                    let (a, b) = roll.to_work(q[0], q[1]);
                                    (q[0], q[1]) = (a, b);
                                }
                                pmask(&t, sc.targets[i])
                            })
                        })
                        .collect();
                    let mut area = Area::new(wbox);
                    area.base = Some(base.clone());
                    for fx in LAYERED {
                        stats.frame_fx[fx] = run_layered(fx, &c, &wv, &mut area, tm(&work_targets, fx));
                    }
                    lap("layered effects");
                    let mut tiles = Tiles::new(w, h);
                    tiles.mark(&area.dirty);
                    let written = rotate_back(&mut up, &base, &tiles, &roll, &out);
                    drop(up);
                    lap("rotate back");
                    cbox = cbox.union(written);
                } else {
                    let mut area = Area::new(cbox);
                    for fx in LAYERED {
                        stats.frame_fx[fx] = run_layered(fx, &c, &out, &mut area, tm(&screen_targets, fx));
                    }
                    cbox = area.cbox;
                }
                let mut area = Area::new(cbox);
                area.limit = limit;
                for fx in 6..10 {
                    stats.frame_fx[fx] = run_masked(fx, &c, &out, &mut area, tm(&screen_targets, fx));
                }
                cbox = area.cbox;
            }
        }

        lap("frame effects");
        // the glitch as a whole: only the allowed parts, only near the subject on a plate
        let allow = lm.as_ref().filter(|_| sc.only != 0 || sc.protect != 0).map(|l| Allow {
            only: (sc.only != 0).then(|| pmask(l, sc.only)),
            protect: (sc.protect != 0).then(|| pmask(l, sc.protect)),
        });
        if allow.is_some() || plate.is_some() || reveal_map.is_some() {
            let clean = |x: usize, y: usize| {
                let s = cur.px(x as i64, y as i64);
                match plate {
                    Some(pv) => over(s, pv.px(x as i64, y as i64)),
                    None => s,
                }
            };
            let lim = limit.unwrap_or(cbox);
            each_px(&out, cbox, |x, y, mut o| {
                let cl = clean(x, y);
                if x < lim.x0 || x >= lim.x1 || y < lim.y0 || y >= lim.y1 {
                    return cl;                        // beyond the spill's reach: the clean plate
                }
                if let Some(a) = &allow {
                    o = mix(cl, o, a.at(x, y));
                }
                let rm = reveal_map.as_ref().map(|(ap, q, m)| reveal::end_mul(*ap, *q, m, x, y, bbox));
                if let (Some(pv), Some(sp)) = (plate, &spill) {
                    o = mix(cl, o, sp.at(x, y));
                    let pp = pv.px(x as i64, y as i64);
                    if let Some(k) = rm {
                        o = mix(pp, o, k);                     // not built yet / gone: the clean plate
                    }
                    o = over(o, pp);                           // no holes
                } else if let Some(k) = rm {
                    o = [o[0] * k, o[1] * k, o[2] * k, o[3] * k];   // glitch only where the subject is
                }
                o
            });
        }
        lap("final blend");
        drop(revealed);
        drop(keep);
        stats.ms = t0.elapsed().as_secs_f64() * 1e3;
        stats
    }
}
