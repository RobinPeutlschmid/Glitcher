//! The time effects - echo, stutter, mosh - ported from glitch.py. They read earlier *source*
//! frames (never earlier output), so every frame can be rendered on its own, in any order.

use crate::detmath::{env, iround};
use crate::image::{Rect, View, over, scale};
use crate::motion::{Caches, MR, motion_cached};
use crate::sched::{Ctx, ECHO, Ev, MAX_BACK, MOSH, STUTTER};
use rayon::prelude::*;
use std::sync::Mutex;

/// Earlier source frames: `get(k)` = k frames back (0 = this frame, clean), None if missing.
pub struct Hist<'a> {
    pub cur: View,
    pub back: &'a [Option<View>],      // index k-1 = k frames back
}

impl Hist<'_> {
    pub fn get(&self, k: usize) -> Option<View> {
        if k == 0 {
            return Some(self.cur);
        }
        self.back.get(k - 1).copied().flatten()
    }
    /// Largest k <= want (and <= MAX_BACK) with a frame, 0 if none.
    pub fn depth(&self, want: i64) -> usize {
        let want = want.clamp(0, MAX_BACK as i64) as usize;
        (1..=want).rev().find(|&k| self.get(k).is_some()).unwrap_or(0)
    }
}

/// (ya, yb, xa, xb) for a time effect: a horizontal band across the subject box (widened by
/// `wide` box widths each side), or with probability `full_p` the whole box.
#[allow(clippy::too_many_arguments)]
fn band(c: &Ctx, ev: &Ev, wide: f64, full_p: f64, lo: f64, hi: f64, w: usize, h: usize) -> (usize, usize, usize, usize) {
    let (x0, y0, x1, y1) = c.bbox;
    let (bw, bh) = ((x1 - x0) as f64, (y1 - y0) as f64);
    let p = &ev.p;
    let (ya, yb) = if p[2] < full_p {
        (y0 as f64 - 0.08 * bh, y1 as f64 + 0.08 * bh)
    } else {
        let hb = bh * (lo + (hi - lo) * p[1]);
        let ya = y0 as f64 + p[0] * (bh - hb).max(1.0);
        (ya, ya + hb)
    };
    let (xa, xb) = (x0 as f64 - wide * bw, x1 as f64 + wide * bw);
    let cl = |v: f64, m: usize| v.clamp(0.0, m as f64) as usize;
    (cl(ya, h), cl(yb, h), cl(xa, w), cl(xb, w))
}

fn box_wh(c: &Ctx) -> (f64, f64) {
    ((c.bbox.2 - c.bbox.0) as f64, (c.bbox.3 - c.bbox.1) as f64)
}

/// Write `f(x, y, old pixel)` into every pixel of a region of `out`, rows in parallel.
fn for_region(out: &View, ya: usize, yb: usize, xa: usize, xb: usize, f: impl Fn(i64, i64, [f32; 4]) -> [f32; 4] + Sync) {
    (ya..yb).into_par_iter().for_each(|y| {
        // SAFETY: one thread per row
        let row = unsafe { out.row_mut(y) };
        for x in xa..xb {
            let i = 4 * x;
            let o = [row[i], row[i + 1], row[i + 2], row[i + 3]];
            let n = f(x as i64, y as i64, o);
            row[i..i + 4].copy_from_slice(&n);
        }
    });
}

// ----------------------------------------------------------------------------------- echo
struct EchoPlan {
    lags: Vec<usize>,
    region: (usize, usize, usize, usize),
    off: Vec<(i64, i64)>,      // (dy, dx) per layer
    alpha: Vec<f32>,           // alpha factor per layer
    mode: usize,               // 0/1 natural, 2 cyan, 3 magenta
    over_front: bool,
}

fn echo_plan(c: &Ctx, ev: &Ev, avail: &dyn Fn(usize) -> bool, w: usize, h: usize) -> Option<EchoPlan> {
    let p = &ev.p;
    let st = env(ev.u) * ev.g;
    if st < 0.05 {
        return None;
    }
    let n = 2 + (p[3] * 3.0) as usize;                         // 2-4 echoes
    let step = 1 + (p[4] * 3.0) as usize;                      // 1-3 frames apart
    let mut lags: Vec<usize> = Vec::new();
    for i in 0..n {
        let k = step * (i + 1);
        if k <= MAX_BACK && avail(k) && !lags.contains(&k) {
            lags.push(k);
        }
    }
    if lags.is_empty() {
        return None;
    }
    let region = band(c, ev, 0.5, 0.3, 0.2, 0.8, w, h);
    if region.1 - region.0 < 2 || region.3 - region.2 < 2 {
        return None;
    }
    let (bw, bh) = box_wh(c);
    let amp = 0.4 + 0.6 * c.p.intensity;
    let ddx = (p[5] - 0.5) * 0.08 * bw * amp;
    let ddy = (p[6] - 0.5) * 0.03 * bh * amp;
    let l = lags.len();
    Some(EchoPlan {
        off: (0..l).map(|i| (iround(ddy * (i + 1) as f64), iround(ddx * (i + 1) as f64))).collect(),
        alpha: (0..l).map(|i| (st * 0.85 * (1.0 - i as f64 / (l + 1) as f64)).min(1.0) as f32).collect(),
        lags,
        region,
        mode: (p[7] * 4.0) as usize,
        over_front: p[0] > 0.7,
    })
}

/// Ghost trails: fading copies of earlier frames behind (sometimes over) the body, drifting a
/// little so they read even when the subject stands still. Returns the events applied.
pub fn echo(c: &Ctx, hist: &Hist, out: &View, wrote: &mut Rect) -> usize {
    let (w, h) = (out.w, out.h);
    let mut n = 0;
    for ev in c.events(ECHO) {
        let Some(pl) = echo_plan(c, &ev, &|k| hist.depth(k as i64) == k, w, h) else { continue };
        let frames: Vec<View> = pl.lags.iter().map(|&k| hist.get(k).unwrap()).collect();
        let (ya, yb, xa, xb) = pl.region;
        *wrote = wrote.union(Rect::new(ya, yb, xa, xb));
        for_region(out, ya, yb, xa, xb, |x, y, o| {
            let mut ghost = [0f32; 4];
            let mut nearest = [0f32; 4];
            for i in (0..frames.len()).rev() {                  // far echoes first, near ones on top
                let (dy, dx) = pl.off[i];
                let mut e = frames[i].px(x - dx, y - dy);
                match pl.mode {
                    2 => e[0] *= 0.15,
                    3 => e[1] *= 0.15,
                    _ => {}
                }
                let e = scale(e, pl.alpha[i]);
                ghost = over(e, ghost);
                if i == 0 {
                    nearest = e;
                }
            }
            let sl = over(o, ghost);                             // trails behind the current body
            if pl.over_front { over(scale(nearest, 0.4), sl) } else { sl }   // + the nearest, faintly over it
        });
        n += 1;
    }
    n
}

// -------------------------------------------------------------------------------- stutter
struct StutterPlan {
    lag: usize,
    region: (usize, usize, usize, usize),
    dx: i64,
}

fn stutter_plan(c: &Ctx, ev: &Ev, depth: &dyn Fn(i64) -> usize, w: usize, h: usize) -> Option<StutterPlan> {
    if ev.g < 0.3 {
        return None;
    }
    let p = &ev.p;
    let el = c.fi - c.start_frame(ev);
    let period = 2 + (p[4] * 7.0) as i64;
    let lag = if p[7] > 0.35 { el.rem_euclid(period) } else { el };   // re-latch every period, or hold
    let lag = depth(lag);
    if lag < 1 {
        return None;
    }
    let (x0, y0, _, _) = c.bbox;
    let (bw, bh) = box_wh(c);
    let region = if p[5] < 0.1 {
        band(c, ev, 0.15, 1.1, 0.0, 0.0, w, h)                     // the whole subject freezes
    } else if p[5] < 0.4 {
        band(c, ev, 0.3, 0.0, 0.04, 0.3, w, h)                     // a band across it
    } else {
        let (rw, rh) = (bw * (0.12 + 0.6 * p[1]), bh * (0.03 + 0.25 * p[3]));   // a block
        let xa = (x0 as f64 + p[0] * (bw - rw).max(1.0)).clamp(0.0, w as f64);
        let ya = (y0 as f64 + p[6] * (bh - rh).max(1.0)).clamp(0.0, h as f64);
        let xb = (xa.floor() + rw).clamp(0.0, w as f64);
        let yb = (ya.floor() + rh).clamp(0.0, h as f64);
        (ya as usize, yb as usize, xa as usize, xb as usize)
    };
    if region.1 <= region.0 || region.3 <= region.2 {
        return None;
    }
    let amp = 0.4 + 0.6 * c.p.intensity;
    let dx = if p[6] > 0.75 { iround((c.held(ev, 1, period) - 0.5) * 0.25 * bw * amp) } else { 0 };
    Some(StutterPlan { lag, region, dx })
}

/// Frame hold: a block, a band or the whole box freezes on an older frame, optionally jumping
/// forward every few frames (dropped packets), sometimes displaced sideways.
pub fn stutter(c: &Ctx, hist: &Hist, out: &View, wrote: &mut Rect) -> usize {
    let mut n = 0;
    for ev in c.events(STUTTER) {
        let Some(pl) = stutter_plan(c, &ev, &|k| hist.depth(k), out.w, out.h) else { continue };
        let old = hist.get(pl.lag).unwrap();
        let (ya, yb, xa, xb) = pl.region;
        *wrote = wrote.union(Rect::new(ya, yb, xa, xb));
        for_region(out, ya, yb, xa, xb, |x, y, _| old.px(x - pl.dx, y));
        n += 1;
    }
    n
}

// ----------------------------------------------------------------------------------- mosh
struct MoshPlan {
    lag: usize,
    region: (usize, usize, usize, usize),
    gain: f64,
}

/// Low-res scale of the motion search.
pub fn mosh_scale(w: usize, h: usize) -> usize {
    ((w.max(h) as f64 / 384.0 + 0.5).floor() as usize).max(1)
}

fn mosh_plan(c: &Ctx, ev: &Ev, depth: &dyn Fn(i64) -> usize, w: usize, h: usize) -> Option<MoshPlan> {
    if ev.g < 0.3 {
        return None;
    }
    let p = &ev.p;
    let lag = depth((c.fi - c.start_frame(ev)).min(16));
    if lag < 1 {
        return None;
    }
    let region = band(c, ev, 0.35, 0.25, 0.15, 0.7, w, h);
    if region.1 - region.0 < 2 || region.3 - region.2 < 2 {
        return None;
    }
    let gain = if p[6] > 0.6 { 1.0 + 1.5 * c.p.intensity * (p[6] - 0.6) / 0.4 } else { 1.0 };   // 'bloom'
    Some(MoshPlan { lag, region, gain })
}

/// Datamosh: inside a band the picture stops refreshing - the frame from when the event began
/// is dragged along by every later frame's block motion (P-frames without their I-frame), so
/// colour and silhouette smear with the movement. It sits over the live frame and ends with a
/// hard cut, like a new keyframe.
pub fn mosh(c: &Ctx, hist: &Hist, out: &View, cache: &Mutex<Caches>, wrote: &mut Rect) -> usize {
    let (w, h) = (out.w, out.h);
    let s = mosh_scale(w, h);
    let mut n = 0;
    for ev in c.events(MOSH) {
        let Some(pl) = mosh_plan(c, &ev, &|k| hist.depth(k), w, h) else { continue };
        // every frame from the held one to now is needed (motion between each pair)
        if (0..=pl.lag).any(|k| hist.get(k).is_none()) {
            continue;
        }
        let (ya, yb, xa, xb) = pl.region;
        let m = ((pl.lag.min(8) as i64 * MR * s as i64) as f64 * pl.gain) as usize + 2;
        let (cy0, cy1, cx0, cx1) = (ya.saturating_sub(m) as i64, (yb + m).min(h) as i64,
                                    xa.saturating_sub(m) as i64, (xb + m).min(w) as i64);
        // motion of each step, newest first: field k = frame k+1 -> frame k
        let fields: Vec<_> = (0..pl.lag)
            .map(|k| motion_cached(cache, &hist.get(k + 1).unwrap(), &hist.get(k).unwrap(), s))
            .collect();
        let held = hist.get(pl.lag).unwrap();
        let gain = pl.gain;
        *wrote = wrote.union(Rect::new(ya, yb, xa, xb));
        // glitch.py replays the held frame through every step: base[x] = base[x + d_k(x)] inside
        // the replay area, oldest step first - lag full passes over a large area. The same
        // result per pixel: follow the source position back through the steps, newest first,
        // stopping where it leaves the area (outside it the held frame was never updated), then
        // read the held frame once. No per-step copies; only the band's pixels are computed.
        for_region(out, ya, yb, xa, xb, |x, y, o| {
            let (mut sy, mut sx) = (y, x);
            for mo in fields.iter() {
                if sy < cy0 || sy >= cy1 || sx < cx0 || sx >= cx1 {
                    break;
                }
                if mo.by == 0 || mo.bx == 0 {
                    continue;
                }
                let iy = (sy as usize / mo.bs).min(mo.by - 1);
                let ix = (sx as usize / mo.bs).min(mo.bx - 1);
                let (vy, vx) = (mo.vy[iy * mo.bx + ix], mo.vx[iy * mo.bx + ix]);
                sy = (sy + iround(vy as f64 * gain)).clamp(0, h as i64 - 1);
                sx = (sx + iround(vx as f64 * gain)).clamp(0, w as i64 - 1);
            }
            // over the live frame: smeared colour on top, gaps (motion-search misses on the
            // silhouette) show the current body instead of see-through holes
            over(held.px(sx, sy), o)
        });
        n += 1;
    }
    n
}

/// Which earlier frames this frame's time effects will read, if `max_back` frames exist
/// before it (k = 1..=max_back): the host fetches exactly these and nothing else.
pub fn needed(c: &Ctx, w: usize, h: usize, max_back: usize) -> Vec<usize> {
    let max_back = max_back.min(MAX_BACK);
    let avail = |k: usize| k >= 1 && k <= max_back;
    let depth = |want: i64| want.clamp(0, max_back as i64) as usize;
    let mut need = [false; MAX_BACK + 1];
    if c.gain(ECHO) > 0.0 {
        for ev in c.events(ECHO) {
            if let Some(pl) = echo_plan(c, &ev, &avail, w, h) {
                pl.lags.iter().for_each(|&k| need[k] = true);
            }
        }
    }
    if c.gain(STUTTER) > 0.0 {
        for ev in c.events(STUTTER) {
            if let Some(pl) = stutter_plan(c, &ev, &depth, w, h) {
                need[pl.lag] = true;
            }
        }
    }
    if c.gain(MOSH) > 0.0 {
        for ev in c.events(MOSH) {
            if let Some(pl) = mosh_plan(c, &ev, &depth, w, h) {
                (1..=pl.lag).for_each(|k| need[k] = true);
            }
        }
    }
    (1..=MAX_BACK).filter(|&k| need[k]).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::image::Frame;
    use crate::sched::Params;

    /// glitch.py's replay, literally: copy the held frame, apply every step to the whole area.
    fn mosh_reference(c: &Ctx, hist: &Hist, out: &View, cache: &Mutex<Caches>) {
        let (w, h) = (out.w, out.h);
        let s = mosh_scale(w, h);
        for ev in c.events(MOSH) {
            let Some(pl) = mosh_plan(c, &ev, &|k| hist.depth(k), w, h) else { continue };
            if (0..=pl.lag).any(|k| hist.get(k).is_none()) {
                continue;
            }
            let (ya, yb, xa, xb) = pl.region;
            let m = ((pl.lag.min(8) as i64 * MR * s as i64) as f64 * pl.gain) as usize + 2;
            let (cy0, cy1, cx0, cx1) = (ya.saturating_sub(m), (yb + m).min(h), xa.saturating_sub(m), (xb + m).min(w));
            let mut base = Frame::from_view(&hist.get(pl.lag).unwrap());
            for k in (0..pl.lag).rev() {
                let mo = motion_cached(cache, &hist.get(k + 1).unwrap(), &hist.get(k).unwrap(), s);
                let old = base.clone();
                for y in cy0..cy1 {
                    for x in cx0..cx1 {
                        let iy = (y / mo.bs).min(mo.by - 1);
                        let ix = (x / mo.bs).min(mo.bx - 1);
                        let sy = (y as i64 + iround(mo.vy[iy * mo.bx + ix] as f64 * pl.gain)).clamp(0, h as i64 - 1) as usize;
                        let sx = (x as i64 + iround(mo.vx[iy * mo.bx + ix] as f64 * pl.gain)).clamp(0, w as i64 - 1) as usize;
                        let (d, s_) = (4 * (y * w + x), 4 * (sy * w + sx));
                        base.data[d..d + 4].copy_from_slice(&old.data[s_..s_ + 4]);
                    }
                }
            }
            let bv = base.view();
            for_region(out, ya, yb, xa, xb, |x, y, o| over(bv.px(x, y), o));
        }
    }

    #[test]
    fn mosh_source_chaining_equals_the_literal_replay() {
        let (w, h) = (256usize, 160usize);
        let mut frames: Vec<Frame> = (0..30)
            .map(|i| {
                let mut f = Frame::new(w, h);
                for y in 20..140 {
                    for x in 0..w {
                        let cx = 70.0 + 4.0 * i as f32 + 6.0 * ((y as f32) * 0.05).sin();   // test data
                        let a = (1.0 - ((x as f32 - cx) / 40.0).powi(2)).clamp(0.0, 1.0);
                        let k = 4 * (y * w + x);
                        let v = ((x * 7 + y * 3) % 23) as f32 / 23.0;
                        f.data[k..k + 4].copy_from_slice(&[v * a, 0.4 * a, (1.0 - v) * a, a]);
                    }
                }
                f
            })
            .collect();
        let views: Vec<View> = frames.iter_mut().map(|f| f.view()).collect();
        let mut tested = 0;
        for seed in 1..40u32 {
            let mut p = Params { seed, intensity: 1.0, bursts: 0.0, ..Default::default() };
            p.gains = [0.0; 13];
            p.gains[MOSH] = 3.0;
            for i in 17..30usize {
                let c = Ctx::new(p.clone(), i as f64 / 25.0, i as i64, (30, 20, 226, 140));
                let back: Vec<Option<View>> = (1..=MAX_BACK).map(|k| if k <= i { Some(views[i - k]) } else { None }).collect();
                let hist = Hist { cur: views[i], back: &back };
                let cache = Mutex::new(Caches::default());
                let (mut a, mut b) = (Frame::from_view(&views[i]), Frame::from_view(&views[i]));
                let na = mosh(&c, &hist, &a.view(), &cache, &mut Rect::new(0, 0, 0, 0));
                mosh_reference(&c, &hist, &b.view(), &cache);
                assert_eq!(a.data, b.data, "seed {seed} frame {i}");
                tested += na;
            }
        }
        assert!(tested > 10, "only {tested} mosh events exercised");
    }
}
