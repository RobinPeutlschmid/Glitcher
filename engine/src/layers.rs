//! 3D mode's passes (glitch.py's layered branch of `glitch()`): each plane effect runs twice,
//! once for the planes in front of the body and once for those behind it; what the behind pass
//! changed is drawn *under* the front result, so it only shows where the body doesn't cover it.
//! The planes are laid out in an upright working frame (the subject's roll undone) that is
//! rotated back afterwards - only the pixels the glitch changed, so the rest stays exact.
//!
//! Both passes start from the same picture and only touch the areas they report (`Area.dirty`),
//! so instead of copying whole frames the behind pass's result is kept in tiles of those areas.

use crate::fx_frame::{Area, Before, run_one};
use std::sync::Arc;
use crate::image::{Frame, Rect, View, over};
use crate::sched::{Ctx, Layer};
use rayon::prelude::*;

const TS: usize = 64;      // = fx_frame's Before tiles: a dirty tile is always a saved one

/// Marked 64x64 tiles of a canvas (optionally with their pixels saved).
pub struct Tiles {
    w: usize,
    h: usize,
    tw: usize,
    th: usize,
    mark: Vec<bool>,
    saved: Vec<Option<Vec<f32>>>,
}

impl Tiles {
    pub fn new(w: usize, h: usize) -> Tiles {
        let (tw, th) = (w.div_ceil(TS), h.div_ceil(TS));
        Tiles { w, h, tw, th, mark: vec![false; tw * th], saved: Vec::new() }
    }

    pub fn mark(&mut self, rects: &[Rect]) {
        for r in rects {
            let r = Rect::new(r.y0.min(self.h), r.y1.min(self.h), r.x0.min(self.w), r.x1.min(self.w));
            if r.is_empty() {
                continue;
            }
            for ty in r.y0 / TS..r.y1.div_ceil(TS) {
                for tx in r.x0 / TS..r.x1.div_ceil(TS) {
                    self.mark[ty * self.tw + tx] = true;
                }
            }
        }
    }

    pub fn any(&self) -> bool {
        self.mark.iter().any(|&m| m)
    }

    fn tile(&self, i: usize) -> Rect {
        let (ty, tx) = (i / self.tw, i % self.tw);
        Rect::new(ty * TS, ((ty + 1) * TS).min(self.h), tx * TS, ((tx + 1) * TS).min(self.w))
    }

    /// Bounding box of the marked tiles.
    pub fn bounds(&self) -> Rect {
        (0..self.mark.len()).filter(|&i| self.mark[i]).fold(Rect::new(0, 0, 0, 0), |a, i| a.union(self.tile(i)))
    }

    /// Copy the marked tiles' pixels out of `v`.
    pub fn save(&mut self, v: &View) {
        let idx: Vec<usize> = (0..self.mark.len()).filter(|&i| self.mark[i]).collect();
        let data: Vec<(usize, Vec<f32>)> = idx
            .par_iter()
            .map(|&i| {
                let r = self.tile(i);
                let mut d = Vec::with_capacity(4 * (r.x1 - r.x0) * (r.y1 - r.y0));
                for y in r.y0..r.y1 {
                    d.extend_from_slice(&v.row(y)[4 * r.x0..4 * r.x1]);
                }
                (i, d)
            })
            .collect();
        self.saved = vec![None; self.mark.len()];
        for (i, d) in data {
            self.saved[i] = Some(d);
        }
    }

    fn saved_px(&self, i: usize, r: &Rect, x: usize, y: usize) -> [f32; 4] {
        let d = self.saved[i].as_ref().unwrap();
        let k = 4 * ((y - r.y0) * (r.x1 - r.x0) + (x - r.x0));
        [d[k], d[k + 1], d[k + 2], d[k + 3]]
    }

    /// For every pixel of the marked tiles: `f(x, y, tile index, current pixel)` -> new pixel.
    /// Tile rows in parallel (each owns its rows).
    fn each(&self, out: &View, f: impl Fn(usize, usize, usize, &Rect, [f32; 4]) -> [f32; 4] + Sync) {
        (0..self.th).into_par_iter().for_each(|ty| {
            let tiles: Vec<usize> = (0..self.tw).map(|tx| ty * self.tw + tx).filter(|&i| self.mark[i]).collect();
            if tiles.is_empty() {
                return;
            }
            for y in ty * TS..((ty + 1) * TS).min(self.h) {
                // SAFETY: this tile row is the only writer of these rows
                let row = unsafe { out.row_mut(y) };
                for &i in &tiles {
                    let r = self.tile(i);
                    for x in r.x0..r.x1 {
                        let o = [row[4 * x], row[4 * x + 1], row[4 * x + 2], row[4 * x + 3]];
                        let n = f(x, y, i, &r, o);
                        row[4 * x..4 * x + 4].copy_from_slice(&n);
                    }
                }
            }
        });
    }
}

/// Mix of two premultiplied pixels: `a` where m = 0, `b` where m = 1 (both exact).
#[inline]
pub fn mix(a: [f32; 4], b: [f32; 4], m: f32) -> [f32; 4] {
    if m >= 0.999 {
        b
    } else if m <= 0.001 {
        a
    } else {
        let k = 1.0 - m;
        [a[0] * k + b[0] * m, a[1] * k + b[1] * m, a[2] * k + b[2] * m, a[3] * k + b[3] * m]
    }
}

/// A soft 0..1 mask over the canvas.
pub trait Mask: Sync {
    fn at(&self, x: usize, y: usize) -> f32;
}

/// One frame effect, limited by `mask` if given: where the mask is 0 the picture stays as it
/// was before the effect.
pub fn run_masked(fx: usize, c: &Ctx, out: &View, area: &mut Area, mask: Option<&dyn Mask>) -> usize {
    let Some(m) = mask else { return run_one(fx, c, out, area) };
    let before = Arc::new(Before::new(out));
    let mut a = area.inner(before.clone());
    let n = run_one(fx, c, out, &mut a);
    let mut t = Tiles::new(out.w, out.h);
    t.mark(&a.dirty);
    t.each(out, |x, y, _, _, o| mix(before.px(x as i64, y as i64), o, m.at(x, y)));
    area.cbox = a.cbox;
    area.dirty.extend(a.dirty);
    n
}

/// A plane effect in 3D: the front and the behind pass, the behind one's changes under the
/// front result (glitch.py: `_under(front, _ghost(back, work))`). Returns events applied.
pub fn run_layered(fx: usize, c: &Ctx, out: &View, area: &mut Area, mask: Option<&dyn Mask>) -> usize {
    let (front, back) = (c.layered(Layer::Front), c.layered(Layer::Back));
    if back.events(fx).is_empty() {
        return run_masked(fx, &front, out, area, mask);
    }
    // one copy-on-write store of the picture before both passes: what either pass writes is
    // saved there first
    let before = Arc::new(Before::new(out));
    // the behind pass, kept in tiles, then undone
    let mut ab = area.inner(before.clone());
    let nb = run_one(fx, &back, out, &mut ab);
    let mut t = Tiles::new(out.w, out.h);
    t.mark(&ab.dirty);
    t.save(out);
    t.each(out, |x, y, _, _, _| before.px(x as i64, y as i64));
    // the front pass from the same picture
    let mut af = area.inner(before.clone());
    let nf = if front.events(fx).is_empty() { 0 } else { run_one(fx, &front, out, &mut af) };
    // behind-the-body changes that are visible go under the front result
    t.each(out, |x, y, i, r, o| {
        let b = t.saved_px(i, r, x, y);
        let w = before.px(x as i64, y as i64);
        if b[3] > 0.0 && b != w { over(o, b) } else { o }
    });
    let mut dirty = ab.dirty;
    dirty.extend(af.dirty);
    if let Some(m) = mask {
        let mut tm = Tiles::new(out.w, out.h);
        tm.mark(&dirty);
        tm.each(out, |x, y, _, _, o| mix(before.px(x as i64, y as i64), o, m.at(x, y)));
    }
    area.cbox = area.cbox.union(ab.cbox).union(af.cbox);
    area.dirty.extend(dirty);
    nb + nf
}

// --------------------------------------------------------------------------------- roll
/// The rotation between the screen and the upright working frame: the work frame is the
/// picture turned counter-clockwise by `deg` about (cx, cy) (PIL's rotate), y down.
#[derive(Clone, Copy, Debug)]
pub struct Roll {
    pub deg: f64,
    pub cx: f64,
    pub cy: f64,
    c: f64,
    s: f64,
}

impl Roll {
    pub fn new(deg: f64, cx: f64, cy: f64) -> Roll {
        let r = deg.to_radians();
        Roll { deg, cx, cy, c: crate::detmath::cos(r), s: crate::detmath::sin(r) }
    }
    /// Screen point -> where it is in the work frame.
    pub fn to_work(&self, x: f64, y: f64) -> (f64, f64) {
        let (dx, dy) = (x - self.cx, y - self.cy);
        (self.cx + self.c * dx + self.s * dy, self.cy - self.s * dx + self.c * dy)
    }
    /// Work-frame point -> screen.
    pub fn to_screen(&self, x: f64, y: f64) -> (f64, f64) {
        let (dx, dy) = (x - self.cx, y - self.cy);
        (self.cx + self.c * dx - self.s * dy, self.cy + self.s * dx + self.c * dy)
    }
}

#[inline]
fn cubic(t: f32) -> [f32; 4] {
    // Catmull-Rom (a = -0.5, PIL's bicubic) weights for taps -1, 0, 1, 2
    let a = -0.5f32;
    let w = |x: f32| {
        let x = x.abs();
        if x <= 1.0 {
            (a + 2.0) * x * x * x - (a + 3.0) * x * x + 1.0
        } else if x < 2.0 {
            a * x * x * x - 5.0 * a * x * x + 8.0 * a * x - 4.0 * a
        } else {
            0.0
        }
    };
    [w(1.0 + t), w(t), w(1.0 - t), w(2.0 - t)]
}

/// Premultiplied bicubic sample at continuous (x, y) (pixel centres at i + 0.5), transparent
/// outside. Overshoot is clipped: alpha to 0..1, colour to 0..alpha x `smax` (the brightest
/// straight colour of the source), so edges get no halos and tiny alphas don't blow up.
#[inline]
fn bicubic(v: &View, x: f64, y: f64, smax: f32) -> [f32; 4] {
    let (u, w) = (x - 0.5, y - 0.5);
    let (fx, fy) = (u.floor(), w.floor());
    let (wx, wy) = (cubic((u - fx) as f32), cubic((w - fy) as f32));
    let (ix, iy) = (fx as i64, fy as i64);
    let mut acc = [0f32; 4];
    if ix >= 1 && iy >= 1 && ix + 2 < v.w as i64 && iy + 2 < v.h as i64 {
        // all 16 taps inside: straight from the rows
        let x0 = 4 * (ix as usize - 1);
        for (j, wyj) in wy.iter().enumerate() {
            let row = &v.row(iy as usize - 1 + j)[x0..x0 + 16];
            for (i, wxi) in wx.iter().enumerate() {
                let k = wyj * wxi;
                for c in 0..4 {
                    acc[c] += row[4 * i + c] * k;
                }
            }
        }
    } else {
        for (j, wyj) in wy.iter().enumerate() {
            let yy = iy - 1 + j as i64;
            if yy < 0 || yy >= v.h as i64 {
                continue;
            }
            for (i, wxi) in wx.iter().enumerate() {
                let p = v.px(ix - 1 + i as i64, yy);
                let k = wyj * wxi;
                for c in 0..4 {
                    acc[c] += p[c] * k;
                }
            }
        }
    }
    let a = acc[3].clamp(0.0, 1.0);
    let cap = a * smax;
    [acc[0].clamp(0.0, cap), acc[1].clamp(0.0, cap), acc[2].clamp(0.0, cap), a]
}

/// Brightest straight colour (at least 1) inside r.
fn straight_max(v: &View, r: Rect) -> f32 {
    (r.y0..r.y1)
        .into_par_iter()
        .map(|y| {
            let row = v.row(y);
            let mut m = 1f32;
            for x in r.x0..r.x1 {
                let a = row[4 * x + 3];
                if a > 0.0 {
                    m = m.max(row[4 * x].max(row[4 * x + 1]).max(row[4 * x + 2]) / a);
                }
            }
            m
        })
        .reduce(|| 1.0, f32::max)
}

/// Screen-space bounding box of a work-frame rect, grown by 2 px, clipped to the canvas.
fn mapped_box(r: Rect, f: impl Fn(f64, f64) -> (f64, f64), w: usize, h: usize) -> Rect {
    let pts = [(r.x0, r.y0), (r.x1, r.y0), (r.x0, r.y1), (r.x1, r.y1)].map(|(x, y)| f(x as f64, y as f64));
    let (mut x0, mut y0, mut x1, mut y1) = (f64::MAX, f64::MAX, f64::MIN, f64::MIN);
    for (x, y) in pts {
        (x0, y0, x1, y1) = (x0.min(x), y0.min(y), x1.max(x), y1.max(y));
    }
    let cl = |v: f64, m: usize| v.clamp(0.0, m as f64) as usize;
    Rect::new(cl(y0.floor() - 2.0, h), cl(y1.ceil() + 2.0, h), cl(x0.floor() - 2.0, w), cl(x1.ceil() + 2.0, w))
}

/// The upright working frame of a rolled shot (see rotate_in).
pub struct Upright {
    pub frame: Frame,
    /// Its content area (the disc the content can reach, clipped to the canvas).
    pub area: Rect,
}

/// The columns of work-frame row y whose source point (on screen) lies in `src` (grown by the
/// bicubic's reach): elsewhere the sample is 0.
fn row_range(roll: &Roll, y: usize, src: Rect, lim: Rect) -> (usize, usize) {
    let (gx0, gx1) = (src.x0 as f64 - 3.0, src.x1 as f64 + 3.0);
    let (gy0, gy1) = (src.y0 as f64 - 3.0, src.y1 as f64 + 3.0);
    let (sx0, sy0) = roll.to_screen(0.5, y as f64 + 0.5);      // source of column 0
    let (kx, ky) = (roll.c, roll.s);                            // per column step
    let mut lo = lim.x0 as f64;
    let mut hi = lim.x1 as f64;
    for (s0, k, a, b) in [(sx0, kx, gx0, gx1), (sy0, ky, gy0, gy1)] {
        if k.abs() < 1e-12 {
            if s0 < a || s0 > b {
                return (0, 0);
            }
        } else {
            let (p, q) = ((a - s0) / k, (b - s0) / k);
            lo = lo.max(p.min(q).floor() - 1.0);
            hi = hi.min(p.max(q).ceil() + 2.0);
        }
    }
    if hi <= lo {
        return (0, 0);
    }
    (lo.max(lim.x0 as f64) as usize, hi.min(lim.x1 as f64) as usize)
}

/// The upright working frame: `src`'s content (inside `cbox`) turned by the roll.
pub fn rotate_in(src: &View, cbox: Rect, roll: &Roll) -> Upright {
    let (w, h) = (src.w, src.h);
    let mut f = Frame::new(w, h);                 // zeroed lazily by the OS: only the disc is touched
    if cbox.is_empty() {
        return Upright { frame: f, area: cbox };
    }
    // content within radius r of the centre stays within it
    let r = [(cbox.x0, cbox.y0), (cbox.x1, cbox.y0), (cbox.x0, cbox.y1), (cbox.x1, cbox.y1)]
        .iter()
        .map(|&(x, y)| ((x as f64 - roll.cx).powi(2) + (y as f64 - roll.cy).powi(2)).sqrt())
        .fold(0.0, f64::max)
        .ceil()
        + 3.0;
    let cl = |v: f64, m: usize| v.clamp(0.0, m as f64) as usize;
    let disc = Rect::new(cl((roll.cy - r).floor(), h), cl((roll.cy + r).ceil(), h),
                         cl((roll.cx - r).floor(), w), cl((roll.cx + r).ceil(), w));
    let smax = straight_max(src, cbox);
    let v = f.view();
    (disc.y0..disc.y1).into_par_iter().for_each(|y| {
        let (x0, x1) = row_range(roll, y, cbox, disc);
        if x1 <= x0 {
            return;
        }
        // SAFETY: one thread per row
        let row = unsafe { v.span_mut(y, x0, x1) };
        for x in x0..x1 {
            let (sx, sy) = roll.to_screen(x as f64 + 0.5, y as f64 + 0.5);
            let p = bicubic(src, sx, sy, smax);
            row[4 * (x - x0)..4 * (x - x0) + 4].copy_from_slice(&p);
        }
    });
    Upright { frame: f, area: disc }
}

/// Rotate back what the glitch changed: the working frame (after the effects) against what
/// rotate_in made of `out` (recomputed: `out` is untouched until now), inside the marked
/// `tiles`. Changed pixels, turned back, replace those of `out`; everything else in `out`
/// stays as it was. Returns the screen area written.
pub fn rotate_back(up: &mut Upright, base: &Before, tiles: &Tiles, roll: &Roll, out: &View) -> Rect {
    let b = tiles.bounds();
    if b.is_empty() {
        return b;
    }
    let work = up.frame.view();
    let bw = b.x1 - b.x0;
    // which work-frame pixels changed (as glitch.py: any channel differs); `base` kept every
    // touched tile as rotate_in made it, untouched ones are unchanged by definition
    let changed: Vec<bool> = (b.y0..b.y1)
        .into_par_iter()
        .flat_map_iter(|y| {
            let row = work.row(y);
            (b.x0..b.x1).map(move |x| {
                let p = [row[4 * x], row[4 * x + 1], row[4 * x + 2], row[4 * x + 3]];
                p != base.px(x as i64, y as i64)
            })
        })
        .collect();
    if !changed.iter().any(|&c| c) {
        return Rect::new(0, 0, 0, 0);
    }
    let ch = |x: i64, y: i64| -> bool {
        x >= b.x0 as i64 && y >= b.y0 as i64 && x < b.x1 as i64 && y < b.y1 as i64
            && changed[(y as usize - b.y0) * bw + (x as usize - b.x0)]
    };
    let smax = straight_max(&work, b);
    let dst = mapped_box(b, |x, y| roll.to_screen(x, y), out.w, out.h);
    (dst.y0..dst.y1).into_par_iter().for_each(|y| {
        // SAFETY: one thread per row
        let row = unsafe { out.row_mut(y) };
        for x in dst.x0..dst.x1 {
            let (wx, wy) = roll.to_work(x as f64 + 0.5, y as f64 + 0.5);
            // the changed mask, turned back bilinearly: any tap with weight counts (PIL: > 0)
            let (u, v) = (wx - 0.5, wy - 0.5);
            let (ix, iy) = (u.floor() as i64, v.floor() as i64);
            let (fx, fy) = (u - u.floor(), v - v.floor());
            let hit = ch(ix, iy)
                || (ch(ix + 1, iy) && fx > 0.0)
                || (ch(ix, iy + 1) && fy > 0.0)
                || (ch(ix + 1, iy + 1) && fx > 0.0 && fy > 0.0);
            if hit {
                let p = bicubic(&work, wx, wy, smax);
                row[4 * x..4 * x + 4].copy_from_slice(&p);
            }
        }
    });
    dst
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roll_maps_round_trip_and_turn_counter_clockwise() {
        let r = Roll::new(30.0, 100.0, 50.0);
        let (x, y) = r.to_work(140.0, 70.0);
        let (a, b) = r.to_screen(x, y);
        assert!((a - 140.0).abs() < 1e-9 && (b - 70.0).abs() < 1e-9);
        // PIL rotate(+deg) is counter-clockwise on screen: a point right of the centre moves up
        let (_, y) = r.to_work(110.0, 50.0);
        assert!(y < 50.0);
    }

    #[test]
    fn cubic_weights_sum_to_one() {
        for t in [0.0, 0.25, 0.5, 0.9] {
            let w = cubic(t);
            assert!((w.iter().sum::<f32>() - 1.0).abs() < 1e-5);
        }
        assert_eq!(cubic(0.0), [0.0, 1.0, 0.0, 0.0]);
    }

    #[test]
    fn tiles_mark_save_and_bound() {
        let mut f = Frame::new(100, 70);
        f.data[4 * (40 * 100 + 50)] = 0.5;
        let v = f.view();
        let mut t = Tiles::new(100, 70);
        t.mark(&[Rect::new(35, 45, 70, 75)]);
        assert_eq!(t.bounds(), Rect::new(0, 64, 64, 100));
        t.mark(&[Rect::new(35, 45, 45, 55)]);
        assert_eq!(t.bounds(), Rect::new(0, 64, 0, 100));
        t.save(&v);
        let i = 0;
        let r = t.tile(i);
        assert_eq!(t.saved_px(i, &r, 50, 40)[0], 0.5);
    }
}
