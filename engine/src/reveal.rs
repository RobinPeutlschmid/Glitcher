//! Appear / vanish (glitch.py's reveals): the subject builds up out of glitch structure, or
//! breaks apart and goes. A transition has an arrival-time map laid over the layout box (so
//! the build-up follows the subject and is coherent from frame to frame): pixels arriving now
//! are "in transit" - shoved sideways in slices that land as they settle, shown as flat mosaic
//! blocks and lit up - the rest is either there or not. Afterwards the glitch shows only where
//! the subject exists, a little ahead of the front so the build-up sparkles. The glitch peaks
//! mid-transition (as a hit).
//!
//! Same structure and constants as glitch.py, drawn from the engine's hash randomness, so a
//! transition looks like the app's in kind, not pixel for pixel.

use crate::detmath::{Stream, round_even, sin};
use crate::image::{Frame, Rect, View};
use rayon::prelude::*;
use std::f64::consts::PI;

const D: f64 = 0.22;                       // how long a pixel stays in transit (share of the ramp)
const U: (f64, f64) = (-0.25, 1.25);       // map coverage in layout-box widths / heights
const V: (f64, f64) = (-0.15, 1.15);
const FH: usize = 144;
const FW: usize = 96;
const GLOW: [[f32; 3]; 3] = [[90.0 / 255.0, 1.0, 235.0 / 255.0], [1.0, 1.0, 1.0], [1.0, 80.0 / 255.0, 230.0 / 255.0]];

/// A transition: (frame, appear?, seconds).
pub type Trans = (f64, bool, f64);

/// "24, 200:1.5" -> transitions of one kind; `len` = default seconds.
pub fn parse(text: &str, appear: bool, len: f64) -> Result<Vec<Trans>, String> {
    let what = if appear { "appear" } else { "vanish" };
    let mut out = Vec::new();
    for part in text.split([',', ';', ' ', '\t', '\n']).map(str::trim).filter(|s| !s.is_empty()) {
        let b: Vec<&str> = part.split(':').collect();
        let f = b[0].trim().parse::<f64>().ok().filter(|f| f.is_finite());
        let sec = match b.get(1).map(|s| s.trim()).filter(|s| !s.is_empty()) {
            None => Some(len),
            Some(s) => s.parse::<f64>().ok(),
        };
        match (f, sec) {
            (Some(f), Some(sec)) if sec > 0.0 && b.len() <= 2 => out.push((f, appear, sec)),
            _ => return Err(format!("{what} '{part}': expected FRAME[:SECONDS], seconds > 0")),
        }
    }
    Ok(out)
}

/// Where the subject is at `frame` (glitch.py's Timing.reveal).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Reveal {
    Visible,
    Hidden,
    /// Mid-transition: appear?, progress 0..1, the transition's index (its pattern).
    Moving(bool, f64, usize),
}

pub fn at(trans: &[Trans], frame: f64, fps: f64) -> Reveal {
    let mut t = trans.to_vec();
    t.sort_by(|a, b| a.0.total_cmp(&b.0));
    let Some(first) = t.first() else { return Reveal::Visible };
    let mut visible = !first.1;                         // the first key says what came before
    for (i, &(tf, appear, sec)) in t.iter().enumerate() {
        if tf > frame {
            break;
        }
        let n = (sec * fps).max(1.0);                   // frames; the last one lands at q = 1
        let later = t.get(i + 1).is_some_and(|nx| nx.0 <= frame);
        if frame - tf < n && !later {
            return Reveal::Moving(appear, ((frame - tf + 1.0) / n).min(1.0), i);
        }
        visible = appear;
    }
    if visible { Reveal::Visible } else { Reveal::Hidden }
}

/// The glitch's peak mid-transition, as a hit level.
pub fn boost(r: Reveal) -> f64 {
    match r {
        Reveal::Moving(_, q, _) => 0.9 * sin(PI * q),
        _ => 0.0,
    }
}

/// A transition's map: arrival time 0..1 per cell (uniformly spread), slice offset per cell
/// row (box widths), glow colour.
pub struct Map {
    rank: Vec<f32>,
    dx: Vec<f32>,
    pub glow: [f32; 3],
}

impl Map {
    pub fn new(seed: u32, idx: usize) -> Map {
        let mut rng = Stream::new(&[seed as u64, 7919, idx as u64]);
        let v = |i: usize| V.0 + (V.1 - V.0) * i as f64 / (FH - 1) as f64;
        let u = |j: usize| U.0 + (U.1 - U.0) * j as f64 / (FW - 1) as f64;
        let c = rng.next_f64();
        let style = if c < 0.35 { 0 } else if c < 0.55 { 1 } else if c < 0.85 { 2 } else { 3 };
        let flip = rng.next_f64() < 0.5;
        let sweep = |i: usize, j: usize| match style {
            0 => v(i),                                                  // from the top
            1 => 1.0 - v(i),                                            // from the bottom
            2 => (((u(j) - 0.5) * 0.7).powi(2) + (v(i) - 0.45).powi(2)).sqrt(),   // from the middle
            _ => if flip { u(j) } else { 1.0 - u(j) },                  // from a side
        };
        // random blocks of b x b cells
        fn blocks(rng: &mut Stream, b: usize) -> Vec<f64> {
            let (gh, gw) = (FH.div_ceil(b), FW.div_ceil(b));
            let g: Vec<f64> = (0..gh * gw).map(|_| rng.next_f64()).collect();
            (0..FH * FW).map(|k| g[(k / FW / b) * gw + (k % FW) / b]).collect()
        }
        let b = [4, 6, 8, 12][(rng.next_f64() * 4.0) as usize % 4];
        let b1 = blocks(&mut rng, b);
        let b2 = blocks(&mut rng, (b / 2).max(2));
        let rows: Vec<f64> = (0..FH.div_ceil(2)).map(|_| rng.next_f64()).collect();
        let (ws, wb) = (0.35 + 0.35 * rng.next_f64(), 0.25 + 0.30 * rng.next_f64());
        let sw: Vec<f64> = (0..FH * FW).map(|k| sweep(k / FW, k % FW)).collect();
        let (lo, hi) = sw.iter().fold((f64::MAX, f64::MIN), |(a, b), &x| (a.min(x), b.max(x)));
        let score: Vec<f64> = (0..FH * FW)
            .map(|k| ws * (sw[k] - lo) / (hi - lo).max(1e-6) + wb * (0.65 * b1[k] + 0.35 * b2[k])
                 + 0.15 * rows[(k / FW) / 2] + 0.05 * rng.next_f64())
            .collect();
        let mut order: Vec<usize> = (0..FH * FW).collect();
        order.sort_by(|&a, &b| score[a].total_cmp(&score[b]));      // stable
        let mut rank = vec![0f32; FH * FW];
        for (pos, &k) in order.iter().enumerate() {
            rank[k] = pos as f32 / (FH * FW - 1) as f32;
        }
        let nb = FH.div_ceil(3);
        let amp = 0.08 + 0.20 * rng.next_f64();
        let bands: Vec<f32> = (0..nb)
            .map(|_| {
                let (a, on) = (2.0 * rng.next_f64() - 1.0, rng.next_f64() < 0.6);
                if on { (a * amp) as f32 } else { 0.0 }
            })
            .collect();
        let dx = (0..FH).map(|i| bands[i / 3]).collect();
        let glow = GLOW[(rng.next_f64() * 3.0) as usize % 3];
        Map { rank, dx, glow }
    }

    /// Arrival time at (x, y) for a layout box (1 = outside the map), and the row's slice
    /// offset in px.
    #[inline]
    fn at(&self, x: usize, y: usize, bbox: (i64, i64, i64, i64)) -> (f32, f32) {
        let (x0, y0, x1, y1) = bbox;
        let (bw, bh) = ((x1 - x0).max(1) as f64, (y1 - y0).max(1) as f64);
        let fy = (((y as f64 - y0 as f64) / bh - V.0) / (V.1 - V.0) * FH as f64).floor();
        let fx = (((x as f64 - x0 as f64) / bw - U.0) / (U.1 - U.0) * FW as f64).floor();
        if fy < 0.0 || fy >= FH as f64 {
            return (1.0, 0.0);
        }
        let dx = self.dx[fy as usize] * bw as f32;
        if fx < 0.0 || fx >= FW as f64 {
            return (1.0, dx);
        }
        (self.rank[fy as usize * FW + fx as usize], dx)
    }
}

#[inline]
fn transit(q: f64, rp: f32) -> f64 {
    (q * (1.0 + D) - rp as f64) / D                 // < 0 not yet, 0..1 in transit, > 1 done
}

/// The subject partly built (appear) or partly gone (vanish): `src` (premultiplied) -> a new
/// frame. Only the content area `cbox` (and where slices shove pixels to) can be non-zero.
pub fn start(src: &View, cbox: Rect, appear: bool, q: f64, map: &Map, bbox: (i64, i64, i64, i64)) -> Frame {
    let (w, h) = (src.w, src.h);
    let mut f = Frame::new(w, h);
    if cbox.is_empty() {
        return f;
    }
    let m = (((bbox.2 - bbox.0).max(1) / 45) as usize).max(4);   // mosaic block size
    // premultiplied mean of each m x m block (content area only; the rest is empty)
    let (gw, gh) = (w.div_ceil(m), h.div_ceil(m));
    let (by0, by1, bx0, bx1) = (cbox.y0 / m, cbox.y1.div_ceil(m), cbox.x0 / m, cbox.x1.div_ceil(m));
    let mut means = vec![[0f32; 4]; gw * gh];
    let rows: Vec<(usize, Vec<[f32; 4]>)> = (by0..by1)
        .into_par_iter()
        .map(|by| {
            let mut row = vec![[0f32; 4]; bx1 - bx0];
            for y in by * m..((by + 1) * m).min(h) {
                for x in bx0 * m..(bx1 * m).min(w) {
                    let p = src.px(x as i64, y as i64);
                    let c = &mut row[x / m - bx0];
                    for k in 0..4 {
                        c[k] += p[k];
                    }
                }
            }
            let n = (m * m) as f32;
            (by, row.into_iter().map(|c| [c[0] / n, c[1] / n, c[2] / n, c[3] / n]).collect())
        })
        .collect();
    for (by, row) in rows {
        means[by * gw + bx0..by * gw + bx1].copy_from_slice(&row);
    }
    let v = f.view();
    (0..h).into_par_iter().for_each(|y| {
        // SAFETY: one thread per row
        let out = unsafe { v.span_mut(y, 0, w) };
        for x in 0..w {
            let (rp, dx) = map.at(x, y, bbox);
            let s = transit(q, rp);
            let (amul, k) = if appear {
                if s >= 0.0 { (1.0, (1.0 - s).clamp(0.0, 1.0)) } else { (0.0, 0.0) }
            } else if s >= 0.0 {
                let a = (1.0 - s).clamp(0.0, 1.0);
                (a, if a > 0.0 { s.clamp(0.0, 1.0) } else { 0.0 })
            } else {
                (1.0, 0.0)
            };
            let mut p = src.px(x as i64, y as i64);
            if k > 0.0 {
                // in transit: shoved sideways, turning into its mosaic block, lit up
                let sx = (x as i64 - round_even(dx as f64 * k * k) as i64).clamp(0, w as i64 - 1);
                let sp = src.px(sx, y as i64);
                let mb = means[(y / m) * gw + sx as usize / m];
                let st = |c: [f32; 4]| if c[3] > 1e-6 { [c[0] / c[3], c[1] / c[3], c[2] / c[3]] } else { [0.0; 3] };
                let (s_rgb, m_rgb) = (st(sp), st(mb));
                let wgt = (k * 1.4 - 0.2).clamp(0.0, 1.0) as f32;
                let g = (k * k * k * 0.65) as f32;
                let alpha = sp[3] * (1.0 - wgt) + mb[3] * wgt;
                let mut rgb = [0f32; 3];
                for c in 0..3 {
                    rgb[c] = (s_rgb[c] * (1.0 - wgt) + m_rgb[c] * wgt) * (1.0 - g) + map.glow[c] * g;
                }
                p = [rgb[0] * alpha, rgb[1] * alpha, rgb[2] * alpha, alpha];
            }
            let a = amul as f32;
            out[4 * x..4 * x + 4].copy_from_slice(&[p[0] * a, p[1] * a, p[2] * a, p[3] * a]);
        }
    });
    f
}

/// After the effects: how much of the glitched picture is kept at (x, y) - only where the
/// subject exists, a little ahead of the front.
#[inline]
pub fn end_mul(appear: bool, q: f64, map: &Map, x: usize, y: usize, bbox: (i64, i64, i64, i64)) -> f32 {
    let s = transit(q, map.at(x, y, bbox).0);
    if appear { (s >= -0.35) as u8 as f32 } else { (1.35 - s).clamp(0.0, 1.0) as f32 }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn timeline_like_glitch_py() {
        // glitch.py: Timing(trans=[(10, "in", 1), (60, "out", 0.4)], fps=25).reveal(f)
        let t = [(10.0, true, 1.0), (60.0, false, 0.4)];
        assert_eq!(at(&t, 5.0, 25.0), Reveal::Hidden);                 // before the first appear
        assert_eq!(at(&t, 10.0, 25.0), Reveal::Moving(true, 1.0 / 25.0, 0));
        assert_eq!(at(&t, 34.0, 25.0), Reveal::Moving(true, 1.0, 0));
        assert_eq!(at(&t, 35.0, 25.0), Reveal::Visible);
        assert_eq!(at(&t, 64.0, 25.0), Reveal::Moving(false, 0.5, 1));
        assert_eq!(at(&t, 70.0, 25.0), Reveal::Hidden);
        assert_eq!(at(&[(10.0, false, 1.0)], 0.0, 25.0), Reveal::Visible); // a vanish first: there before
        assert_eq!(at(&[], 0.0, 25.0), Reveal::Visible);
        assert!(parse("12, 40:1.5", true, 1.0).unwrap() == vec![(12.0, true, 1.0), (40.0, true, 1.5)]);
        assert!(parse("12:0", true, 1.0).is_err() && parse("x", false, 1.0).is_err());
    }

    #[test]
    fn the_map_spreads_arrivals_evenly() {
        for idx in 0..6 {
            let m = Map::new(3, idx);
            let mut r = m.rank.clone();
            r.sort_by(|a, b| a.total_cmp(b));
            assert_eq!(r[0], 0.0);
            assert_eq!(*r.last().unwrap(), 1.0);
            assert!((r[r.len() / 2] - 0.5).abs() < 0.01);
            assert!(m.dx.iter().all(|d| d.abs() <= 0.28));
        }
        // same seed and index: the same map
        assert_eq!(Map::new(3, 1).rank, Map::new(3, 1).rank);
    }

    #[test]
    fn appear_builds_up_and_vanish_takes_away() {
        let (w, h) = (120usize, 160usize);
        let mut src = Frame::new(w, h);
        for y in 20..140 {
            for x in 30..90 {
                src.data[4 * (y * w + x)..4 * (y * w + x) + 4].copy_from_slice(&[0.3, 0.4, 0.5, 1.0]);
            }
        }
        let bbox = (30, 20, 90, 140);
        let cbox = Rect::new(20, 140, 30, 90);
        let map = Map::new(1, 0);
        let cover = |f: &Frame| f.data.chunks(4).map(|p| p[3]).sum::<f32>();
        let full = cover(&src);
        let mut last = -1.0;
        for q in [0.05, 0.3, 0.6, 0.9, 1.0] {
            let c = cover(&start(&src.view(), cbox, true, q, &map, bbox));
            assert!(c >= last - 0.02 * full, "appear shrank at {q}");
            last = c;
        }
        assert!(last > 0.97 * full);
        let gone = cover(&start(&src.view(), cbox, false, 1.0, &map, bbox));
        assert!(gone < 0.05 * full, "{gone}");
        let f = start(&src.view(), cbox, true, 0.5, &map, bbox);
        assert!(f.data.iter().all(|v| v.is_finite() && *v >= 0.0));
        assert!(f.data.chunks(4).all(|p| p[0].max(p[1]).max(p[2]) <= p[3] + 1e-5));
    }
}
