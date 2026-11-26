//! Soft masks that limit where the glitch may change the picture (glitch.py's part_mask and
//! _spill_mask): body parts drawn as capsules around MediaPipe landmarks, and the area around the
//! subject in which it may damage a background plate. Both are drawn on a coarse grid (they are
//! soft anyway) and sampled bilinearly, as glitch.py's PIL resize does.

use crate::image::View;
use crate::layers::Mask;
use rayon::prelude::*;

pub const PARTS: [&str; 7] = ["face", "head", "torso", "arms", "hands", "legs", "feet"];
const FACE: u32 = 1;
const HEAD: u32 = 2;
const TORSO: u32 = 4;
const ARMS: u32 = 8;
const HANDS: u32 = 16;
const LEGS: u32 = 32;
const FEET: u32 = 64;

/// "head,hands" / "upper" -> part bits (face 1, head 2, torso 4, arms 8, hands 16, legs 32,
/// feet 64). Groups: upper, lower, body; "all" and empty items are ignored. Parts may be
/// separated by commas, '+', or spaces.
pub fn parse_parts(text: &str) -> Result<u32, String> {
    let mut bits = 0;
    for it in text.split([',', '+', ' ', '\t']).map(|s| s.trim().to_lowercase()) {
        bits |= match it.as_str() {
            "" | "all" | "none" => 0,
            "upper" => HEAD | TORSO | ARMS | HANDS,
            "lower" => LEGS | FEET,
            "body" => 127,
            p => match PARTS.iter().position(|q| *q == p) {
                Some(i) => 1 << i,
                None => return Err(format!("unknown body part '{p}': use {}, upper, lower or body", PARTS.join(", "))),
            },
        };
    }
    Ok(bits)
}

/// "mosh=head; jpeg=torso+arms" -> per-effect part bits (glitch.py's order). Pairs are
/// separated by ';' or new lines.
pub fn parse_targets(text: &str, effects: &[&str]) -> Result<[u32; 13], String> {
    let mut out = [0u32; 13];
    for pair in text.split([';', '\n']).map(str::trim).filter(|s| !s.is_empty()) {
        let (fx, parts) = pair.split_once('=').ok_or_else(|| format!("'{pair}': expected EFFECT=PARTS, e.g. mosh=head"))?;
        let fx = fx.trim().to_lowercase();
        let i = effects.iter().position(|e| *e == fx).ok_or_else(|| format!("unknown effect '{fx}'"))?;
        out[i] |= parse_parts(parts)?;
    }
    Ok(out)
}

/// 33 MediaPipe body landmarks: x, y in canvas pixels, visibility.
pub type Landmarks = [[f64; 3]; 33];

type Capsule = ([f64; 2], [f64; 2], f64);

fn mean(lm: &Landmarks, ix: &[usize]) -> [f64; 2] {
    let n = ix.len() as f64;
    let (x, y) = ix.iter().fold((0.0, 0.0), |(a, b), &i| (a + lm[i][0], b + lm[i][1]));
    [x / n, y / n]
}

fn dist(a: [f64; 2], b: [f64; 2]) -> f64 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)).sqrt()
}

/// Body size S: shoulder width, or half the torso length when side-on, at least 4 px.
fn body_size(lm: &Landmarks) -> f64 {
    let (ls, rs) = (mean(lm, &[11]), mean(lm, &[12]));
    let (ms, mh) = (mean(lm, &[11, 12]), mean(lm, &[23, 24]));
    dist(ls, rs).max(0.55 * dist(ms, mh)).max(4.0)
}

/// Capsules (p, q, radius) for one part (glitch.py's _part_shapes); a circle is p == q.
fn part_shapes(lm: &Landmarks, part: u32) -> Vec<Capsule> {
    let p = |ix: &[usize]| mean(lm, ix);
    let (ls, rs, lh, rh) = (p(&[11]), p(&[12]), p(&[23]), p(&[24]));
    let ms = [(ls[0] + rs[0]) / 2.0, (ls[1] + rs[1]) / 2.0];
    let mh = [(lh[0] + rh[0]) / 2.0, (lh[1] + rh[1]) / 2.0];
    let s = body_size(lm);
    let rhd = (0.62 * dist(p(&[7]), p(&[8]))).max(0.3 * s);                  // head radius
    let c0 = p(&[0, 7, 8]);
    let hc = [c0[0], c0[1] - 0.25 * rhd];                                  // a little up: hair, crown
    match part {
        HEAD => vec![(hc, hc, 1.4 * rhd), (hc, ms, 0.18 * s)],
        FACE => {
            let fc = p(&[0, 2, 5, 9, 10]);
            vec![(fc, fc, 0.72 * rhd)]
        }
        TORSO => vec![(ms, mh, 0.42 * s), (ls, rs, 0.14 * s), (lh, rh, 0.16 * s), (ls, lh, 0.15 * s), (rs, rh, 0.15 * s)],
        ARMS => {
            let r = 0.14 * s;
            vec![(p(&[11]), p(&[13]), r), (p(&[13]), p(&[15]), r), (p(&[12]), p(&[14]), r), (p(&[14]), p(&[16]), r)]
        }
        HANDS => {
            let r = 0.11 * s;
            let (l, rr) = (p(&[15, 17, 19, 21]), p(&[16, 18, 20, 22]));
            vec![(p(&[15]), p(&[17, 19]), r), (l, l, 0.13 * s), (p(&[16]), p(&[18, 20]), r), (rr, rr, 0.13 * s)]
        }
        LEGS => {
            let r = 0.21 * s;
            vec![(p(&[23]), p(&[25]), r), (p(&[25]), p(&[27]), r), (p(&[24]), p(&[26]), r), (p(&[26]), p(&[28]), r)]
        }
        FEET => {
            let r = 0.08 * s;
            vec![(p(&[27]), p(&[29]), r), (p(&[29]), p(&[31]), r), (p(&[27]), p(&[31]), r),
                 (p(&[28]), p(&[30]), r), (p(&[30]), p(&[32]), r), (p(&[28]), p(&[32]), r)]
        }
        _ => vec![],
    }
}

/// A coarse grid, sampled bilinearly at full resolution (PIL's upscaling resize, cropped).
#[derive(Clone)]
pub struct Grid {
    q: usize,
    gw: usize,
    gh: usize,
    data: Vec<f32>,
}

impl Grid {
    fn new(q: usize, w: usize, h: usize) -> Grid {
        let (gw, gh) = (w.div_ceil(q), h.div_ceil(q));
        Grid { q, gw, gh, data: vec![0.0; gw * gh] }
    }
    #[inline]
    fn at(&self, x: usize, y: usize) -> f32 {
        let q = self.q as f32;
        let u = ((x as f32 + 0.5) / q - 0.5).clamp(0.0, (self.gw - 1) as f32);
        let v = ((y as f32 + 0.5) / q - 0.5).clamp(0.0, (self.gh - 1) as f32);
        let (i, j) = (u as usize, v as usize);
        let (i1, j1) = ((i + 1).min(self.gw - 1), (j + 1).min(self.gh - 1));
        let (fu, fv) = (u - i as f32, v - j as f32);
        let d = &self.data;
        let a = d[j * self.gw + i] * (1.0 - fu) + d[j * self.gw + i1] * fu;
        let b = d[j1 * self.gw + i] * (1.0 - fu) + d[j1 * self.gw + i1] * fu;
        a * (1.0 - fv) + b * fv
    }
}

/// Soft mask of body parts (glitch.py's part_mask), drawn at quarter resolution.
#[derive(Clone)]
pub struct PartMask {
    grid: Grid,
}

impl PartMask {
    /// `parts` = part bits; `feather` scales the soft edge (1 = default).
    pub fn new(lm: &Landmarks, parts: u32, w: usize, h: usize, feather: f64) -> PartMask {
        let mut g = Grid::new(4, w, h);
        if lm.iter().any(|p| !p[0].is_finite() || !p[1].is_finite()) {
            return PartMask { grid: g };                       // no usable pose: an empty mask
        }
        let fe = (0.08 * body_size(lm) * feather.max(0.0)).max(2.0);
        let q = g.q as f64;
        let (gw, gh) = (g.gw, g.gh);
        for bit in 0..7 {
            if parts & (1 << bit) == 0 {
                continue;
            }
            for (p0, p1, r) in part_shapes(lm, 1 << bit) {
                let d = [p1[0] - p0[0], p1[1] - p0[1]];
                let l2 = d[0] * d[0] + d[1] * d[1];
                let (x0, x1) = (p0[0].min(p1[0]) - r - fe, p0[0].max(p1[0]) + r + fe);   // only its box
                let (y0, y1) = (p0[1].min(p1[1]) - r - fe, p0[1].max(p1[1]) + r + fe);
                let (i0, i1) = (((y0 / q).floor().max(0.0)) as usize, (((y1 / q).floor() + 2.0).max(0.0) as usize).min(gh));
                let (j0, j1) = (((x0 / q).floor().max(0.0)) as usize, (((x1 / q).floor() + 2.0).max(0.0) as usize).min(gw));
                for i in i0..i1 {
                    let yy = i as f64 * q + q / 2.0;
                    for j in j0..j1 {
                        let xx = j as f64 * q + q / 2.0;
                        let dd = if l2 < 1e-6 {
                            dist([xx, yy], p0)
                        } else {
                            let t = (((xx - p0[0]) * d[0] + (yy - p0[1]) * d[1]) / l2).clamp(0.0, 1.0);
                            dist([xx, yy], [p0[0] + t * d[0], p0[1] + t * d[1]])
                        };
                        let v = ((r + fe / 2.0 - dd) / fe).clamp(0.0, 1.0) as f32;
                        let c = &mut g.data[i * gw + j];
                        *c = c.max(v);
                    }
                }
            }
        }
        PartMask { grid: g }
    }
}

impl Mask for PartMask {
    fn at(&self, x: usize, y: usize) -> f32 {
        self.grid.at(x, y).clamp(0.0, 1.0)
    }
}

/// Where the glitch as a whole may act: inside `only` (if given) and outside `protect`.
pub struct Allow {
    pub only: Option<PartMask>,
    pub protect: Option<PartMask>,
}

impl Mask for Allow {
    fn at(&self, x: usize, y: usize) -> f32 {
        let a = self.only.as_ref().map_or(1.0, |m| m.at(x, y));
        if a <= 0.0 {
            return 0.0;
        }
        a * self.protect.as_ref().map_or(1.0, |m| 1.0 - m.at(x, y))
    }
}

/// Where the glitch may change the picture once the subject sits on a plate: the subject
/// itself, plus a soft falloff around it that reaches ~spill x box size (glitch.py's
/// _spill_mask). Distances on an 8 px grid.
pub struct Spill {
    alpha: View,
    grid: Option<Grid>,
    reach: usize,
}

impl Spill {
    /// `cut` = the subject (premultiplied; its alpha), `bbox` = the layout box.
    pub fn new(cut: View, spill: f64, bbox: (i64, i64, i64, i64)) -> Spill {
        if spill.is_nan() || spill <= 0.0 {
            return Spill { alpha: cut, grid: None, reach: 0 };
        }
        let mut g = Grid::new(8, cut.w, cut.h);
        let (gw, gh) = (g.gw, g.gh);
        // cell = the max alpha in its 8x8 pixels
        let small: Vec<f32> = (0..gh)
            .into_par_iter()
            .flat_map_iter(|i| {
                let mut row = vec![0f32; gw];
                for y in i * 8..((i + 1) * 8).min(cut.h) {
                    let r = cut.row(y);
                    for (x, px) in r.chunks_exact(4).enumerate() {
                        let a = px[3].clamp(0.0, 1.0);
                        if a > row[x / 8] {
                            row[x / 8] = a;
                        }
                    }
                }
                row
            })
            .collect();
        let side = ((bbox.2 - bbox.0).max(bbox.3 - bbox.1)) as f64;
        let rr = (crate::detmath::round_even(spill * 0.5 * side / 8.0) as i64).max(1);
        // Chebyshev distance in cells from the subject (= growing it one 8-connected ring per
        // step), two-pass chamfer; beyond rr it no longer matters
        let big = i64::MAX / 4;
        let mut d: Vec<i64> = small.iter().map(|&a| if a > 0.5 { 0 } else { big }).collect();
        for i in 0..gh {
            for j in 0..gw {
                let mut v = d[i * gw + j];
                if i > 0 {
                    v = v.min(d[(i - 1) * gw + j] + 1);
                    if j > 0 { v = v.min(d[(i - 1) * gw + j - 1] + 1); }
                    if j + 1 < gw { v = v.min(d[(i - 1) * gw + j + 1] + 1); }
                }
                if j > 0 { v = v.min(d[i * gw + j - 1] + 1); }
                d[i * gw + j] = v;
            }
        }
        for i in (0..gh).rev() {
            for j in (0..gw).rev() {
                let mut v = d[i * gw + j];
                if i + 1 < gh {
                    v = v.min(d[(i + 1) * gw + j] + 1);
                    if j > 0 { v = v.min(d[(i + 1) * gw + j - 1] + 1); }
                    if j + 1 < gw { v = v.min(d[(i + 1) * gw + j + 1] + 1); }
                }
                if j + 1 < gw { v = v.min(d[i * gw + j + 1] + 1); }
                d[i * gw + j] = v;
            }
        }
        for (c, &dd) in g.data.iter_mut().zip(d.iter()) {
            let b = if dd >= big { 0.0 } else { (1.0 - dd as f64 / rr as f64).clamp(0.0, 1.0) };
            *c = (b * b * (3.0 - 2.0 * b)) as f32;               // full next to the body, 0 at rr
        }
        Spill { alpha: cut, grid: Some(g), reach: (rr as usize + 2) * 8 }
    }
}

impl Spill {
    /// How far past the subject (px) the mask can be above zero.
    pub fn reach(&self) -> usize {
        self.reach
    }
}

impl Mask for Spill {
    fn at(&self, x: usize, y: usize) -> f32 {
        let a = self.alpha.px(x as i64, y as i64)[3].clamp(0.0, 1.0);
        match &self.grid {
            None => a,
            Some(g) => a.max(g.at(x, y)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::image::Frame;

    fn standing(cx: f64, top: f64, s: f64) -> Landmarks {
        // a rough person: head at top, shoulders, hips, knees, ankles, feet, arms down
        let mut lm = [[0.0, 0.0, 1.0]; 33];
        let set = |lm: &mut Landmarks, i: usize, x: f64, y: f64| lm[i] = [cx + x * s, top + y * s, 1.0];
        for (i, x, y) in [(0, 0.0, 0.3), (2, -0.1, 0.25), (5, 0.1, 0.25), (7, -0.25, 0.3), (8, 0.25, 0.3),
                          (9, -0.05, 0.4), (10, 0.05, 0.4), (11, -0.5, 0.8), (12, 0.5, 0.8), (13, -0.6, 1.4),
                          (14, 0.6, 1.4), (15, -0.65, 1.9), (16, 0.65, 1.9), (17, -0.7, 2.0), (18, 0.7, 2.0),
                          (19, -0.65, 2.05), (20, 0.65, 2.05), (21, -0.6, 2.0), (22, 0.6, 2.0), (23, -0.3, 2.0),
                          (24, 0.3, 2.0), (25, -0.3, 2.9), (26, 0.3, 2.9), (27, -0.3, 3.8), (28, 0.3, 3.8),
                          (29, -0.35, 3.9), (30, 0.35, 3.9), (31, -0.2, 3.95), (32, 0.2, 3.95)] {
            set(&mut lm, i, x, y);
        }
        lm
    }

    #[test]
    fn parts_parse_with_groups_and_errors() {
        assert_eq!(parse_parts("head, hands").unwrap(), HEAD | HANDS);
        assert_eq!(parse_parts("upper").unwrap(), HEAD | TORSO | ARMS | HANDS);
        assert_eq!(parse_parts("").unwrap(), 0);
        assert!(parse_parts("tail").is_err());
        let fx = crate::sched::EFFECTS;
        let t = parse_targets("mosh=head; jpeg = torso+arms", &fx).unwrap();
        assert_eq!(t[12], HEAD);
        assert_eq!(t[0], TORSO | ARMS);
        assert!(parse_targets("mosh", &fx).is_err());
    }

    #[test]
    fn head_mask_covers_the_head_not_the_feet() {
        let lm = standing(200.0, 40.0, 100.0);
        let m = PartMask::new(&lm, HEAD, 400, 500, 1.0);
        assert!(m.at(200, 70) > 0.99);          // head centre
        assert!(m.at(200, 420) < 0.01);         // feet
        let f = PartMask::new(&lm, FEET, 400, 500, 1.0);
        assert!(f.at(170, 430) > 0.5);
        assert!(f.at(200, 70) < 0.01);
    }

    #[test]
    fn spill_reaches_out_from_the_subject_and_fades() {
        let mut f = Frame::new(400, 300);
        for y in 100..200 {
            for x in 150..250 {
                f.data[4 * (y * 400 + x) + 3] = 1.0;
            }
        }
        let s0 = Spill::new(f.view(), 0.0, (150, 100, 250, 200));
        assert_eq!(s0.at(149, 150), 0.0);
        assert_eq!(s0.at(160, 150), 1.0);
        let s = Spill::new(f.view(), 0.5, (150, 100, 250, 200));   // ~25 px
        assert!(s.at(145, 150) > 0.5);
        assert!(s.at(100, 150) < 0.01);
        assert!(s.at(130, 150) < s.at(145, 150));
    }
}
