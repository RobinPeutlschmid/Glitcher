//! What the app knows about the whole shot before rendering a frame, gathered by a host that
//! feeds every frame of its range once (glitch.py's planner pass):
//! - the layout box: the union of the subject's boxes over the shot (the app's default), or a
//!   box that follows the subject's smoothed centre (`--follow`, the app's default in 3D);
//! - how fast the subject moves at each frame, for the motion-reactive glitch (`--motion`).
//!
//! The boxes and motion are kept per node; the follow boxes and the motion curve are computed
//! once, when the scan is complete.

use crate::detmath::exp;
use crate::engine::layout_box;
use crate::image::View;
use rayon::prelude::*;

type Box4 = (i64, i64, i64, i64);
/// A follow box, fractional (cx +- w/2).
pub type FBox = (f64, f64, f64, f64);

/// Motion features: a small luminance x alpha and alpha image (longest side ~160 px), 0..255.
struct Feat {
    w: usize,
    data: Vec<[u8; 2]>,
}

pub struct ShotBox {
    /// Which scan this is (a node's Rescan count): a new one starts over.
    pub generation: i64,
    pub first: i64,
    pub last: i64,
    pub size: (usize, usize),
    boxes: Vec<Option<Option<Box4>>>,   // None = not scanned, Some(None) = empty frame
    feats: Vec<Option<Feat>>,
    diffs: Vec<Option<f64>>,            // motion between frame i-1 and i
    follow_cache: Option<(u64, u64, Vec<FBox>)>,
    motion_cache: Option<Vec<f64>>,
}

/// numpy's percentile (linear between the closest ranks).
fn percentile(v: &[f64], q: f64) -> f64 {
    let mut s: Vec<f64> = v.iter().copied().filter(|x| x.is_finite()).collect();
    if s.is_empty() {
        return 0.0;
    }
    s.sort_by(|a, b| a.total_cmp(b));
    let pos = q / 100.0 * (s.len() - 1) as f64;
    let (i, f) = (pos.floor() as usize, pos - pos.floor());
    if i + 1 >= s.len() { s[s.len() - 1] } else { s[i] + (s[i + 1] - s[i]) * f }
}

/// np.interp over the known entries (held beyond the ends).
fn fill(v: &mut [f64]) {
    let known: Vec<usize> = (0..v.len()).filter(|&i| v[i].is_finite()).collect();
    if known.is_empty() {
        return;
    }
    for i in 0..v.len() {
        if v[i].is_finite() {
            continue;
        }
        let j = known.partition_point(|&k| k < i);
        v[i] = if j == 0 {
            v[known[0]]
        } else if j == known.len() {
            v[known[known.len() - 1]]
        } else {
            let (a, b) = (known[j - 1], known[j]);
            v[a] + (v[b] - v[a]) * (i - a) as f64 / (b - a) as f64
        };
    }
}

fn feature(v: &View) -> Feat {
    let s = ((v.w.max(v.h) as f64 / 160.0).round() as usize).max(1);
    let (w, h) = (v.w / s, v.h / s);
    let data = (0..h)
        .into_par_iter()
        .flat_map_iter(|y| {
            (0..w).map(move |x| {
                let mut acc = [0f32; 4];
                for yy in y * s..(y + 1) * s {
                    for xx in x * s..(x + 1) * s {
                        let p = v.px(xx as i64, yy as i64);
                        for c in 0..4 {
                            acc[c] += p[c];
                        }
                    }
                }
                let n = (s * s) as f32;
                let a = (acc[3] / n).clamp(0.0, 1.0);
                // luminance of the premultiplied colour = straight luminance x alpha (as glitch.py)
                let l = ((acc[0] * 0.299 + acc[1] * 0.587 + acc[2] * 0.114) / n).clamp(0.0, a);
                [(l * 255.0 + 0.5) as u8, (a * 255.0 + 0.5) as u8]
            })
        })
        .collect();
    Feat { w, data }
}

/// Mean absolute change inside the subject (either frame's alpha > 8), as glitch.py's
/// motion_curve; None if the frames have no subject.
fn diff(a: &Feat, b: &Feat) -> Option<f64> {
    if a.w != b.w || a.data.len() != b.data.len() {
        return Some(0.0);
    }
    let (mut s, mut n) = (0f64, 0usize);
    for (p, q) in a.data.iter().zip(b.data.iter()) {
        if p[1].max(q[1]) > 8 {
            s += (p[0] as f64 - q[0] as f64).abs() + (p[1] as f64 - q[1] as f64).abs();
            n += 2;
        }
    }
    Some(if n > 0 { s / n as f64 } else { 0.0 })
}

impl ShotBox {
    pub fn new(first: i64, last: i64, w: usize, h: usize) -> ShotBox {
        let n = (last - first + 1).clamp(0, 1_000_000) as usize;
        ShotBox { generation: 0, first, last, size: (w, h), boxes: vec![None; n], feats: (0..n).map(|_| None).collect(),
                  diffs: vec![None; n], follow_cache: None, motion_cache: None }
    }

    pub fn same(&self, first: i64, last: i64, w: usize, h: usize) -> bool {
        (self.first, self.last, self.size) == (first, last, (w, h))
    }

    pub fn missing(&self, max: usize) -> Vec<i64> {
        (0..self.boxes.len()).filter(|&i| self.boxes[i].is_none()).take(max).map(|i| self.first + i as i64).collect()
    }

    pub fn remaining(&self) -> usize {
        self.boxes.iter().filter(|b| b.is_none()).count()
    }

    pub fn complete(&self) -> bool {
        self.boxes.iter().all(|b| b.is_some())
    }

    /// Frame `frame`'s picture (None = the host has none there).
    pub fn feed(&mut self, frame: i64, img: Option<&View>) {
        if frame < self.first || frame > self.last {
            return;
        }
        let i = (frame - self.first) as usize;
        if self.boxes[i].is_some() {
            return;
        }
        let v = img.filter(|v| (v.w, v.h) == self.size);
        self.boxes[i] = Some(v.and_then(layout_box));
        self.feats[i] = Some(v.map(feature).unwrap_or(Feat { w: 0, data: vec![] }));
        // motion with each neighbour that is there; a feature is dropped once both are used
        for j in [i, i + 1] {
            if j >= 1 && j < self.feats.len() && self.diffs[j].is_none() {
                if let (Some(a), Some(b)) = (&self.feats[j - 1], &self.feats[j]) {
                    self.diffs[j] = diff(a, b);
                }
            }
        }
        for j in i.saturating_sub(1)..=(i + 1).min(self.feats.len() - 1) {
            let left = j == 0 || self.diffs[j].is_some();
            let right = j + 1 >= self.diffs.len() || self.diffs[j + 1].is_some();
            if left && right {
                self.feats[j] = None;
            }
        }
        self.follow_cache = None;
        self.motion_cache = None;
    }

    /// The union over the shot, once every frame is scanned (Some(None): nothing in any frame).
    pub fn result(&self) -> Option<Option<Box4>> {
        if !self.complete() {
            return None;
        }
        Some(self.boxes.iter().flatten().flatten().fold(None, |u: Option<Box4>, b| {
            Some(match u {
                None => *b,
                Some(u) => (u.0.min(b.0), u.1.min(b.1), u.2.max(b.2), u.3.max(b.3)),
            })
        }))
    }

    /// The follow box at `frame` (glitch.py's _plan_boxes with --follow): the subject's centre
    /// smoothed over `sig` frames (Gaussian), a constant size (the 90th percentile of the
    /// subject's) - or, with `scales` (the pose's apparent size per frame of the range), a
    /// constant body-relative size times that scale. None until the scan is complete or if the
    /// shot has no subject.
    pub fn follow(&mut self, frame: i64, sig: f64, scales: Option<&[f64]>) -> Option<FBox> {
        if !self.complete() || frame < self.first - 1_000_000 {
            return None;
        }
        let skey = scales.map_or(0, |s| s.iter().fold(0xcbf29ce484222325u64, |h, v| (h ^ v.to_bits()).wrapping_mul(0x100000001b3)) | 1);
        if !matches!(&self.follow_cache, Some((a, b, _)) if *a == sig.to_bits() && *b == skey) {
            let n = self.boxes.len();
            let mut cols: [Vec<f64>; 4] = std::array::from_fn(|_| vec![f64::NAN; n]);
            for (i, b) in self.boxes.iter().enumerate() {
                if let Some(Some((x0, y0, x1, y1))) = b {
                    let v = [(x0 + x1) as f64 / 2.0, (y0 + y1) as f64 / 2.0, (x1 - x0) as f64, (y1 - y0) as f64];
                    for j in 0..4 {
                        cols[j][i] = v[j];
                    }
                }
            }
            if cols[0].iter().all(|v| !v.is_finite()) {
                return None;
            }
            for c in cols.iter_mut() {
                fill(c);                                     // hold / interpolate over empty frames
            }
            if sig > 0.3 && n > 1 {
                let r = (3.0 * sig) as i64 + 1;
                let k: Vec<f64> = (-r..=r).map(|i| exp(-0.5 * (i as f64 / sig).powi(2))).collect();
                let ks: f64 = k.iter().sum();
                for c in cols.iter_mut().take(2) {
                    let src = c.clone();
                    for (i, v) in c.iter_mut().enumerate() {
                        *v = (-r..=r).map(|o| src[(i as i64 + o).clamp(0, n as i64 - 1) as usize] * k[(o + r) as usize]).sum::<f64>() / ks;
                    }
                }
            }
            let boxes: Vec<FBox> = match scales.filter(|s| s.len() == n) {
                None => {
                    let (wr, hr) = (percentile(&cols[2], 90.0), percentile(&cols[3], 90.0));
                    (0..n).map(|i| (cols[0][i] - wr / 2.0, cols[1][i] - hr / 2.0, cols[0][i] + wr / 2.0, cols[1][i] + hr / 2.0)).collect()
                }
                Some(sc) => {
                    let sc: Vec<f64> = sc.iter().map(|v| if v.is_finite() && *v > 1e-3 { *v } else { 1.0 }).collect();
                    let wr = percentile(&(0..n).map(|i| cols[2][i] / sc[i]).collect::<Vec<_>>(), 90.0);
                    let hr = percentile(&(0..n).map(|i| cols[3][i] / sc[i]).collect::<Vec<_>>(), 90.0);
                    (0..n).map(|i| {
                        let (w, h) = (wr * sc[i], hr * sc[i]);
                        (cols[0][i] - w / 2.0, cols[1][i] - h / 2.0, cols[0][i] + w / 2.0, cols[1][i] + h / 2.0)
                    }).collect()
                }
            };
            self.follow_cache = Some((sig.to_bits(), skey, boxes));
        }
        let b = &self.follow_cache.as_ref()?.2;
        let i = (frame - self.first).clamp(0, b.len() as i64 - 1) as usize;
        Some(b[i])
    }

    /// How fast the subject moves at `frame` (glitch.py's motion_curve): 0..1.5, 1 = the shot's
    /// fast moments (its 95th percentile). None until the scan is complete.
    pub fn motion(&mut self, frame: i64) -> Option<f64> {
        if !self.complete() {
            return None;
        }
        if self.motion_cache.is_none() {
            let n = self.diffs.len();
            let mut m: Vec<f64> = self.diffs.iter().map(|d| d.unwrap_or(0.0)).collect();
            if n >= 2 {
                m[0] = m[1];
            }
            if n >= 3 {                                      // [1 2 1]: frame-difference jitter
                let src = m.clone();
                for i in 0..n {
                    let (a, b) = (src[i.saturating_sub(1)], src[(i + 1).min(n - 1)]);
                    m[i] = 0.25 * a + 0.5 * src[i] + 0.25 * b;
                }
            }
            let rf = percentile(&m, 95.0);
            let m = if n < 2 || rf <= 1e-6 { vec![0.0; n] } else { m.iter().map(|v| (v / rf).clamp(0.0, 1.5)).collect() };
            self.motion_cache = Some(m);
        }
        let m = self.motion_cache.as_ref()?;
        if m.is_empty() {
            return Some(0.0);
        }
        Some(m[(frame - self.first).clamp(0, m.len() as i64 - 1) as usize])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::image::Frame;

    fn disc(cx: f64, cy: f64, r: f64, w: usize, h: usize) -> Frame {
        let mut f = Frame::new(w, h);
        for y in 0..h {
            for x in 0..w {
                if ((x as f64 - cx).powi(2) + (y as f64 - cy).powi(2)).sqrt() < r {
                    let k = 4 * (y * w + x);
                    f.data[k..k + 4].copy_from_slice(&[0.5, 0.4, 0.3, 1.0]);
                }
            }
        }
        f
    }

    #[test]
    fn union_over_the_shot() {
        let mut s = ShotBox::new(10, 12, 50, 40);
        let mut f = Frame::new(50, 40);
        f.data[4 * (5 * 50 + 7) + 3] = 1.0;
        s.feed(10, Some(&f.view()));
        assert_eq!(s.result(), None);
        assert_eq!(s.missing(5), vec![11, 12]);
        let mut g = Frame::new(50, 40);
        g.data[4 * (30 * 50 + 40) + 3] = 1.0;
        s.feed(11, Some(&g.view()));
        s.feed(12, None);
        assert_eq!(s.result(), Some(Some((7, 5, 42, 32))));      // a box is at least 2 px
    }

    #[test]
    fn follow_box_tracks_the_subject_with_a_constant_size() {
        let (w, h, n) = (200usize, 100usize, 30i64);
        let mut s = ShotBox::new(0, n - 1, w, h);
        for i in (0..n).rev() {                                  // any order
            s.feed(i, Some(&disc(30.0 + 4.0 * i as f64, 50.0, 12.0, w, h).view()));
        }
        assert!(s.follow(0, 10.0, None).is_some());
        let sizes: Vec<f64> = (0..n).map(|i| { let b = s.follow(i, 0.0, None).unwrap(); b.2 - b.0 }).collect();
        assert!(sizes.iter().all(|v| (v - sizes[0]).abs() < 1e-9));
        let c = |i: i64, sig: f64, s: &mut ShotBox| { let b = s.follow(i, sig, None).unwrap(); (b.0 + b.2) / 2.0 };
        assert!((c(15, 0.0, &mut s) - 90.0).abs() < 1.0);         // unsmoothed: on the subject
        assert!(c(29, 0.0, &mut s) - c(0, 0.0, &mut s) > 100.0);
        // smoothing pulls the ends toward the middle; the middle stays
        assert!(c(0, 10.0, &mut s) > c(0, 0.0, &mut s));
        assert!((c(15, 10.0, &mut s) - c(15, 0.0, &mut s)).abs() < 2.0);
        // with pose scales the box grows with the scale
        let sc: Vec<f64> = (0..n).map(|i| 1.0 + i as f64 / 29.0).collect();
        let a = s.follow(0, 0.0, Some(&sc)).unwrap();
        let b = s.follow(29, 0.0, Some(&sc)).unwrap();
        assert!(((b.2 - b.0) / (a.2 - a.0) - 2.0).abs() < 1e-6);
    }

    #[test]
    fn motion_is_high_where_the_subject_moves() {
        let (w, h, n) = (160usize, 90usize, 40i64);
        let mut s = ShotBox::new(0, n - 1, w, h);
        let x = |i: i64| if i < 20 { 40.0 } else { 40.0 + 5.0 * (i - 20) as f64 };      // still, then moving
        for i in 0..n {
            s.feed(i, Some(&disc(x(i), 45.0, 15.0, w, h).view()));
        }
        assert!(s.feats.iter().all(|f| f.is_none()), "features not freed");
        let still = s.motion(10).unwrap();
        let moving = s.motion(32).unwrap();
        assert!(still < 0.05 && moving > 0.8, "{still} {moving}");
    }
}
