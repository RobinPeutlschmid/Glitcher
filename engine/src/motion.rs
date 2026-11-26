//! Block motion between two frames, for the datamosh: a port of glitch.py's _feat and
//! _block_motion. Exhaustive search of 8x8 blocks, +-3 px, on a downscaled luminance + alpha
//! image, with a small bias toward no motion.

use crate::detmath::hash;
use crate::image::View;
use rayon::prelude::*;
use std::sync::{Arc, Mutex};

pub const MB: usize = 8; // block size, in low-res pixels
pub const MR: i64 = 3; // search radius, in low-res pixels

/// Low-res matching image: luminance where visible plus some alpha, box-averaged by `s`.
pub struct Feat {
    pub w: usize,
    pub h: usize,
    pub v: Vec<f32>,
}

pub fn feature(img: &View, s: usize) -> Feat {
    let (w, h) = (img.w / s, img.h / s);
    let mut v = vec![0f32; w * h];
    let inv = 1.0 / (s * s) as f32;
    v.par_chunks_mut(w.max(1)).enumerate().for_each(|(fy, row)| {
        let mut acc = vec![[0f32; 4]; w];
        for yy in 0..s {
            let r = img.row(fy * s + yy);
            for (fx, a) in acc.iter_mut().enumerate() {
                for xx in 0..s {
                    let i = 4 * (fx * s + xx);
                    a[0] += r[i];
                    a[1] += r[i + 1];
                    a[2] += r[i + 2];
                    a[3] += r[i + 3];
                }
            }
        }
        for (o, a) in row.iter_mut().zip(acc.iter()) {
            let (r, g, b, al) = (a[0] * inv, a[1] * inv, a[2] * inv, a[3] * inv);
            let lum = (r * 0.299 + g * 0.587 + b * 0.114).min(al.max(0.0));   // premultiplied
            *o = 0.8 * lum + 0.2 * al;
        }
    });
    Feat { w, h, v }
}

/// Backward motion: for every block of `nxt`, the offset (dy, dx) in full-res px where its
/// content sat in `prev` (nxt(x) ~ prev(x + d)). `bs` = block size in full-res px.
pub struct Motion {
    pub bx: usize,
    pub by: usize,
    pub vy: Vec<i32>,
    pub vx: Vec<i32>,
    pub bs: usize,
}

pub fn block_motion(fp: &Feat, fnx: &Feat, s: usize) -> Motion {
    let (by, bx) = (fnx.h / MB, fnx.w / MB);
    let mut vy = vec![0i32; by * bx];
    let mut vx = vec![0i32; by * bx];
    let bias = 0.5 * (MB * MB) as f32 / 255.0;             // glitch.py's 0..255 bias, on 0..1 data
    // blocks whose search window is empty in both frames always pick (0, 0): search only the
    // block rows/columns around content
    let mut lo = (usize::MAX, usize::MAX);
    let mut hi = (0usize, 0usize);
    for y in 0..by * MB {
        for x in 0..bx * MB {
            if fp.v[y * fp.w + x] > 0.0 || fnx.v[y * fnx.w + x] > 0.0 {
                lo = (lo.0.min(y), lo.1.min(x));
                hi = (hi.0.max(y), hi.1.max(x));
            }
        }
    }
    if lo.0 == usize::MAX || by == 0 || bx == 0 {
        return Motion { bx, by, vy, vx, bs: MB * s };
    }
    let r = MR as usize;
    let b0 = lo.0.saturating_sub(r) / MB;
    let b1 = ((hi.0 + r) / MB + 1).min(by);
    let c0 = lo.1.saturating_sub(r) / MB;
    let c1 = ((hi.1 + r) / MB + 1).min(bx);
    let pw = fp.w as i64;
    let ph = fp.h as i64;
    let sample = |y: i64, x: i64| fp.v[(y.clamp(0, ph - 1) * pw + x.clamp(0, pw - 1)) as usize]; // edge pad
    let rows: Vec<Vec<(i32, i32)>> = (b0..b1)
        .into_par_iter()
        .map(|bi| {
            (c0..c1)
                .map(|bj| {
                    let (mut best, mut w) = (f32::INFINITY, (0i32, 0i32));
                    for dy in -MR..=MR {
                        for dx in -MR..=MR {
                            let mut sad = 0f32;
                            for u in 0..MB {
                                let y = (bi * MB + u) as i64;
                                for v in 0..MB {
                                    let x = (bj * MB + v) as i64;
                                    sad += (fnx.v[y as usize * fnx.w + x as usize] - sample(y + dy, x + dx)).abs();
                                }
                            }
                            sad += bias * (dy.abs() + dx.abs()) as f32;
                            if sad < best {
                                best = sad;
                                w = (dy as i32, dx as i32);
                            }
                        }
                    }
                    w
                })
                .collect()
        })
        .collect();
    for (i, row) in rows.iter().enumerate() {
        for (j, &(dy, dx)) in row.iter().enumerate() {
            let k = (b0 + i) * bx + c0 + j;
            vy[k] = dy * s as i32;
            vx[k] = dx * s as i32;
        }
    }
    Motion { bx, by, vy, vx, bs: MB * s }
}

/// A cheap fingerprint of a frame's content (a grid of 64 x 64 sampled pixels plus its size):
/// names a frame for the caches without trusting the host's memory addresses, which get reused.
pub fn fingerprint(img: &View) -> u64 {
    let n = 64usize;
    let mut key = Vec::with_capacity(n * n + 2);
    key.push(img.w as u64);
    key.push(img.h as u64);
    for i in 0..n {
        let y = (i * img.h) / n;
        let r = img.row(y);
        let mut acc = 0u64;
        for j in 0..n {
            let x = (j * img.w) / n;
            for c in 0..4 {
                acc = acc.rotate_left(7) ^ r[4 * x + c].to_bits() as u64;
            }
        }
        key.push(acc);
    }
    hash(&key)
}

/// Small LRU caches of features and motion fields, shared by every frame an engine renders.
#[derive(Default)]
pub struct Caches {
    feats: Vec<((u64, usize), Arc<Feat>)>,
    motion: Vec<((u64, u64, usize), Arc<Motion>)>,
}

const FEATS_MAX: usize = 64;
const MOTION_MAX: usize = 96;

fn lru_get<K: PartialEq + Copy, V: Clone>(v: &mut Vec<(K, V)>, k: &K) -> Option<V> {
    let i = v.iter().position(|(kk, _)| kk == k)?;
    let e = v.remove(i);
    let out = e.1.clone();
    v.push(e);
    Some(out)
}

fn lru_put<K, V>(v: &mut Vec<(K, V)>, k: K, val: V, max: usize) {
    v.push((k, val));
    if v.len() > max {
        v.remove(0);
    }
}

/// Motion from `prev` (one frame older) to `nxt`, cached by both frames' content.
pub fn motion_cached(cache: &Mutex<Caches>, prev: &View, nxt: &View, s: usize) -> Arc<Motion> {
    let (kp, kn) = (fingerprint(prev), fingerprint(nxt));
    if let Some(m) = lru_get(&mut cache.lock().unwrap().motion, &(kp, kn, s)) {
        return m;
    }
    let feat = |img: &View, k: u64| {
        if let Some(f) = lru_get(&mut cache.lock().unwrap().feats, &(k, s)) {
            return f;
        }
        let f = Arc::new(feature(img, s));
        lru_put(&mut cache.lock().unwrap().feats, (k, s), f.clone(), FEATS_MAX);
        f
    };
    let (fp, fnx) = (feat(prev, kp), feat(nxt, kn));
    let m = Arc::new(block_motion(&fp, &fnx, s));
    lru_put(&mut cache.lock().unwrap().motion, (kp, kn, s), m.clone(), MOTION_MAX);
    m
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::image::Frame;

    fn square(w: usize, h: usize, x0: usize, y0: usize, size: usize) -> Frame {
        let mut f = Frame::new(w, h);
        for y in y0..y0 + size {
            for x in x0..x0 + size {
                let i = 4 * (y * w + x);
                // a gradient, so blocks inside the square have something to match
                let v = 0.2 + 0.6 * ((x - x0) as f32 / size as f32);
                f.data[i..i + 4].copy_from_slice(&[v, v * 0.5, 0.3, 1.0]);
            }
        }
        f
    }

    #[test]
    fn finds_a_known_shift() {
        // the square moves 2 low-res px right: content of nxt at x sat in prev at x - 2
        let s = 2;
        let mut prev = square(256, 128, 80, 40, 48);
        let mut nxt = square(256, 128, 80 + 2 * s, 40, 48);
        let m = block_motion(&feature(&prev.view(), s), &feature(&nxt.view(), s), s);
        let k = (40 / s / MB + 1) * m.bx + (90 / s / MB + 1);      // a block inside the square
        assert_eq!((m.vy[k], m.vx[k]), (0, -(2 * s as i32)));
        // empty areas stay (0, 0)
        assert_eq!((m.vy[0], m.vx[0]), (0, 0));
        assert_eq!(m.bs, MB * s);
    }

    #[test]
    fn fingerprint_tracks_content_not_address() {
        let mut a = square(200, 100, 20, 20, 30);
        let mut b = a.clone();
        assert_eq!(fingerprint(&a.view()), fingerprint(&b.view()));
        let (y, x) = ((13 * 100) / 64, (10 * 200) / 64);           // a pixel on the sampling grid
        b.data[4 * (y * 200 + x)] = 0.9;
        assert_ne!(fingerprint(&a.view()), fingerprint(&b.view()));
    }
}
