//! The frame effects - jpeg, blocks, slices, wave, sort, smear, split, tint, scanlines, noise -
//! ported from glitch.py, with its 2-D placement and its 3-D plane placement (ev.pl; the
//! front / back passes and the roll are run by `layers`).
//!
//! glitch.py works on straight alpha, 0..255; the engine on premultiplied 0..1. Effects that
//! change colour (jpeg, invert/posterize blocks, tint, noise, split) do it on the straight
//! colour (premultiplied / alpha) and premultiply the result, so they look the same; effects
//! that only move pixels move premultiplied ones.
//!
//! Each effect reads the picture as it was before that effect (glitch.py: `out = a.copy()`,
//! read `a`, write `out`). That "before" is a `Snap` of only the area that holds content: the
//! engine tracks that area (`Rect`) as effects write, so it never scans or copies a whole 4K
//! frame to find it.

/// A block through the codec: (block row, block column, its 64 straight RGB pixels, colour cap).
type Coded = (usize, usize, [[f32; 3]; 64], f32);
/// A slice: (y, rows, dx, column window or None for the whole width).
type SlicePlan = (usize, usize, i64, Option<(usize, usize)>);

use crate::detmath::{cos, env, gauss, hash, round_even, sin, u, vnoise};
use crate::image::{Rect, View, over};
use crate::sched::{BLOCKS, Ctx, JPEG, NOISE, SCANLINES, SLICES, SMEAR, SORT, SPLIT, TINT, WAVE};
use rayon::prelude::*;
use std::f64::consts::{PI, TAU};
use std::sync::{Arc, OnceLock};

/// Where the picture has content, and what the effects wrote.
///
/// `cbox`: outside it the picture is all zero (transparent), so effects never look further.
/// `dirty`: every area an effect wrote, for the passes that need to know exactly (3D front /
/// back layering, per-effect body-part masks). `before`: the picture as it was before this
/// effect, kept copy-on-write: effects `touch` an area just before writing it, which saves the
/// tiles there; everything else is still the live picture. Passes that start from the same
/// state (3D front / back) share one.
pub struct Area {
    pub cbox: Rect,
    pub dirty: Vec<Rect>,
    pub before: Option<Arc<Before>>,
    /// A second store that keeps what every touched area was like before a whole sequence of
    /// effects (the rolled 3D pass compares against it to find what changed).
    pub base: Option<Arc<Before>>,
    /// Per-pixel effects (split, tint, scanlines, noise) may skip pixels outside this: on a
    /// plate, beyond where the spill mask reaches the clean plate is put back anyway.
    pub limit: Option<Rect>,
    shared: bool,
}

impl Area {
    pub fn new(cbox: Rect) -> Self {
        Area { cbox, dirty: Vec::new(), before: None, base: None, limit: None, shared: false }
    }
    /// An area for a pass inside another one's sequence: same content box, the given store,
    /// the outer one's base store and limit.
    pub fn inner(&self, before: Arc<Before>) -> Self {
        Area { cbox: self.cbox, dirty: Vec::new(), before: Some(before), base: self.base.clone(), limit: self.limit,
               shared: false }
    }
    fn painted(&self, r: Rect) -> Rect {
        self.limit.map_or(r, |l| r.intersect(l))
    }
    /// Note an area as written without keeping its old pixels.
    pub fn mark(&mut self, r: Rect) {
        if !r.is_empty() {
            self.cbox = self.cbox.union(r);
            self.dirty.push(r);
        }
    }
    /// About to write `r`: keep its old pixels, then note it as written.
    pub fn touch(&mut self, r: Rect) {
        if let Some(b) = &self.before {
            b.touch(r);
        }
        if let Some(b) = &self.base {
            b.touch(r);
        }
        self.mark(r);
    }
    /// About to write `r` without reading the store: keep the old pixels only if another
    /// pass needs them.
    fn touch_if_shared(&mut self, r: Rect) {
        if self.shared {
            self.touch(r);
        } else {
            if let Some(b) = &self.base {
                b.touch(r);
            }
            self.mark(r);
        }
    }
    fn store(&self) -> Arc<Before> {
        self.before.clone().expect("run_one sets the store")
    }
}

const BT: usize = 64;

/// The picture before an effect, copy-on-write in 64 x 64 tiles (see Area).
pub struct Before {
    v: View,
    tw: usize,
    th: usize,
    tiles: Vec<OnceLock<Box<[f32]>>>,
}

impl Before {
    pub fn new(v: &View) -> Before {
        let (tw, th) = (v.w.div_ceil(BT), v.h.div_ceil(BT));
        Before { v: *v, tw, th, tiles: (0..tw * th).map(|_| OnceLock::new()).collect() }
    }

    /// Save the (not yet saved) tiles under `r` from the live picture.
    pub fn touch(&self, r: Rect) {
        let r = Rect::new(r.y0.min(self.v.h), r.y1.min(self.v.h), r.x0.min(self.v.w), r.x1.min(self.v.w));
        if r.is_empty() {
            return;
        }
        let idx: Vec<usize> = (r.y0 / BT..r.y1.div_ceil(BT))
            .flat_map(|ty| (r.x0 / BT..r.x1.div_ceil(BT)).map(move |tx| ty * self.tw + tx))
            .filter(|&i| self.tiles[i].get().is_none())
            .collect();
        idx.par_iter().for_each(|&i| {
            let (ty, tx) = (i / self.tw, i % self.tw);
            let (x0, x1) = (tx * BT, ((tx + 1) * BT).min(self.v.w));
            let mut d = Vec::with_capacity(4 * BT * BT);
            for y in ty * BT..((ty + 1) * BT).min(self.v.h) {
                for x in x0..x1 {
                    d.extend_from_slice(&self.v.px_raw(x as i64, y as i64));
                }
            }
            let _ = self.tiles[i].set(d.into_boxed_slice());
        });
    }

    #[inline]
    pub fn px(&self, x: i64, y: i64) -> [f32; 4] {
        if x < 0 || y < 0 || x >= self.v.w as i64 || y >= self.v.h as i64 {
            return [0.0; 4];
        }
        let (xu, yu) = (x as usize, y as usize);
        let (tx, ty) = (xu / BT, yu / BT);
        match self.tiles[ty * self.tw + tx].get() {
            Some(d) => {
                let tw = (((tx + 1) * BT).min(self.v.w)) - tx * BT;
                let k = 4 * ((yu - ty * BT) * tw + (xu - tx * BT));
                [d[k], d[k + 1], d[k + 2], d[k + 3]]
            }
            None => self.v.px_raw(x, y),
        }
    }
    #[inline]
    pub fn alpha(&self, x: i64, y: i64) -> f32 {
        self.px(x, y)[3]
    }
    pub fn tiles_wide(&self) -> (usize, usize) {
        (self.tw, self.th)
    }
}

/// The picture before an effect, inside `r` (outside: transparent, as it really is there).
#[derive(Clone)]
pub struct Snap {
    r: Rect,
    w: usize,
    data: Vec<f32>,
}

impl Snap {
    pub fn take(v: &View, r: Rect) -> Snap {
        let w = r.x1 - r.x0;
        let mut data = vec![0f32; 4 * w * (r.y1 - r.y0)];
        if w > 0 {
            data.par_chunks_mut(4 * w).enumerate().for_each(|(i, row)| {
                row.copy_from_slice(&v.row(r.y0 + i)[4 * r.x0..4 * r.x1]);
            });
        }
        Snap { r, w, data }
    }
    #[inline]
    pub fn px(&self, x: i64, y: i64) -> [f32; 4] {
        let r = &self.r;
        if x < r.x0 as i64 || y < r.y0 as i64 || x >= r.x1 as i64 || y >= r.y1 as i64 {
            return [0.0; 4];
        }
        let i = 4 * ((y as usize - r.y0) * self.w + (x as usize - r.x0));
        [self.data[i], self.data[i + 1], self.data[i + 2], self.data[i + 3]]
    }
    #[inline]
    pub fn alpha(&self, x: i64, y: i64) -> f32 {
        self.px(x, y)[3]
    }
    pub fn rect(&self) -> Rect {
        self.r
    }
    /// Row y of the snapshot, if inside it.
    #[inline]
    pub fn row_at(&self, y: i64) -> Option<&[f32]> {
        if y < self.r.y0 as i64 || y >= self.r.y1 as i64 {
            return None;
        }
        let i = 4 * (y as usize - self.r.y0) * self.w;
        Some(&self.data[i..i + 4 * self.w])
    }
    /// Pixel x of a row from row_at (transparent outside).
    #[inline]
    pub fn in_row(&self, row: Option<&[f32]>, x: i64) -> [f32; 4] {
        match row {
            Some(r) if x >= self.r.x0 as i64 && x < self.r.x1 as i64 => {
                let i = 4 * (x as usize - self.r.x0);
                [r[i], r[i + 1], r[i + 2], r[i + 3]]
            }
            _ => [0.0; 4],
        }
    }
}

#[inline]
fn straight(p: [f32; 4]) -> [f32; 3] {
    if p[3] > 0.0 { [p[0] / p[3], p[1] / p[3], p[2] / p[3]] } else { [0.0; 3] }
}

#[inline]
fn lum3(c: [f32; 3]) -> f32 {
    c[0] * 0.299 + c[1] * 0.587 + c[2] * 0.114
}

/// The pixels x0..x1 of one row, addressed by their x.
pub struct Span<'a> {
    s: &'a mut [f32],
    x0: usize,
}

impl Span<'_> {
    #[inline]
    pub fn get(&self, x: usize) -> [f32; 4] {
        let i = 4 * (x - self.x0);
        [self.s[i], self.s[i + 1], self.s[i + 2], self.s[i + 3]]
    }
    #[inline]
    pub fn put(&mut self, x: usize, p: [f32; 4]) {
        let i = 4 * (x - self.x0);
        self.s[i..i + 4].copy_from_slice(&p);
    }
}

/// Run `f` on every row y of `r` (rows in parallel) with that row's pixels of `r` as a Span.
fn spans(out: &View, r: Rect, f: impl Fn(usize, &mut Span) + Sync) {
    if r.is_empty() {
        return;
    }
    (r.y0..r.y1).into_par_iter().for_each(|y| {
        // SAFETY: one thread per row, and only r's pixels of it
        let mut sp = Span { s: unsafe { out.span_mut(y, r.x0, r.x1) }, x0: r.x0 };
        f(y, &mut sp);
    });
}

fn bbox_wh(c: &Ctx) -> (f64, f64) {
    ((c.bbox.2 - c.bbox.0) as f64, (c.bbox.3 - c.bbox.1) as f64)
}

fn clipf(v: f64, lo: f64, hi: f64) -> f64 {
    v.max(lo).min(hi)
}

fn amp(c: &Ctx) -> f64 {
    0.4 + 0.6 * c.p.intensity
}

// ---------------------------------------------------------------------------------- jpeg
struct Dct {
    c: [[f32; 8]; 8],
}

fn dct_mat() -> &'static Dct {
    static D: OnceLock<Dct> = OnceLock::new();
    D.get_or_init(|| {
        let mut c = [[0f32; 8]; 8];
        for (k, row) in c.iter_mut().enumerate() {
            for (n, v) in row.iter_mut().enumerate() {
                let mut x = (2.0f64 / 8.0).sqrt() * cos((2 * n + 1) as f64 * k as f64 * PI / 16.0);
                if k == 0 {
                    x /= 2f64.sqrt();
                }
                *v = x as f32;
            }
        }
        Dct { c }
    })
}

const QY: [f32; 64] = [16., 11., 10., 16., 24., 40., 51., 61., 12., 12., 14., 19., 26., 58., 60., 55., 14., 13., 16., 24.,
    40., 57., 69., 56., 14., 17., 22., 29., 51., 87., 80., 62., 18., 22., 37., 56., 68., 109., 103., 77., 24., 35., 55.,
    64., 81., 104., 113., 92., 49., 64., 78., 87., 103., 121., 120., 101., 72., 92., 95., 98., 112., 100., 103., 99.];

fn qtabs(q: i32) -> [[f32; 64]; 3] {
    let q = q.clamp(1, 100) as f64;
    let s = if q < 50.0 { 5000.0 / q } else { 200.0 - 2.0 * q };
    let f = |t: f32| (((t as f64 * s + 50.0) / 100.0).floor()).clamp(1.0, 255.0) as f32;
    let mut qc = [99f32; 64];
    let top = [[17., 18., 24., 47.], [18., 21., 26., 66.], [24., 26., 56., 99.], [47., 66., 99., 99.]];
    for (i, r) in top.iter().enumerate() {
        for (j, v) in r.iter().enumerate() {
            qc[i * 8 + j] = *v;
        }
    }
    let mut out = [[0f32; 64]; 3];
    for i in 0..64 {
        out[0][i] = f(QY[i]);
        out[1][i] = f(qc[i]);
        out[2][i] = f(qc[i]);
    }
    out
}

/// One 8x8 block through the codec: straight RGB 0..255 in, 0..255 out.
fn codec_block(src: &[[f32; 3]; 64], kind: usize, dc: [f32; 3], ac: Option<(&[u64; 4], usize, usize, f32)>, q: &[[f32; 64]; 3]) -> [[f32; 3]; 64] {
    let d = &dct_mat().c;
    let mut out = [[0f32; 3]; 64];
    if kind == 0 {                                             // decoder ran dry: neutral grey
        for o in out.iter_mut() {
            *o = [128.0; 3];
        }
        return out;
    }
    for ch in 0..3 {
        // colour transform for this channel
        let mut b = [0f32; 64];
        for (i, s) in src.iter().enumerate() {
            let (r, g, bl) = (s[0], s[1], s[2]);
            b[i] = match ch {
                0 => 0.299 * r + 0.587 * g + 0.114 * bl - 128.0,
                1 => -0.168736 * r - 0.331264 * g + 0.5 * bl,
                _ => 0.5 * r - 0.418688 * g - 0.081312 * bl,
            };
        }
        // co = C b C^T
        let mut t = [0f32; 64];
        for k in 0..8 {
            for n in 0..8 {
                let mut acc = 0f32;
                for m in 0..8 {
                    acc += d[k][m] * b[m * 8 + n];
                }
                t[k * 8 + n] = acc;
            }
        }
        let mut co = [0f32; 64];
        for k in 0..8 {
            for l in 0..8 {
                let mut acc = 0f32;
                for n in 0..8 {
                    acc += t[k * 8 + n] * d[l][n];
                }
                co[k * 8 + l] = acc;
            }
        }
        if kind == 1 {
            co[0] += 8.0 * dc[ch];                             // DC drift: the run changes colour
        } else if let Some((key, by, bx, e)) = ac {           // AC garbage, fixed per block position
            for uu in 0..8 {
                for vv in 0..8 {
                    let g = gauss(hash(&[key[0], key[1], key[2], key[3], 0x4AC, (by % 8) as u64, (bx % 8) as u64,
                                        ch as u64, uu as u64, vv as u64]));
                    co[uu * 8 + vv] += g * (1.0 / (1 + uu + vv) as f32) * 160.0 * e;
                }
            }
        }
        for i in 0..64 {
            co[i] = (co[i] / q[ch][i]).round() * q[ch][i];
        }
        // back: C^T co C
        let mut t2 = [0f32; 64];
        for m in 0..8 {
            for l in 0..8 {
                let mut acc = 0f32;
                for k in 0..8 {
                    acc += d[k][m] * co[k * 8 + l];
                }
                t2[m * 8 + l] = acc;
            }
        }
        for m in 0..8 {
            for n in 0..8 {
                let mut acc = 0f32;
                for l in 0..8 {
                    acc += t2[m * 8 + l] * d[l][n];
                }
                b[m * 8 + n] = acc;
            }
        }
        for i in 0..64 {
            out[i][ch] = b[i];
        }
    }
    for o in out.iter_mut() {
        let (y, cb, cr) = (o[0] + 128.0, o[1], o[2]);
        *o = [y + 1.402 * cr, y - 0.344136 * cb - 0.714136 * cr, y + 1.772 * cb];
    }
    out
}

/// Square dilation of "visible" (alpha > 0) by r px, inside area `a`: what JPEG junk may spill
/// into. Two passes of running counts, O(pixels) whatever r is.
fn dilate(snap: &Before, a: Rect, r: usize) -> Vec<bool> {
    let (w, h) = (a.x1 - a.x0, a.y1 - a.y0);
    let mut horiz = vec![false; w * h];
    horiz.par_chunks_mut(w.max(1)).enumerate().for_each(|(i, row)| {
        let y = (a.y0 + i) as i64;
        let mut pre = vec![0u32; w + 1];
        for x in 0..w {
            pre[x + 1] = pre[x] + (snap.alpha((a.x0 + x) as i64, y) > 0.0) as u32;
        }
        #[allow(clippy::needless_range_loop)]
        for x in 0..w {
            let (lo, hi) = (x.saturating_sub(r), (x + r + 1).min(w));
            row[x] = pre[hi] > pre[lo];
        }
    });
    let mut out = vec![false; w * h];
    let cols: Vec<Vec<bool>> = (0..w)
        .into_par_iter()
        .map(|x| {
            let mut pre = vec![0u32; h + 1];
            for y in 0..h {
                pre[y + 1] = pre[y] + horiz[y * w + x] as u32;
            }
            (0..h).map(|y| pre[(y + r + 1).min(h)] > pre[y.saturating_sub(r)]).collect()
        })
        .collect();
    for (x, col) in cols.iter().enumerate() {
        for (y, &v) in col.iter().enumerate() {
            out[y * w + x] = v;
        }
    }
    out
}

pub fn jpeg(c: &Ctx, out: &View, area: &mut Area) -> usize {
    let evs = c.events(JPEG);
    let cbox = area.cbox;
    if evs.is_empty() || cbox.is_empty() {
        return 0;
    }
    let (w, h) = (out.w, out.h);
    let (gh, gw) = (h / 8, w / 8);
    if gh < 2 || gw < 2 {
        return 0;
    }
    let (x0, y0, _, _) = c.bbox;
    let (bw, bh) = bbox_wh(c);
    let bleed = if c.p.bleed < 0 { (w as f64 * 0.03) as usize } else { c.p.bleed as usize };
    // the runs, in glitch.py's terms: a run of block indices, row-major (it wraps across rows);
    // a plane's run is limited to the block columns of its footprint
    struct Run { i0: usize, i1: usize, kind: usize, rep: usize, dc: [f32; 3], key: [u64; 4], e: f32,
                 xwin: Option<(i64, i64)> }
    let mut runs = Vec::new();
    for ev in &evs {
        let e = env(ev.u) * ev.g;
        let p = &ev.p;
        let (i0, rows, xwin) = match &ev.pl {
            None => {
                let i0 = (clipf(((y0 as f64 + p[1] * bh) / 8.0).floor(), 0.0, (gh - 1) as f64) as usize) * gw
                    + clipf(((x0 as f64 + p[0] * bw) / 8.0).floor(), 0.0, (gw - 1) as f64) as usize;
                (i0, 1.0 + p[2] * 0.15 * bh / 8.0, None)
            }
            Some(pl) => {
                let xw = (((pl.sx - pl.xr).max(0.0) / 8.0).floor() as i64, ((pl.sx + pl.xr) / 8.0).floor() as i64);
                let i0 = (clipf((pl.sy / 8.0).floor(), 0.0, (gh - 1) as f64) as usize) * gw
                    + clipf(xw.0 as f64, 0.0, (gw - 1) as f64) as usize;
                (i0, 1.0 + p[2] * 0.15 * bh / 8.0 * pl.s * pl.vy, Some(xw))
            }
        };
        let i1 = (i0 + (rows * gw as f64 * e) as usize).min(gh * gw);
        if i1 <= i0 {
            continue;
        }
        let kind = (p[3] * 4.0) as usize;
        let k = (e * 1.5).min(1.0);
        let dc = [((p[4] - 0.5) * 140.0 * k) as f32, ((p[5] - 0.5) * 320.0 * k) as f32, ((p[6] - 0.5) * 320.0 * k) as f32];
        runs.push(Run { i0, i1, kind, rep: 1 + (p[4] * 5.0) as usize, dc, key: ev.key, e: e as f32, xwin });
    }
    if runs.is_empty() {
        return 0;
    }
    let snap = area.store();
    // JPEG has no alpha: transparent areas take the mean visible colour (less ringing); every
    // 4th pixel each way is plenty for a mean
    let (sum, n) = (cbox.y0..cbox.y1)
        .into_par_iter()
        .step_by(4)
        .map(|y| {
            let (mut s, mut n) = ([0f64; 3], 0f64);
            for x in (cbox.x0..cbox.x1).step_by(4) {
                let p = snap.px(x as i64, y as i64);
                if p[3] > 0.0 {
                    let st = straight(p);
                    for ch in 0..3 {
                        s[ch] += st[ch] as f64 * 255.0;
                    }
                    n += 1.0;
                }
            }
            (s, n)
        })
        .reduce(|| ([0f64; 3], 0f64), |a, b| ([a.0[0] + b.0[0], a.0[1] + b.0[1], a.0[2] + b.0[2]], a.1 + b.1));
    if n == 0.0 {
        return 0;
    }
    let mean = [(sum[0] / n) as f32, (sum[1] / n) as f32, (sum[2] / n) as f32];
    let q = qtabs(if c.p.quality > 0 { c.p.quality } else { (75.0 - 55.0 * c.p.intensity) as i32 });
    let spill_area = cbox.grow(bleed, w, h);
    let block_src = |by: usize, bx: usize| {
        let mut s = [[0f32; 3]; 64];
        let mut cap = 1f32;
        for v in 0..8 {
            for uu in 0..8 {
                let p = snap.px((bx * 8 + uu) as i64, (by * 8 + v) as i64);
                let px = if p[3] > 0.0 { let st = straight(p); [st[0] * 255.0, st[1] * 255.0, st[2] * 255.0] } else { mean };
                cap = cap.max(px[0].max(px[1]).max(px[2]) / 255.0);
                s[v * 8 + uu] = px;
            }
        }
        (s, cap)
    };
    let mut done = 0;
    for run in &runs {
        // every block of the run, through the codec (blocks are independent: in parallel)
        let blocks: Vec<Coded> = (run.i0..run.i1)
            .into_par_iter()
            .filter(|idx| run.xwin.is_none_or(|(a, b)| (a..=b).contains(&((idx % gw) as i64))))
            .map(|idx| {
                let (by, bx) = (idx / gw, idx % gw);
                let sidx = if run.kind == 3 { run.i0 + (idx - run.i0) % run.rep } else { idx };
                let (src, cap) = block_src(sidx / gw, sidx % gw);
                let ac = if run.kind == 2 { Some((&run.key, by, bx, run.e)) } else { None };
                let kind = if run.kind == 3 { 4 } else { run.kind };        // 3 = repeat: codec as plain
                (by, bx, codec_block(&src, kind, run.dc, ac, &q), cap)
            })
            .collect();
        if blocks.is_empty() {
            continue;
        }
        let (r0, r1) = (blocks[0].0, blocks[blocks.len() - 1].0);
        let (c0, c1) = blocks.iter().fold((gw, 0), |(a, b), k| (a.min(k.1), b.max(k.1)));
        let wr = Rect::new(r0 * 8, (r1 + 1) * 8, c0 * 8, (c1 + 1) * 8);
        // where junk may spill past the silhouette: the visible pixels grown by `bleed`, only
        // around this run (before it writes)
        let da = wr.intersect(spill_area);
        let dil = if bleed > 0 && !da.is_empty() {
            let g = da.grow(bleed, w, h);
            Some((g, dilate(&snap, g, bleed)))
        } else {
            None
        };
        area.touch(wr);
        // write: inside the silhouette the colour changes (alpha kept); in the bleed zone
        // outside it, junk that differs enough from the fill becomes opaque
        let by_row: Vec<Vec<&Coded>> = (r0..=r1)
            .map(|r| blocks.iter().filter(|b| b.0 == r).collect())
            .collect();
        spans(out, wr, |y, row| {
            let br = &by_row[y / 8 - r0];
            for (_, bx, res, cap) in br.iter() {
                for uu in 0..8 {
                    let x = bx * 8 + uu;
                    let i = (y % 8) * 8 + uu;
                    let mut rgb = res[i];
                    for v in rgb.iter_mut() {
                        *v = (*v / 255.0).clamp(0.0, *cap);
                    }
                    let a = snap.alpha(x as i64, y as i64);
                    if a > 0.0 {
                        row.put(x, [rgb[0] * a, rgb[1] * a, rgb[2] * a, a]);
                    } else if let Some((g, d)) = &dil {
                        let inside = y >= da.y0 && y < da.y1 && x >= da.x0 && x < da.x1;
                        if inside && d[(y - g.y0) * (g.x1 - g.x0) + (x - g.x0)] {
                            let diff: f32 = (0..3).map(|ch| (rgb[ch] * 255.0 - mean[ch]).abs()).sum();
                            if run.kind == 0 || diff > 90.0 {
                                row.put(x, [rgb[0], rgb[1], rgb[2], 1.0]);
                            }
                        }
                    }
                }
            }
        });
        done += 1;
    }
    done
}

// -------------------------------------------------------------------------------- blocks
pub fn blocks(c: &Ctx, out: &View, area: &mut Area) -> usize {
    let evs: Vec<_> = c.events(BLOCKS).into_iter().filter(|e| e.g >= 0.3).collect();
    if evs.is_empty() {
        return 0;
    }
    let (w, h) = (out.w, out.h);
    let (x0, y0, _, _) = c.bbox;
    let (bw, bh) = bbox_wh(c);
    let am = amp(c);
    let snap = area.store();
    let mut n = 0;
    for ev in &evs {
        let p = &ev.p;
        let j = 0.6 + 0.4 * c.held(ev, 0, 3);
        let (bwid, bhei, sx, sy, dx, dy);
        match &ev.pl {
            None => {
                bwid = ((bw * (0.04 + 0.20 * p[2])) as usize).max(2).min(w);
                bhei = ((bh * (0.01 + 0.07 * p[3])) as usize).max(2).min(h);
                sx = clipf(x0 as f64 + p[0] * (bw - bwid as f64).max(1.0), 0.0, (w - bwid) as f64) as i64;
                sy = clipf(y0 as f64 + p[1] * (bh - bhei as f64).max(1.0), 0.0, (h - bhei) as f64) as i64;
                dx = clipf(sx as f64 + (p[4] - 0.5) * 5.0 * bwid as f64 * am * j, 0.0, (w - bwid) as f64) as usize;
                dy = clipf(sy as f64 + (p[5] - 0.5) * 5.0 * bhei as f64 * am * j, 0.0, (h - bhei) as f64) as usize;
            }
            Some(pl) => {
                // the block's size and its offset live on the plane, then are projected
                let (wl, hl) = (bw * (0.04 + 0.20 * p[2]), bh * (0.01 + 0.07 * p[3]));
                bwid = ((wl * pl.ax.abs() * pl.s) as usize).max(2).min(w);
                bhei = ((hl * pl.s * pl.vy) as usize).max(2).min(h);
                sx = clipf(pl.sx - bwid as f64 / 2.0, 0.0, (w - bwid) as f64) as i64;
                sy = clipf(pl.sy - bhei as f64 / 2.0, 0.0, (h - bhei) as f64) as i64;
                let uu = (p[4] - 0.5) * 5.0 * wl * am * j;
                let vv = (p[5] - 0.5) * 5.0 * hl * am * j;
                dx = clipf(sx as f64 + pl.s * uu * (pl.ax + pl.rho * pl.nx), 0.0, (w - bwid) as f64) as usize;
                dy = clipf(sy as f64 + pl.s * vv * pl.vy, 0.0, (h - bhei) as f64) as usize;
            }
        }
        let mode = (p[6] * 4.0) as usize;
        area.touch(Rect::new(dy, dy + bhei, dx, dx + bwid));
        spans(out, Rect::new(dy, dy + bhei, dx, dx + bwid), |y, row| {
            let syy = sy + (y - dy) as i64;
            for x in dx..dx + bwid {
                let mut b = snap.px(sx + (x - dx) as i64, syy);
                match mode {
                    1 => b = [b[1], b[2], b[0], b[3]],                        // channels rotated
                    2 => b = [(b[3] - b[0]).max(0.0), (b[3] - b[1]).max(0.0), (b[3] - b[2]).max(0.0), b[3]],  // inverted
                    3 => {                                                    // posterised, 4 levels
                        let s = straight(b);
                        let q = |v: f32| ((v * 255.0 / 64.0).floor() * 64.0 / 255.0) * b[3];
                        b = [q(s[0]), q(s[1]), q(s[2]), b[3]];
                    }
                    _ => {}
                }
                if mode == 0 || b[3] > 0.0 {
                    row.put(x, b);
                }
            }
        });
        n += 1;
    }
    n
}

// -------------------------------------------------------------------------------- slices
pub fn slices(c: &Ctx, out: &View, area: &mut Area) -> usize {
    let evs = c.events(SLICES);
    if evs.is_empty() {
        return 0;
    }
    let (w, h) = (out.w, out.h);
    let (_, y0, _, _) = c.bbox;
    let (bw, bh) = bbox_wh(c);
    let am = amp(c);
    // (y, rows, dx, window): a whole-width row shift, or (3D) a window of the plane's width cut
    // out and pasted dx further
    let mut plans: Vec<SlicePlan> = Vec::new();
    for ev in &evs {
        let p = &ev.p;
        let sign = if p[3] < 0.5 { -1.0 } else { 1.0 };
        let u = sign * bw * (0.02 + 0.14 * p[2]) * am * env(ev.u) * ev.g * (0.35 + 0.65 * c.held(ev, 0, 3));
        match &ev.pl {
            None => {
                let sh = ((bh * (0.003 + 0.04 * p[1] * p[1])) as usize).max(1);
                let y = clipf(y0 as f64 + p[0] * (bh - sh as f64).max(1.0), 0.0, (h - 1) as f64) as usize;
                let dx = u as i64;                                      // int(): toward zero
                if dx != 0 {
                    plans.push((y, sh.min(h - y), dx, None));
                }
            }
            Some(pl) => {
                let sh = ((bh * (0.003 + 0.04 * p[1] * p[1]) * pl.s * pl.vy) as usize).max(1);
                let y = clipf(pl.sy - sh as f64 / 2.0, 0.0, (h - 1) as f64) as usize;
                let dx = (pl.s * u * (pl.ax + pl.rho * pl.nx)) as i64;  // projected 3-D displacement
                let xa = clipf(pl.sx - pl.xr, 0.0, w as f64) as usize;
                let xb = clipf(pl.sx + pl.xr, 0.0, w as f64) as usize;
                if dx != 0 && xb >= xa + 2 {
                    plans.push((y, sh.min(h - y), dx, Some((xa, xb))));
                }
            }
        }
    }
    if plans.is_empty() {
        return 0;
    }
    let snap = area.store();
    for &(y, sh, dx, win) in &plans {
        match win {
            None => {
                let cb = area.cbox;
                // outside the content area (shifted) the row is and stays empty
                let r = Rect::new(y, y + sh, cb.x0.saturating_sub(dx.unsigned_abs() as usize),
                                  (cb.x1 + dx.unsigned_abs() as usize).min(w));
                area.touch(r);
                spans(out, r, |yy, row| {
                    for x in r.x0..r.x1 {
                        row.put(x, snap.px(x as i64 - dx, yy as i64));
                    }
                });
            }
            Some((xa, xb)) => {
                let s0 = (-(xa as i64 + dx)).max(0);
                let s1 = (xb - xa) as i64 - (xb as i64 + dx - w as i64).max(0);
                let lo = (xa as i64 + (dx + s0).min(0)).max(0) as usize;
                let hi = ((xb as i64).max(xa as i64 + dx + s1)).min(w as i64) as usize;
                let r = Rect::new(y, y + sh, lo, hi);
                area.touch(r);
                spans(out, r, |yy, row| {
                    for x in xa..xb {
                        row.put(x, [0.0; 4]);
                    }
                    for s in s0..s1.max(s0) {
                        let src = snap.px(xa as i64 + s, yy as i64);
                        if src[3] > 0.0 {
                            row.put((xa as i64 + dx + s) as usize, src);
                        }
                    }
                });
            }
        }
    }
    plans.len()
}

// ---------------------------------------------------------------------------------- wave
pub fn wave(c: &Ctx, out: &View, area: &mut Area) -> usize {
    let evs = c.events(WAVE);
    let (w, h) = (out.w, out.h);
    let (_, y0, _, _) = c.bbox;
    let (bw, bh) = bbox_wh(c);
    let am = amp(c);
    let mut n = 0;
    for ev in &evs {
        let p = &ev.p;
        let (hb, y) = match &ev.pl {
            None => {
                let hb = ((bh * (0.08 + 0.25 * p[1])) as usize).max(8);
                (hb, clipf(y0 as f64 + p[0] * (bh - hb as f64).max(1.0), 0.0, (h - 1) as f64) as usize)
            }
            Some(pl) => {
                if pl.xr < 2.0 {
                    continue;
                }
                let hb = ((bh * (0.08 + 0.25 * p[1]) * pl.s * pl.vy) as usize).max(8);
                (hb, clipf(pl.sy - hb as f64 / 2.0, 0.0, (h - 1) as f64) as usize)
            }
        };
        let hb = hb.min(h - y);
        if hb < 4 {
            continue;
        }
        let mut a = bw * (0.01 + 0.045 * p[2]) * env(ev.u) * ev.g * am;
        if let Some(pl) = &ev.pl {
            a *= pl.s * (pl.ax + pl.rho * pl.nx);
        }
        let period = hb as f64 * (0.2 + 0.6 * p[3]);
        let speed = (0.4 + 2.0 * p[5]) * if p[6] < 0.5 { 1.0 } else { -1.0 };   // cycles per second
        let phase = TAU * (p[4] + speed * c.t);                                  // scrolls in time
        let rowd: Vec<f64> = (0..hb)
            .map(|ys| {
                let win = sin(PI * ys as f64 / (hb as f64 - 1.0).max(1.0));        // fades at the band's edges
                a * sin(TAU * ys as f64 / period + phase) * win
            })
            .collect();
        let maxd = rowd.iter().map(|d| d.abs().ceil() as usize).max().unwrap_or(0);
        if maxd == 0 && rowd.iter().all(|d| round_even(*d) == 0.0) {
            continue;
        }
        // columns that can change: around the content, and (3D) inside the plane, which fades
        // to nothing at its sides
        let cb = area.cbox;
        let (mut xa, mut xb) = (cb.x0.saturating_sub(maxd), (cb.x1 + maxd).min(w));
        let pl = ev.pl;
        if let Some(pl) = &pl {
            xa = xa.max(clipf((pl.sx - pl.xr).floor(), 0.0, w as f64) as usize);
            xb = xb.min(clipf((pl.sx + pl.xr).ceil() + 1.0, 0.0, w as f64) as usize);
        }
        if xb <= xa {
            continue;
        }
        let wx = |x: usize| -> f64 {
            match &pl {
                None => 1.0,
                Some(pl) => (1.0 - ((x as f64 - pl.sx) / pl.xr).powi(2)).clamp(0.0, 1.0),
            }
        };
        // rows read the current state (glitch.py reads `out` here): copy the band first
        let band = Snap::take(out, Rect::new(y, y + hb, cb.x0.min(xa), cb.x1.max(xb).min(w)));
        area.touch(Rect::new(y, y + hb, xa, xb));
        spans(out, Rect::new(y, y + hb, xa, xb), |yy, row| {
            let dr = rowd[yy - y];
            #[allow(clippy::needless_range_loop)]
            for x in xa..xb {
                let d = if pl.is_none() { round_even(dr) } else { round_even(dr * wx(x)) } as i64;
                row.put(x, band.px((x as i64 - d).clamp(0, w as i64 - 1), yy as i64));
            }
        });
        n += 1;
    }
    n
}

// ---------------------------------------------------------------------------------- sort
pub fn sort(c: &Ctx, out: &View, area: &mut Area) -> usize {
    let evs = c.events(SORT);
    let (w, h) = (out.w, out.h);
    let (x0, y0, _, _) = c.bbox;
    let (bw, bh) = bbox_wh(c);
    let mut n = 0;
    for ev in &evs {
        let p = &ev.p;
        let (bwid, xa, ya, l);
        match &ev.pl {
            None => {
                bwid = ((bw * (0.01 + 0.07 * p[1])) as usize).max(1).min(w);
                xa = clipf(x0 as f64 + p[0] * (bw - bwid as f64).max(1.0), 0.0, (w - bwid) as f64) as usize;
                ya = clipf(y0 as f64 + p[2] * 0.8 * bh, 0.0, (h - 1) as f64) as usize;
                l = (bh * (0.1 + 0.4 * p[3]) * env(ev.u) * ev.g) as usize;      // the sorted run grows/retracts
            }
            Some(pl) => {
                bwid = ((bw * (0.01 + 0.07 * p[1]) * pl.ax.abs() * pl.s) as usize).max(1).min(w);
                xa = clipf(pl.sx - bwid as f64 / 2.0, 0.0, (w - bwid) as f64) as usize;
                l = (bh * (0.1 + 0.4 * p[3]) * env(ev.u) * ev.g * pl.s * pl.vy) as usize;
                ya = clipf(pl.sy - l as f64 / 2.0, 0.0, (h - 1) as f64) as usize;
            }
        }
        let yb = (ya + l).min(h);
        if yb < ya + 2 {
            continue;
        }
        let bright_first = p[4] < 0.5;
        // each column on its own: visible pixels sorted by brightness, the silhouette unchanged
        let cols: Vec<Vec<(usize, [f32; 4])>> = (xa..xa + bwid)
            .into_par_iter()
            .map(|x| {
                let mut vis: Vec<(usize, [f32; 4], f32)> = (ya..yb)
                    .filter_map(|y| {
                        let px = out.px(x as i64, y as i64);
                        (px[3] > 0.0).then(|| (y, px, lum3(straight(px))))
                    })
                    .collect();
                let slots: Vec<usize> = vis.iter().map(|v| v.0).collect();
                if bright_first {
                    vis.sort_by(|a, b| b.2.total_cmp(&a.2));            // stable
                } else {
                    vis.sort_by(|a, b| a.2.total_cmp(&b.2));
                }
                slots.into_iter().zip(vis.into_iter().map(|v| v.1)).collect()
            })
            .collect();
        area.touch(Rect::new(ya, yb, xa, xa + bwid));
        for (i, col) in cols.iter().enumerate() {
            for &(y, px) in col {
                // SAFETY: sequential here; one pixel at a time
                unsafe { out.span_mut(y, xa + i, xa + i + 1) }.copy_from_slice(&px);
            }
        }
        n += 1;
    }
    n
}

// --------------------------------------------------------------------------------- smear
pub fn smear(c: &Ctx, out: &View, area: &mut Area) -> usize {
    let evs = c.events(SMEAR);
    let cbox = area.cbox;
    if evs.is_empty() || cbox.is_empty() {
        return 0;
    }
    let (w, h) = (out.w, out.h);
    let (x0, _, _, _) = c.bbox;
    let (bw, bh) = bbox_wh(c);
    let am = amp(c);
    let snap = area.store();
    let mut n = 0;
    for ev in &evs {
        let p = &ev.p;
        let (width, x, l);
        match &ev.pl {
            None => {
                width = ((bw * (0.002 + 0.012 * p[1] * p[1])) as usize).max(1).min(w);
                x = clipf(x0 as f64 + p[0] * bw, 0.0, (w - width) as f64) as usize;
                l = (bh * (0.08 + 0.5 * p[5]) * am * env(ev.u) * ev.g) as usize;
            }
            Some(pl) => {
                width = ((bw * (0.002 + 0.012 * p[1] * p[1]) * pl.s) as usize).max(1).min(w);
                x = clipf(pl.sx - width as f64 / 2.0, 0.0, (w - width) as f64) as usize;
                l = (bh * (0.08 + 0.5 * p[5]) * am * env(ev.u) * ev.g * pl.s * pl.vy) as usize;
            }
        }
        if l < 2 {
            continue;
        }
        let up = p[2] < 0.25;
        // rows where the column is mostly solid (alpha > 128/255): only the content area has any
        let rows: Vec<usize> = (cbox.y0..cbox.y1)
            .filter(|&y| (x..x + width).any(|xx| snap.alpha(xx as i64, y as i64) > 128.0 / 255.0))
            .collect();
        if rows.is_empty() {
            continue;
        }
        let ysrc = if p[3] < 0.7 {
            if up { rows[0] } else { *rows.last().unwrap() }             // the silhouette edge (tracks it)
        } else {
            rows[((p[4] * rows.len() as f64) as usize).min(rows.len() - 1)]   // a fixed row
        };
        let (ya, yb) = if up { (ysrc.saturating_sub(l), ysrc) } else { (ysrc + 1, (ysrc + 1 + l).min(h)) };
        let cnt = yb.saturating_sub(ya);
        if cnt < 1 {
            continue;
        }
        let lm1 = (l - 1).max(1) as f32;
        let fade = |i: usize| -> f32 {
            // np.linspace(1, .15, L)[:n] downward, np.linspace(.15, 1, L)[-n:] upward
            if up { 0.15 + 0.85 * (l - cnt + i) as f32 / lm1 } else { 1.0 - 0.85 * i as f32 / lm1 }
        };
        let src: Vec<[f32; 4]> = (x..x + width).map(|xx| snap.px(xx as i64, ysrc as i64)).collect();
        area.touch(Rect::new(ya, yb, x, x + width));
        spans(out, Rect::new(ya, yb, x, x + width), |y, row| {
            let f = fade(y - ya);
            for xx in x..x + width {
                let s = src[xx - x];
                let s = [s[0] * f, s[1] * f, s[2] * f, s[3] * f];
                let d = row.get(xx);
                // over visible pixels instead of cutting a see-through hole into the body
                row.put(xx, if d[3] > 0.0 { over(s, d) } else { s });
            }
        });
        n += 1;
    }
    n
}

// --------------------------------------------------------------------------------- split
pub fn split(c: &Ctx, out: &View, ar: &mut Area) -> usize {
    let cbox = ar.cbox;
    if cbox.is_empty() {
        return 0;
    }
    let w = out.w as f64;
    let ph = u(&[c.p.seed as u64, SPLIT as u64 + 1, 0, 0]) * TAU;
    let gate = (c.p.intensity * 5.0).min(1.0) * c.gain(SPLIT);
    let am = w * (0.002 + 0.02 * c.p.intensity) * (0.35 + 0.65 * c.act) * gate;   // 0 at intensity 0
    let wob = am * (0.7 + 0.3 * sin(TAU * 0.7 * c.t + ph));
    // 3D: the channels as depth layers (R in front, B behind) give a parallax as the camera turns
    let (mut pxf, mut pyf) = (0.0, 0.0);
    if let Some(sp) = &c.space {
        let k = (c.bbox.2 - c.bbox.0) as f64 * (0.015 + 0.05 * c.p.intensity);
        (pxf, pyf) = (k * sp.right[2] * gate, k * sp.up[2] * gate);    // screen image of a depth step
        if sp.roll != 0.0 {
            let (cr, sr) = (cos(sp.roll), sin(sp.roll));
            (pxf, pyf) = (pxf * cr - pyf * sr, pxf * sr + pyf * cr);
        }
    }
    let dx = round_even(wob + pxf) as i64;
    let dy = round_even(round_even(wob) * 0.25 * sin(TAU * 0.31 * c.t + ph * 1.7) + pyf) as i64;
    if dx == 0 && dy == 0 {
        return 0;
    }
    let m = dx.unsigned_abs().max(dy.unsigned_abs()) as usize + 1;
    let area = ar.painted(cbox.grow(m, out.w, out.h));
    // every pixel reads three: one dense copy of the content, read row by row
    let snap = Snap::take(out, cbox);
    ar.touch_if_shared(area);
    spans(out, area, |y, row| {
        let y = y as i64;
        let (rr, rg, rb) = (snap.row_at(y - dy), snap.row_at(y), snap.row_at(y + dy));
        for x in area.x0..area.x1 {
            let xi = x as i64;
            let (pr, pg, pb) = (snap.in_row(rr, xi - dx), snap.in_row(rg, xi), snap.in_row(rb, xi + dx));
            // R from one side, B from the other, G in place; alpha grows a fringe from both
            let a = pg[3].max(0.75 * pr[3]).max(0.75 * pb[3]);
            let (r, g, b) = (straight(pr)[0], straight(pg)[1], straight(pb)[2]);
            row.put(x, [r * a, g * a, b * a, a]);
        }
    });
    1
}

// ----------------------------------------------------------------------------- per pixel
pub fn tint(c: &Ctx, out: &View, cbox: &Rect) -> usize {
    let amount = (c.p.tint * c.gain(TINT)).min(1.0) as f32;
    if amount <= 0.0 || cbox.is_empty() {
        return 0;
    }
    let (dark, light) = ([40.0 / 255.0, 10.0 / 255.0, 90.0 / 255.0f32], [120.0 / 255.0, 1.0, 240.0 / 255.0f32]);
    spans(out, *cbox, |_, row| {
        for x in cbox.x0..cbox.x1 {
            let p = row.get(x);
            if p[3] <= 0.0 {
                continue;
            }
            let s = straight(p);
            let t = lum3(s);
            let mut o = [0f32; 4];
            for ch in 0..3 {
                let v = s[ch] * (1.0 - amount) + (dark[ch] * (1.0 - t) + light[ch] * t) * amount;
                o[ch] = v.clamp(0.0, 1.0f32.max(s[ch])) * p[3];
            }
            o[3] = p[3];
            row.put(x, o);
        }
    });
    1
}

pub fn scanlines(c: &Ctx, out: &View, cbox: &Rect) -> usize {
    let gs = c.gain(SCANLINES);
    if gs <= 0.0 || cbox.is_empty() {
        return 0;
    }
    let period = (round_even(out.h as f64 / 300.0) as usize).max(2);
    let dark = (1.0 - (0.15 + 0.35 * c.p.intensity) * gs).max(0.0) as f32;
    let flicker = (1.0 - 0.06 * c.p.intensity * gs * vnoise(c.p.seed as u64, 2, c.t * 18.0)).max(0.0) as f32;
    spans(out, *cbox, |y, row| {
        let f = if y % period == 0 { dark * flicker } else { flicker };
        for v in row.s.chunks_mut(4) {
            v[0] *= f;
            v[1] *= f;
            v[2] *= f;
        }
    });
    1
}

pub fn noise(c: &Ctx, out: &View, cbox: &Rect) -> usize {
    let gn = c.gain(NOISE);
    if gn <= 0.0 || cbox.is_empty() {
        return 0;
    }
    let sigma = ((3.0 + 10.0 * c.p.intensity) * gn / 255.0) as f32;
    let (seed, fi) = (c.p.seed as u64, c.fi as u64);
    spans(out, *cbox, |y, row| {
        for x in cbox.x0..cbox.x1 {
            let mut p = row.get(x);
            let a = p[3];
            if a <= 0.0 {
                continue;
            }
            // fresh grain per frame, one value per pixel for all three channels (as glitch.py)
            let n = gauss(hash(&[seed, NOISE as u64 + 1, 0x6E, fi, x as u64, y as u64])) * sigma;
            for v in p.iter_mut().take(3) {
                let old = *v;
                // straight + n, premultiplied; clipped to the displayable range as glitch.py's
                // 8-bit, HDR values stay HDR
                *v = (old + n * a).clamp(0.0, if old > a { f32::MAX } else { a });
            }
            row.put(x, p);
        }
    });
    1
}

/// One frame effect by id (jpeg .. noise) on `out`. Returns events / effects applied.
pub fn run_one(fx: usize, c: &Ctx, out: &View, area: &mut Area) -> usize {
    if c.gain(fx) <= 0.0 && fx != TINT {
        return 0;
    }
    // the picture before this effect: a shared store (3D passes, masks) or a fresh one
    let shared = area.before.is_some();
    if !shared {
        area.before = Some(Arc::new(Before::new(out)));
    }
    area.shared = shared;
    let n = run_fx(fx, c, out, area, shared);
    if !shared {
        area.before = None;
    }
    n
}

fn run_fx(fx: usize, c: &Ctx, out: &View, area: &mut Area, shared: bool) -> usize {
    match fx {
        JPEG => jpeg(c, out, area),
        BLOCKS => blocks(c, out, area),
        SLICES => slices(c, out, area),
        WAVE => wave(c, out, area),
        SORT => sort(c, out, area),
        SMEAR => smear(c, out, area),
        SPLIT => split(c, out, area),
        TINT | SCANLINES | NOISE => {
            // per pixel, in place: nothing to keep unless someone else needs the old picture
            let cb = area.painted(area.cbox);
            let _ = shared;
            area.touch_if_shared(cb);
            match fx {
                TINT => tint(c, out, &cb),
                SCANLINES => scanlines(c, out, &cb),
                _ => noise(c, out, &cb),
            }
        }
        _ => 0,
    }
}

/// All frame effects in glitch.py's order (2-D placement). Returns events / effects applied.
pub fn run_all(c: &Ctx, out: &View, cbox: &mut Rect) -> [usize; 10] {
    let mut n = [0usize; 10];
    let mut area = Area::new(*cbox);
    for (fx, v) in n.iter_mut().enumerate() {
        *v = run_one(fx, c, out, &mut area);
    }
    *cbox = area.cbox;
    n
}
