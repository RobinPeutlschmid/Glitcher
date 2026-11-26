//! Images the engine works on: float32 RGBA, **premultiplied**, nominally 0..1 (above 1 is HDR
//! and kept), rows top to bottom. A `View` borrows the host's memory, which can store its rows
//! bottom-up (Fusion does): then the view starts at the top row and steps back through memory
//! (a negative stride). `Frame` owns its pixels.

/// A window onto host memory: `ptr` is the top row, `stride` the distance between rows in
/// floats (negative when rows are stored bottom-up).
#[derive(Clone, Copy, Debug)]
pub struct View {
    ptr: *mut f32,
    pub w: usize,
    pub h: usize,
    stride: isize,
}

// SAFETY: a View is only a pointer and a shape. Effects write to disjoint rows from different
// threads and never resize; the host keeps the memory alive for the length of a call.
unsafe impl Send for View {}
unsafe impl Sync for View {}

impl View {
    /// # Safety
    /// `ptr` must address `h` rows of `4 * w` floats, `stride` floats apart, valid (and, if
    /// written to, writable and not otherwise borrowed) for as long as the view is used.
    pub unsafe fn from_raw(ptr: *mut f32, w: usize, h: usize, stride: isize) -> Option<View> {
        if ptr.is_null() || w == 0 || h == 0 || (stride.unsigned_abs() < 4 * w) {
            return None;
        }
        Some(View { ptr, w, h, stride })
    }

    #[inline]
    pub fn row(&self, y: usize) -> &[f32] {
        debug_assert!(y < self.h);
        // SAFETY: from_raw's contract
        unsafe { std::slice::from_raw_parts(self.ptr.offset(y as isize * self.stride), 4 * self.w) }
    }

    /// # Safety
    /// No other reference to row `y` may be alive (rows are handed to one thread each).
    #[inline]
    #[allow(clippy::mut_from_ref)]
    pub unsafe fn row_mut(&self, y: usize) -> &mut [f32] {
        debug_assert!(y < self.h);
        unsafe { std::slice::from_raw_parts_mut(self.ptr.offset(y as isize * self.stride), 4 * self.w) }
    }

    /// Columns x0..x1 of row y, mutable (4 floats per pixel). Unlike `row_mut` this borrows only
    /// those pixels, so other threads may read the rest of the row (with `px_raw`) meanwhile.
    ///
    /// # Safety
    /// No other reference to these pixels may be alive.
    #[inline]
    #[allow(clippy::mut_from_ref)]
    pub unsafe fn span_mut(&self, y: usize, x0: usize, x1: usize) -> &mut [f32] {
        debug_assert!(y < self.h && x0 <= x1 && x1 <= self.w);
        unsafe { std::slice::from_raw_parts_mut(self.ptr.offset(y as isize * self.stride).add(4 * x0), 4 * (x1 - x0)) }
    }

    /// Pixel (x, y) read through the raw pointer, without borrowing the row: safe to use while
    /// other threads write other pixels of it through `span_mut`.
    #[inline]
    pub fn px_raw(&self, x: i64, y: i64) -> [f32; 4] {
        if x < 0 || y < 0 || x >= self.w as i64 || y >= self.h as i64 {
            return [0.0; 4];
        }
        // SAFETY: in bounds (from_raw's contract); nobody writes this pixel during the read
        unsafe { std::ptr::read(self.ptr.offset(y as isize * self.stride).add(4 * x as usize) as *const [f32; 4]) }
    }

    /// Pixel (x, y), or transparent black outside the image.
    #[inline]
    pub fn px(&self, x: i64, y: i64) -> [f32; 4] {
        if x < 0 || y < 0 || x >= self.w as i64 || y >= self.h as i64 {
            return [0.0; 4];
        }
        let r = self.row(y as usize);
        let i = 4 * x as usize;
        [r[i], r[i + 1], r[i + 2], r[i + 3]]
    }

    pub fn same_shape(&self, o: &View) -> bool {
        self.w == o.w && self.h == o.h
    }

    pub fn ptr(&self) -> *const f32 {
        self.ptr
    }
}

/// Pixels the engine owns.
#[derive(Clone)]
pub struct Frame {
    pub w: usize,
    pub h: usize,
    pub data: Vec<f32>,
}

impl Frame {
    pub fn new(w: usize, h: usize) -> Frame {
        Frame { w, h, data: vec![0.0; 4 * w * h] }
    }

    pub fn from_view(v: &View) -> Frame {
        let mut f = Frame::new(v.w, v.h);
        for y in 0..v.h {
            f.data[4 * v.w * y..4 * v.w * (y + 1)].copy_from_slice(v.row(y));
        }
        f
    }

    /// A view for reading only.
    pub fn view_const(&self) -> View {
        // the caller only reads through it
        View { ptr: self.data.as_ptr() as *mut f32, w: self.w, h: self.h, stride: 4 * self.w as isize }
    }

    pub fn view(&mut self) -> View {
        View { ptr: self.data.as_mut_ptr(), w: self.w, h: self.h, stride: 4 * self.w as isize }
    }
}

/// A pixel area (y0, y1, x0, x1), half-open.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Rect {
    pub y0: usize,
    pub y1: usize,
    pub x0: usize,
    pub x1: usize,
}

impl Rect {
    pub fn new(y0: usize, y1: usize, x0: usize, x1: usize) -> Rect {
        Rect { y0, y1: y1.max(y0), x0, x1: x1.max(x0) }
    }
    pub fn is_empty(&self) -> bool {
        self.y1 <= self.y0 || self.x1 <= self.x0
    }
    pub fn union(self, o: Rect) -> Rect {
        if o.is_empty() {
            return self;
        }
        if self.is_empty() {
            return o;
        }
        Rect { y0: self.y0.min(o.y0), y1: self.y1.max(o.y1), x0: self.x0.min(o.x0), x1: self.x1.max(o.x1) }
    }
    pub fn intersect(self, o: Rect) -> Rect {
        Rect::new(self.y0.max(o.y0), self.y1.min(o.y1), self.x0.max(o.x0), self.x1.min(o.x1))
    }
    pub fn grow(self, m: usize, w: usize, h: usize) -> Rect {
        if self.is_empty() {
            return self;
        }
        Rect { y0: self.y0.saturating_sub(m), y1: (self.y1 + m).min(h), x0: self.x0.saturating_sub(m), x1: (self.x1 + m).min(w) }
    }
}

/// `src` into `dst` (same size), rows in parallel.
pub fn copy(src: &View, dst: &View) {
    use rayon::prelude::*;
    (0..src.h).into_par_iter().for_each(|y| {
        // SAFETY: one thread per row
        unsafe { dst.row_mut(y) }.copy_from_slice(src.row(y));
    });
}

/// Both boxes in one pass: (pixels with alpha above `thr` as (x0, y0, x1, y1), pixels with any
/// channel non-zero as a Rect).
pub fn boxes(v: &View, thr: f32) -> (Option<(i64, i64, i64, i64)>, Rect) {
    use rayon::prelude::*;
    type Span = Option<(usize, usize)>;
    let rows: Vec<(Span, Span)> = (0..v.h)
        .into_par_iter()
        .map(|y| {
            let r = v.row(y);
            let px_on = |x: usize| r[4 * x] != 0.0 || r[4 * x + 1] != 0.0 || r[4 * x + 2] != 0.0 || r[4 * x + 3] != 0.0;
            let al = |x: usize| r[4 * x + 3] > thr;
            let a = (0..v.w).find(|&x| al(x)).map(|f| (f, (0..v.w).rev().find(|&x| al(x)).unwrap()));
            let c = (0..v.w).find(|&x| px_on(x)).map(|f| (f, (0..v.w).rev().find(|&x| px_on(x)).unwrap()));
            (a, c)
        })
        .collect();
    let mut lb: Option<(i64, i64, i64, i64)> = None;
    let mut cb = Rect::new(0, 0, 0, 0);
    for (y, (a, c)) in rows.iter().enumerate() {
        if let Some((x0, x1)) = a {
            let (x0, x1, yy) = (*x0 as i64, *x1 as i64, y as i64);
            lb = Some(match lb {
                None => (x0, yy, x1 + 1, yy + 1),
                Some((p, q, r2, _)) => (p.min(x0), q, r2.max(x1 + 1), yy + 1),
            });
        }
        if let Some((x0, x1)) = c {
            cb = cb.union(Rect::new(y, y + 1, *x0, x1 + 1));
        }
    }
    (lb, cb)
}

/// Bounding box (x0, y0, x1, y1) of the pixels with alpha above `thr`, or None.
pub fn alpha_box(v: &View, thr: f32) -> Option<(i64, i64, i64, i64)> {
    use rayon::prelude::*;
    let rows: Vec<Option<(usize, usize)>> = (0..v.h)
        .into_par_iter()
        .map(|y| {
            let r = v.row(y);
            let first = (0..v.w).find(|&x| r[4 * x + 3] > thr)?;
            let last = (0..v.w).rev().find(|&x| r[4 * x + 3] > thr)?;
            Some((first, last))
        })
        .collect();
    let mut b: Option<(i64, i64, i64, i64)> = None;
    for (y, r) in rows.iter().enumerate() {
        if let Some((x0, x1)) = r {
            let (x0, x1, y) = (*x0 as i64, *x1 as i64, y as i64);
            b = Some(match b {
                None => (x0, y, x1 + 1, y + 1),
                Some((a, c, d, _)) => (a.min(x0), c, d.max(x1 + 1), y + 1),
            });
        }
    }
    b
}

/// `v` resized to w x h (bilinear, edges held): a plate of another size.
pub fn resample(v: &View, w: usize, h: usize) -> Frame {
    use rayon::prelude::*;
    let mut f = Frame::new(w, h);
    let (kx, ky) = (v.w as f64 / w as f64, v.h as f64 / h as f64);
    f.data.par_chunks_mut(4 * w).enumerate().for_each(|(y, row)| {
        let sy = ((y as f64 + 0.5) * ky - 0.5).clamp(0.0, (v.h - 1) as f64);
        let (j, fy) = (sy as usize, (sy - sy.floor()) as f32);
        let j1 = (j + 1).min(v.h - 1);
        for x in 0..w {
            let sx = ((x as f64 + 0.5) * kx - 0.5).clamp(0.0, (v.w - 1) as f64);
            let (i, fx) = (sx as usize, (sx - sx.floor()) as f32);
            let i1 = (i + 1).min(v.w - 1);
            let (a, b, c, d) = (v.px(i as i64, j as i64), v.px(i1 as i64, j as i64), v.px(i as i64, j1 as i64), v.px(i1 as i64, j1 as i64));
            for ch in 0..4 {
                let top = a[ch] * (1.0 - fx) + b[ch] * fx;
                let bot = c[ch] * (1.0 - fx) + d[ch] * fx;
                row[4 * x + ch] = top * (1.0 - fy) + bot * fy;
            }
        }
    });
    f
}

/// `top` over `bottom`, premultiplied: top + bottom * (1 - top alpha).
#[inline]
pub fn over(top: [f32; 4], bottom: [f32; 4]) -> [f32; 4] {
    let k = 1.0 - top[3].clamp(0.0, 1.0);
    [top[0] + bottom[0] * k, top[1] + bottom[1] * k, top[2] + bottom[2] * k, top[3] + bottom[3] * k]
}

#[inline]
pub fn scale(p: [f32; 4], f: f32) -> [f32; 4] {
    [p[0] * f, p[1] * f, p[2] * f, p[3] * f]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bottom_up_view_reads_top_row_first() {
        // memory holds the bottom row first, like Fusion; the view must still see y=0 at the top
        let (w, h) = (3usize, 2usize);
        let mut mem = vec![0f32; 4 * w * h];
        mem[0] = 1.0;                                          // bottom-left pixel's red
        let top = unsafe { mem.as_mut_ptr().add(4 * w) };      // top row starts one row in
        let v = unsafe { View::from_raw(top, w, h, -(4 * w as isize)) }.unwrap();
        assert_eq!(v.px(0, 1)[0], 1.0);                        // bottom row = y 1
        assert_eq!(v.px(0, 0)[0], 0.0);
        assert_eq!(v.px(-1, 0), [0.0; 4]);
        assert!(unsafe { View::from_raw(top, w, h, 4) }.is_none());   // stride shorter than a row
    }

    #[test]
    fn box_and_over() {
        let mut f = Frame::new(10, 8);
        for (x, y) in [(2, 3), (6, 5)] {
            f.data[4 * (y * 10 + x) + 3] = 1.0;
        }
        assert_eq!(alpha_box(&f.view(), 0.03), Some((2, 3, 7, 6)));
        assert_eq!(alpha_box(&Frame::new(4, 4).view(), 0.0), None);
        assert_eq!(over([0.5, 0.0, 0.0, 0.5], [0.0, 1.0, 0.0, 1.0]), [0.5, 0.5, 0.0, 1.0]);
    }
}
