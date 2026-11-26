//! The first spike (GlitcherProbe.fuse uses it): one echo + grain on raw pointers, and a row copy.

use rayon::prelude::*;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::time::Instant;

/// Integer hash of (seed, frame, x, y) -> u32. Grain from a hash instead of a random stream:
/// every pixel's value is known on its own, so rows run in parallel and only the area that's
/// needed is computed; and it is pure integer maths, so every OS and CPU gives the same grain.
#[inline]
fn hash4(seed: u32, frame: u32, x: u32, y: u32) -> u32 {
    let mut h = seed ^ frame.wrapping_mul(0x9E37_79B9);
    h ^= x.wrapping_mul(0x85EB_CA6B);
    h = h.rotate_left(13).wrapping_mul(0xC2B2_AE35);
    h ^= y.wrapping_mul(0x27D4_EB2F);
    h ^= h >> 16;
    h = h.wrapping_mul(0x7FEB_352D);
    h ^= h >> 15;
    h = h.wrapping_mul(0x846C_A68B);
    h ^ (h >> 16)
}

/// Roughly Gaussian (mean 0, sd ~1) from one hash: the sum of four 8-bit uniforms. No libm
/// (ln, cos) - those differ between platforms - so the result is identical everywhere.
#[inline]
fn grain(h: u32) -> f32 {
    let s = (h & 255) + ((h >> 8) & 255) + ((h >> 16) & 255) + (h >> 24);
    (s as f32 - 510.0) * (1.0 / 147.8)
}

#[allow(clippy::too_many_arguments)]
fn spike(cur: &[f32], prev: Option<&[f32]>, out: &mut [f32], w: usize, h: usize, stride: usize,
         frame: u32, seed: u32, strength: f32, dx: isize) {
    let ghost_a = 0.6 * strength.clamp(0.0, 1.0);
    let sigma = 6.0 / 255.0 * strength;                    // grain, on a 0..1 scale
    out.par_chunks_mut(stride).take(h).enumerate().for_each(|(y, row_out)| {
        let row = &cur[y * stride..y * stride + 4 * w];
        for x in 0..w {
            let i = 4 * x;
            let (fr, fg, fb, fa) = (row[i], row[i + 1], row[i + 2], row[i + 3]);
            // echo: the earlier frame, shifted sideways and tinted cyan, *under* the current one
            let (mut r, mut g, mut b, mut a) = (fr, fg, fb, fa);
            if let Some(p) = prev {
                let sx = x as isize - dx;
                if sx >= 0 && (sx as usize) < w {
                    let j = y * stride + 4 * sx as usize;
                    let ga = p[j + 3] * ghost_a;
                    let k = 1.0 - fa;                          // premultiplied "under"
                    r += p[j] * 0.15 * ghost_a * k;
                    g += p[j + 1] * ghost_a * k;
                    b += p[j + 2] * ghost_a * k;
                    a += ga * k;
                }
            }
            if a > 0.0 && sigma > 0.0 {                      // grain where something is visible
                let n = grain(hash4(seed, frame, x as u32, y as u32)) * sigma * a;
                r = (r + n).max(0.0);
                g = (g + n).max(0.0);
                b = (b + n).max(0.0);
            }
            row_out[i] = r;
            row_out[i + 1] = g;
            row_out[i + 2] = b;
            row_out[i + 3] = a;
        }
    });
}

/// The spike effect. `prev` may be NULL (no earlier frame). Returns the milliseconds it took,
/// or a negative number on bad arguments (-1) or an internal error (-2): never crashes the host.
///
/// # Safety
/// `cur` and `prev` (if not NULL) must point to `stride * height` floats, `out` to as many
/// writable floats, and must not overlap `out`.
#[unsafe(no_mangle)]
#[allow(clippy::too_many_arguments)]                  // a C signature: one flat argument list
pub unsafe extern "C" fn gc_spike(cur: *const f32, prev: *const f32, out: *mut f32, width: i32,
                                  height: i32, stride: i64, frame: i32, seed: u32, strength: f32,
                                  dx: i32) -> f64 {
    if cur.is_null() || out.is_null() || width <= 0 || height <= 0 || stride < 4 * width as i64 {
        return -1.0;
    }
    let (w, h, st) = (width as usize, height as usize, stride as usize);
    let t0 = Instant::now();
    let r = catch_unwind(AssertUnwindSafe(|| {
        // SAFETY: the caller guarantees the sizes (see above)
        let cur = unsafe { std::slice::from_raw_parts(cur, st * h) };
        let prev = (!prev.is_null()).then(|| unsafe { std::slice::from_raw_parts(prev, st * h) });
        let out = unsafe { std::slice::from_raw_parts_mut(out, st * h) };
        spike(cur, prev, out, w, h, st, frame as u32, seed, strength, dx as isize);
    }));
    match r {
        Ok(()) => t0.elapsed().as_secs_f64() * 1e3,
        Err(_) => -2.0,
    }
}

/// Copy `rows` rows of `row_floats` floats between two buffers with their own strides - what the
/// Fuse needs to move Fusion's image rows in and out of contiguous buffers, at memcpy speed and
/// in parallel. Returns milliseconds, or -1 on bad arguments.
///
/// # Safety
/// Both buffers must hold `rows` rows at their strides; they must not overlap.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gc_copy_rows(src: *const f32, src_stride: i64, dst: *mut f32, dst_stride: i64,
                                      rows: i32, row_floats: i64) -> f64 {
    if src.is_null() || dst.is_null() || rows < 0 || row_floats < 0
        || src_stride < row_floats || dst_stride < row_floats {
        return -1.0;
    }
    let t0 = Instant::now();
    let (n, ss, ds, rf) = (rows as usize, src_stride as usize, dst_stride as usize, row_floats as usize);
    // SAFETY: as documented
    let s = unsafe { std::slice::from_raw_parts(src, ss * n.saturating_sub(1) + rf) };
    let d = unsafe { std::slice::from_raw_parts_mut(dst, ds * n.saturating_sub(1) + rf) };
    let d_rows: Vec<&mut [f32]> = d.chunks_mut(ds).take(n).collect();
    d_rows.into_par_iter().enumerate().for_each(|(y, row)| {
        row[..rf].copy_from_slice(&s[y * ss..y * ss + rf]);
    });
    t0.elapsed().as_secs_f64() * 1e3
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grain_is_centred_and_unit_ish() {
        let n = 200_000u32;
        let (mut s, mut s2) = (0f64, 0f64);
        for i in 0..n {
            let g = grain(hash4(7, 3, i % 997, i / 997)) as f64;
            s += g;
            s2 += g * g;
        }
        let mean = s / n as f64;
        let sd = (s2 / n as f64 - mean * mean).sqrt();
        assert!(mean.abs() < 0.02, "mean {mean}");
        assert!((sd - 1.0).abs() < 0.05, "sd {sd}");
    }

    #[test]
    fn spike_is_deterministic_and_respects_alpha() {
        let (w, h) = (64usize, 32usize);
        let st = 4 * w + 8;                                  // padded rows, like a host may use
        let mut cur = vec![0f32; st * h];
        for y in 0..h {
            for x in 20..40 {
                let i = y * st + 4 * x;
                cur[i..i + 4].copy_from_slice(&[0.4, 0.3, 0.2, 1.0]);
            }
        }
        let prev = cur.clone();
        let (mut a, mut b) = (vec![0f32; st * h], vec![0f32; st * h]);
        let t = unsafe { gc_spike(cur.as_ptr(), prev.as_ptr(), a.as_mut_ptr(), w as i32, h as i32, st as i64, 5, 1, 1.0, 6) };
        unsafe { gc_spike(cur.as_ptr(), prev.as_ptr(), b.as_mut_ptr(), w as i32, h as i32, st as i64, 5, 1, 1.0, 6) };
        assert!(t >= 0.0);
        assert_eq!(a, b);                                     // same inputs, same bytes
        let px = |buf: &Vec<f32>, x: usize, y: usize| { let i = y * st + 4 * x; [buf[i], buf[i + 1], buf[i + 2], buf[i + 3]] };
        assert_eq!(px(&a, 5, 3), [0.0; 4]);                   // empty stays empty (no grain on nothing)
        assert!(px(&a, 42, 3)[3] > 0.0);                     // the shifted echo appears right of the body
        assert_eq!(px(&a, 30, 3)[3], 1.0);                   // opaque body stays opaque
        let bad = unsafe { gc_spike(cur.as_ptr(), prev.as_ptr(), a.as_mut_ptr(), w as i32, h as i32, 3, 5, 1, 1.0, 6) };
        assert_eq!(bad, -1.0);                               // a too-small stride is refused
    }

    #[test]
    fn copy_rows_handles_strides() {
        let src: Vec<f32> = (0..30).map(|v| v as f32).collect();     // 3 rows of stride 10
        let mut dst = vec![-1f32; 3 * 8];                              // stride 8
        let t = unsafe { gc_copy_rows(src.as_ptr(), 10, dst.as_mut_ptr(), 8, 3, 6) };
        assert!(t >= 0.0);
        assert_eq!(&dst[8..14], &src[10..16]);
        assert_eq!(dst[14], -1.0);                                     // beyond the row: untouched
    }
}
