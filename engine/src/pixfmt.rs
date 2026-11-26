//! Host pixel formats. The engine works in float32. Hosts don't always hand it float32:
//! Resolve's Fusion page gives a node 16-bit float frames during playback and 32-bit float
//! ones when scrubbing (seen in the node's debug log). Others are converted in and out here.
//! The half-float conversions are exact bit manipulation (no libm), like the rest of the engine.

use crate::image::{Frame, View};
use rayon::prelude::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PixFmt {
    F32 = 0,
    F16 = 1,
    U16 = 2, // 0..65535 = 0..1
    U8 = 3,  // 0..255 = 0..1
}

impl PixFmt {
    pub fn from_i32(v: i32) -> Option<PixFmt> {
        match v {
            0 => Some(PixFmt::F32),
            1 => Some(PixFmt::F16),
            2 => Some(PixFmt::U16),
            3 => Some(PixFmt::U8),
            _ => None,
        }
    }
    pub fn bytes(self) -> usize {
        match self {
            PixFmt::F32 => 4,
            PixFmt::F16 | PixFmt::U16 => 2,
            PixFmt::U8 => 1,
        }
    }
}

/// IEEE binary16 -> f32, exact (every half value is representable as f32).
#[inline]
pub fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h & 0x8000) as u32) << 16;
    let exp = ((h >> 10) & 0x1F) as u32;
    let man = (h & 0x3FF) as u32;
    let bits = if exp == 0 {
        if man == 0 {
            sign                                                // +-0
        } else {
            // subnormal: normalise the mantissa
            let mut e = 113u32;                               // 127 - 15 + 1
            let mut m = man;
            while m & 0x400 == 0 {
                m <<= 1;
                e -= 1;
            }
            sign | (e << 23) | ((m & 0x3FF) << 13)
        }
    } else if exp == 31 {
        sign | 0x7F80_0000 | (man << 13)                      // inf / NaN
    } else {
        sign | ((exp + 112) << 23) | (man << 13)
    };
    f32::from_bits(bits)
}

/// f32 -> IEEE binary16, rounding to nearest even; too large -> inf, NaN stays NaN.
#[inline]
pub fn f32_to_f16(f: f32) -> u16 {
    let x = f.to_bits();
    let sign = ((x >> 16) & 0x8000) as u16;
    let exp = ((x >> 23) & 0xFF) as i32;
    let man = x & 0x7F_FFFF;
    if exp == 255 {
        return sign | 0x7C00 | if man != 0 { 0x200 } else { 0 };
    }
    let e = exp - 127 + 15;
    if e >= 31 {
        return sign | 0x7C00;                                 // overflow -> inf
    }
    if e <= 0 {
        if e < -10 {
            return sign;                                      // underflow -> +-0
        }
        // subnormal half: shift the full mantissa (with its hidden 1) into place, round
        let m = man | 0x80_0000;
        let shift = (14 - e) as u32;
        let half = 1u32 << (shift - 1);
        let rest = m & ((1u32 << shift) - 1);
        let mut v = m >> shift;
        if rest > half || (rest == half && (v & 1) == 1) {
            v += 1;
        }
        return sign | v as u16;
    }
    let mut v = ((e as u32) << 10) | (man >> 13);
    let rest = man & 0x1FFF;
    if rest > 0x1000 || (rest == 0x1000 && (v & 1) == 1) {
        v += 1;                                               // may carry into the exponent: correct
    }
    sign | v as u16
}

/// A host image in any format as an engine view: float32 in place, others converted into
/// `keep` (which the view then borrows).
///
/// # Safety
/// `ptr` addresses `h` rows of `4 * w` elements of `fmt`, `stride` elements apart.
pub unsafe fn input_view(ptr: *mut u8, w: usize, h: usize, stride: isize, fmt: PixFmt, keep: &mut Option<Frame>) -> Option<View> {
    if ptr.is_null() || w == 0 || h == 0 || stride.unsigned_abs() < 4 * w {
        return None;
    }
    if fmt == PixFmt::F32 {
        return unsafe { View::from_raw(ptr as *mut f32, w, h, stride) };
    }
    let mut f = Frame::new(w, h);
    let base = ptr as usize;
    let bytes = fmt.bytes() as isize;
    f.data.par_chunks_mut(4 * w).enumerate().for_each(|(y, row)| {
        let p = (base as isize + y as isize * stride * bytes) as usize;
        match fmt {
            PixFmt::F16 => {
                let src = unsafe { std::slice::from_raw_parts(p as *const u16, 4 * w) };
                for (o, &v) in row.iter_mut().zip(src) {
                    *o = f16_to_f32(v);
                }
            }
            PixFmt::U16 => {
                let src = unsafe { std::slice::from_raw_parts(p as *const u16, 4 * w) };
                for (o, &v) in row.iter_mut().zip(src) {
                    *o = v as f32 * (1.0 / 65535.0);
                }
            }
            PixFmt::U8 => {
                let src = unsafe { std::slice::from_raw_parts(p as *const u8, 4 * w) };
                for (o, &v) in row.iter_mut().zip(src) {
                    *o = v as f32 * (1.0 / 255.0);
                }
            }
            PixFmt::F32 => unreachable!(),
        }
    });
    *keep = Some(f);
    Some(keep.as_mut().unwrap().view())
}

/// Write the engine's float32 `src` into a host image of format `fmt`.
///
/// # Safety
/// As `input_view`, and the memory must be writable.
pub unsafe fn write_out(src: &View, ptr: *mut u8, stride: isize, fmt: PixFmt) {
    let (w, h) = (src.w, src.h);
    let base = ptr as usize;
    let bytes = fmt.bytes() as isize;
    (0..h).into_par_iter().for_each(|y| {
        let p = (base as isize + y as isize * stride * bytes) as usize;
        let s = src.row(y);
        match fmt {
            PixFmt::F32 => unsafe { std::slice::from_raw_parts_mut(p as *mut f32, 4 * w) }.copy_from_slice(s),
            PixFmt::F16 => {
                let d = unsafe { std::slice::from_raw_parts_mut(p as *mut u16, 4 * w) };
                for (o, &v) in d.iter_mut().zip(s) {
                    *o = f32_to_f16(v);
                }
            }
            PixFmt::U16 => {
                let d = unsafe { std::slice::from_raw_parts_mut(p as *mut u16, 4 * w) };
                for (o, &v) in d.iter_mut().zip(s) {
                    *o = (v.clamp(0.0, 1.0) * 65535.0 + 0.5) as u16;
                }
            }
            PixFmt::U8 => {
                let d = unsafe { std::slice::from_raw_parts_mut(p as *mut u8, 4 * w) };
                for (o, &v) in d.iter_mut().zip(s) {
                    *o = (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
                }
            }
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_half_round_trips() {
        for h in 0..=u16::MAX {
            let f = f16_to_f32(h);
            let back = f32_to_f16(f);
            if f.is_nan() {
                assert!(f16_to_f32(back).is_nan(), "{h:#x}");
            } else {
                assert_eq!(back, h, "{h:#x} -> {f} -> {back:#x}");
            }
        }
    }

    #[test]
    fn known_values_and_rounding() {
        assert_eq!(f16_to_f32(0x3C00), 1.0);
        assert_eq!(f16_to_f32(0xC000), -2.0);
        assert_eq!(f16_to_f32(0x7BFF), 65504.0);
        assert_eq!(f16_to_f32(0x0001), 5.960_464_5e-8);          // smallest subnormal
        assert_eq!(f32_to_f16(1.0), 0x3C00);
        assert_eq!(f32_to_f16(65520.0), 0x7C00);                 // rounds up past the largest: inf
        assert_eq!(f32_to_f16(1.0 + 1.0 / 2048.0), 0x3C00);      // halfway: to even (down)
        assert_eq!(f32_to_f16(1.0 + 3.0 / 2048.0), 0x3C02);      // halfway: to even (up)
        assert_eq!(f32_to_f16(1e-9), 0);
        // nearest: every float between two halves goes to the closer one
        for i in 0..2000 {
            let f = i as f32 * 0.000_731 + 0.25;
            let h = f32_to_f16(f);
            let (a, b) = (f16_to_f32(h), f16_to_f32(h + 1));
            let c = if h > 0 { f16_to_f32(h - 1) } else { a };
            assert!((f - a).abs() <= (f - b).abs() && (f - a).abs() <= (f - c).abs(), "{f}");
        }
    }

    #[test]
    fn convert_in_and_out() {
        let (w, h) = (3usize, 2usize);
        let vals: Vec<f32> = (0..4 * w * h).map(|i| i as f32 / 23.0).collect();
        let mut halfs: Vec<u16> = vals.iter().map(|&v| f32_to_f16(v)).collect();
        let mut keep = None;
        // bottom-up, like Fusion: start at the last row, negative stride (in elements)
        let top = unsafe { (halfs.as_mut_ptr() as *mut u8).add(2 * 4 * w * (h - 1)) };
        let v = unsafe { input_view(top, w, h, -(4 * w as isize), PixFmt::F16, &mut keep) }.unwrap();
        assert!((v.px(0, 1)[0] - vals[0]).abs() < 1e-3);         // memory row 0 = bottom
        let mut out = vec![0u16; 4 * w * h];
        let otop = unsafe { (out.as_mut_ptr() as *mut u8).add(2 * 4 * w * (h - 1)) };
        unsafe { write_out(&v, otop, -(4 * w as isize), PixFmt::F16) };
        assert_eq!(out, halfs);
        let mut b8 = vec![0u8; 4 * w * h];
        unsafe { write_out(&v, b8.as_mut_ptr().add(4 * w * (h - 1)), -(4 * w as isize), PixFmt::U8) };
        assert_eq!(b8[4], (vals[4] * 255.0 + 0.5) as u8);
    }
}
