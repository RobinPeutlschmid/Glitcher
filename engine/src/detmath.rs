//! Deterministic numbers: hashing, uniform draws and the few smooth functions the scheduler
//! needs, built only from +, -, *, /, sqrt and floor. Those are exactly specified by IEEE 754,
//! so every OS and CPU gives the same bits. Library functions such as `exp`, `ln` or `sin` are
//! not (each platform's maths library rounds its own way), and one bit can decide whether an
//! event happens - so they are not used.

/// SplitMix64 finaliser: a strong 64-bit mixer.
#[inline]
pub fn mix64(mut z: u64) -> u64 {
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Hash of a sequence of integers (order matters, length matters).
#[inline]
pub fn hash(key: &[u64]) -> u64 {
    let mut h = 0x9E37_79B9_7F4A_7C15u64 ^ (key.len() as u64).wrapping_mul(0xD6E8_FEB8_6659_FD93);
    for &k in key {
        h = mix64(h ^ k.wrapping_mul(0x9E37_79B9_7F4A_7C15));
    }
    h
}

/// Uniform in [0, 1) from the top 53 bits of a hash.
#[inline]
pub fn unit(h: u64) -> f64 {
    (h >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
}

/// Uniform in [0, 1) for a key: the stateless "random value of this thing" (Python's `_u`).
#[inline]
pub fn u(key: &[u64]) -> f64 {
    unit(hash(key))
}

/// A sequential generator (for drawing several values from one bucket).
pub struct Stream(u64);

impl Stream {
    pub fn new(key: &[u64]) -> Self {
        Stream(hash(key))
    }
    #[inline]
    pub fn next_f64(&mut self) -> f64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        unit(mix64(self.0))
    }
    /// Poisson-distributed count with mean `lam` (Knuth's product of uniforms; lam <= ~600).
    pub fn poisson(&mut self, lam: f64) -> u32 {
        if lam <= 0.0 {
            return 0;
        }
        let l = exp(-lam);
        let (mut p, mut n) = (1.0f64, 0u32);
        loop {
            p *= self.next_f64();
            if p <= l || n > 10_000 {
                return n;
            }
            n += 1;
        }
    }
}

/// exp(x) from basic arithmetic: x = n ln2 + r with |r| <= ln2/2, e^r by its Taylor series
/// (13 terms: error far below f64 precision for |r| < 0.35), times 2^n set in the exponent bits.
pub fn exp(x: f64) -> f64 {
    if x.is_nan() {
        return x;
    }
    if x < -745.0 {
        return 0.0;
    }
    if x > 709.0 {
        return f64::INFINITY;
    }
    const LN2: f64 = std::f64::consts::LN_2;
    let n = (x / LN2 + 0.5).floor();
    let r = x - n * LN2;
    let mut term = 1.0f64;
    let mut sum = 1.0f64;
    for k in 1..=13 {
        term = term * r / k as f64;
        sum += term;
    }
    let n = n as i64;
    // 2^n by exponent bits, in two steps so subnormal results stay correct-ish
    let (n1, n2) = if n < -1000 { (-1000, n + 1000) } else { (n, 0) };
    let p1 = f64::from_bits(((n1 + 1023) as u64) << 52);
    let p2 = f64::from_bits(((n2 + 1023) as u64) << 52);
    sum * p1 * p2
}

/// Smoothstep from a (-> 0) to b (-> 1); a > b gives a falling edge (Python's `_ss`).
#[inline]
pub fn ss(x: f64, a: f64, b: f64) -> f64 {
    if a == b {
        return if x >= a { 1.0 } else { 0.0 };
    }
    let t = ((x - a) / (b - a)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Event envelope over its life u in [0, 1]: quick attack, long decay.
#[inline]
pub fn env(u: f64) -> f64 {
    ss(u, 0.0, 0.1) * ss(u, 1.0, 0.6)
}

/// Smooth 1-D value noise in [0, 1].
pub fn vnoise(seed: u64, ch: u64, x: f64) -> f64 {
    let i = x.floor();
    let f = x - i;
    let s = f * f * (3.0 - 2.0 * f);
    let i = i as i64 as u64;
    let a = u(&[seed, 900 + ch, i]);
    let b = u(&[seed, 900 + ch, i.wrapping_add(1)]);
    a + (b - a) * s
}

/// Round half away from zero, like Python's `int(round(x))` closely enough for pixel offsets
/// (Python rounds half to even; offsets of exactly .5 px are rare and harmless).
#[inline]
pub fn iround(x: f64) -> i64 {
    if x >= 0.0 { (x + 0.5).floor() as i64 } else { -((-x + 0.5).floor() as i64) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exp_matches_the_library_closely() {
        for i in -2000..=200 {
            let x = i as f64 * 0.05;
            let (a, b) = (exp(x), x.exp());
            assert!(((a - b) / b).abs() < 1e-14, "x={x}: {a} vs {b}");
        }
        assert_eq!(exp(0.0), 1.0);
    }

    #[test]
    fn uniforms_are_uniform_and_keyed() {
        let n = 100_000;
        let mean: f64 = (0..n).map(|i| u(&[1, 2, i])).sum::<f64>() / n as f64;
        assert!((mean - 0.5).abs() < 0.005);
        assert_ne!(u(&[1, 2, 3]), u(&[1, 2, 3, 0]));         // length is part of the key
        assert_ne!(u(&[1, 2, 3]), u(&[3, 2, 1]));            // so is order
        assert_eq!(u(&[7, 7]), u(&[7, 7]));
    }

    #[test]
    fn poisson_has_the_right_mean() {
        for &lam in &[0.8, 3.0, 16.0, 80.0] {
            let n = 20_000;
            let s: u64 = (0..n).map(|i| Stream::new(&[9, i]).poisson(lam) as u64).sum();
            let m = s as f64 / n as f64;
            assert!((m - lam).abs() < 0.05 * lam.max(1.0), "lam {lam}: mean {m}");
        }
    }

    #[test]
    fn smooth_functions() {
        assert_eq!(ss(0.5, 0.0, 1.0), 0.5);
        assert_eq!(ss(2.0, 1.0, 0.0), 0.0);
        assert!(env(0.05) > 0.0 && env(0.05) < 1.0 && env(0.3) == 1.0 && env(1.0) == 0.0);
        let v = vnoise(3, 0, 1.0);
        assert_eq!(v, u(&[3, 900, 1]));                       // passes through the lattice values
        assert_eq!(iround(2.5), 3);
        assert_eq!(iround(-2.5), -3);
    }
}

/// sin(x) from basic arithmetic (no libm, same bits on every platform): reduce to [-pi, pi],
/// then to [-pi/2, pi/2] by symmetry, then an odd Taylor series to x^23 (error < 1e-16).
pub fn sin(x: f64) -> f64 {
    if !x.is_finite() {
        return f64::NAN;
    }
    use std::f64::consts::{FRAC_PI_2, PI, TAU};
    let mut r = x - (x / TAU).floor() * TAU;               // [0, 2pi)
    if r > PI {
        r -= TAU;                                          // (-pi, pi]
    }
    if r > FRAC_PI_2 {
        r = PI - r;
    } else if r < -FRAC_PI_2 {
        r = -PI - r;
    }
    let r2 = r * r;
    let mut term = r;
    let mut sum = r;
    for k in 1..=11 {
        term = -term * r2 / ((2 * k) as f64 * (2 * k + 1) as f64);
        sum += term;
    }
    sum
}

pub fn cos(x: f64) -> f64 {
    sin(x + std::f64::consts::FRAC_PI_2)
}

/// Round half to even (numpy's / Python's round) for scalars.
pub fn round_even(x: f64) -> f64 {
    let r = x.round();                                     // half away from zero
    if (x - x.trunc()).abs() == 0.5 && r % 2.0 != 0.0 { r - x.signum() } else { r }
}

/// A roughly standard-normal value from a hash: the sum of four 8-bit uniforms, scaled
/// (Irwin-Hall; sd 1, tails to +-3.5). Pure integer maths: identical everywhere.
#[inline]
pub fn gauss(h: u64) -> f32 {
    let h = (h ^ (h >> 32)) as u32;
    let s = (h & 255) + ((h >> 8) & 255) + ((h >> 16) & 255) + (h >> 24);
    (s as f32 - 510.0) * (1.0 / 147.8)
}

#[cfg(test)]
mod trig_tests {
    use super::*;

    #[test]
    fn sin_cos_match_the_library() {
        for i in -20000..20000 {
            let x = i as f64 * 0.0137;
            assert!((sin(x) - x.sin()).abs() < 1e-12, "sin {x}");
            assert!((cos(x) - x.cos()).abs() < 1e-12, "cos {x}");
        }
        assert!((sin(12345.678) - 12345.678f64.sin()).abs() < 1e-9);
        assert_eq!(round_even(2.5), 2.0);
        assert_eq!(round_even(3.5), 4.0);
        assert_eq!(round_even(-2.5), -2.0);
        assert_eq!(round_even(2.4), 2.0);
    }
}
