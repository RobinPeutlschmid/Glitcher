//! "Show pose": a check drawn over the output - the body landmarks (subject's left orange,
//! right blue, so a mirrored fit is easy to spot), a ring around the body seen through the 3D
//! camera with an arrow where the subject faces, and the values used on this frame as text.
//! Meant for setting up a shot, not for the final render.

use crate::image::View;
use crate::layers::Roll;
use crate::masks::Landmarks;
use crate::sched::Space3D;

type Rgb = [f32; 3];
const LEFT: Rgb = [1.0, 0.55, 0.1];
const RIGHT: Rgb = [0.2, 0.6, 1.0];
const MID: Rgb = [0.95, 0.95, 0.95];
const RING_FRONT: Rgb = [0.2, 1.0, 0.85];
const RING_BACK: Rgb = [0.1, 0.45, 0.4];
const FACE: Rgb = [1.0, 0.15, 0.6];
const UP: Rgb = [0.5, 1.0, 0.2];
const TEXT: Rgb = [1.0, 1.0, 1.0];

/// MediaPipe's body connections (face outline left out).
const BONES: [(usize, usize); 31] = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24), (23, 24), (23, 25), (25, 27),
    (24, 26), (26, 28), (27, 29), (29, 31), (27, 31), (28, 30), (30, 32), (28, 32), (15, 17), (15, 19),
    (15, 21), (17, 19), (16, 18), (16, 20), (16, 22), (18, 20), (0, 2), (2, 7), (0, 5), (5, 8), (9, 10),
];

fn side(i: usize) -> Rgb {
    // MediaPipe: odd = the subject's left, even = right (0 = nose)
    if i == 0 { MID } else if i % 2 == 1 { LEFT } else { RIGHT }
}

/// An opaque disc of radius r at (x, y).
fn dot(v: &View, x: f64, y: f64, r: f64, c: Rgb) {
    let (y0, y1) = ((y - r).floor().max(0.0) as i64, ((y + r).ceil() as i64).min(v.h as i64 - 1));
    let (x0, x1) = ((x - r).floor().max(0.0) as i64, ((x + r).ceil() as i64).min(v.w as i64 - 1));
    for yy in y0..=y1 {
        let mut lo = None;
        let mut hi = 0;
        for xx in x0..=x1 {
            let (dx, dy) = (xx as f64 + 0.5 - x, yy as f64 + 0.5 - y);
            if dx * dx + dy * dy <= r * r {
                lo.get_or_insert(xx);
                hi = xx;
            }
        }
        if let Some(lo) = lo {
            // SAFETY: drawing is single-threaded, one row span at a time
            let s = unsafe { v.span_mut(yy as usize, lo as usize, hi as usize + 1) };
            for p in s.chunks_mut(4) {
                p.copy_from_slice(&[c[0], c[1], c[2], 1.0]);
            }
        }
    }
}

fn line(v: &View, a: (f64, f64), b: (f64, f64), r: f64, c: Rgb) {
    let len = ((b.0 - a.0).powi(2) + (b.1 - a.1).powi(2)).sqrt();
    if !len.is_finite() || len > 1e5 {
        return;
    }
    let n = (len / (0.5 * r).max(0.5)).ceil().max(1.0) as usize;
    for i in 0..=n {
        let t = i as f64 / n as f64;
        dot(v, a.0 + (b.0 - a.0) * t, a.1 + (b.1 - a.1) * t, r, c);
    }
}

/// 5 x 7 glyphs, one row per byte, leftmost pixel = bit 4.
fn glyph(ch: char) -> [u8; 7] {
    match ch.to_ascii_uppercase() {
        '0' => [0x0E, 0x11, 0x13, 0x15, 0x19, 0x11, 0x0E],
        '1' => [0x04, 0x0C, 0x04, 0x04, 0x04, 0x04, 0x0E],
        '2' => [0x0E, 0x11, 0x01, 0x02, 0x04, 0x08, 0x1F],
        '3' => [0x1F, 0x02, 0x04, 0x02, 0x01, 0x11, 0x0E],
        '4' => [0x02, 0x06, 0x0A, 0x12, 0x1F, 0x02, 0x02],
        '5' => [0x1F, 0x10, 0x1E, 0x01, 0x01, 0x11, 0x0E],
        '6' => [0x06, 0x08, 0x10, 0x1E, 0x11, 0x11, 0x0E],
        '7' => [0x1F, 0x01, 0x02, 0x04, 0x08, 0x08, 0x08],
        '8' => [0x0E, 0x11, 0x11, 0x0E, 0x11, 0x11, 0x0E],
        '9' => [0x0E, 0x11, 0x11, 0x0F, 0x01, 0x02, 0x0C],
        '-' => [0, 0, 0, 0x1F, 0, 0, 0],
        '+' => [0, 0x04, 0x04, 0x1F, 0x04, 0x04, 0],
        '.' => [0, 0, 0, 0, 0, 0x0C, 0x0C],
        ':' => [0, 0x0C, 0x0C, 0, 0x0C, 0x0C, 0],
        '/' => [0x01, 0x01, 0x02, 0x04, 0x08, 0x10, 0x10],
        '(' => [0x02, 0x04, 0x08, 0x08, 0x08, 0x04, 0x02],
        ')' => [0x08, 0x04, 0x02, 0x02, 0x02, 0x04, 0x08],
        'A' => [0x0E, 0x11, 0x11, 0x1F, 0x11, 0x11, 0x11],
        'B' => [0x1E, 0x11, 0x11, 0x1E, 0x11, 0x11, 0x1E],
        'C' => [0x0E, 0x11, 0x10, 0x10, 0x10, 0x11, 0x0E],
        'D' => [0x1C, 0x12, 0x11, 0x11, 0x11, 0x12, 0x1C],
        'E' => [0x1F, 0x10, 0x10, 0x1E, 0x10, 0x10, 0x1F],
        'F' => [0x1F, 0x10, 0x10, 0x1E, 0x10, 0x10, 0x10],
        'G' => [0x0E, 0x11, 0x10, 0x17, 0x11, 0x11, 0x0F],
        'H' => [0x11, 0x11, 0x11, 0x1F, 0x11, 0x11, 0x11],
        'I' => [0x0E, 0x04, 0x04, 0x04, 0x04, 0x04, 0x0E],
        'J' => [0x07, 0x02, 0x02, 0x02, 0x02, 0x12, 0x0C],
        'K' => [0x11, 0x12, 0x14, 0x18, 0x14, 0x12, 0x11],
        'L' => [0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x1F],
        'M' => [0x11, 0x1B, 0x15, 0x15, 0x11, 0x11, 0x11],
        'N' => [0x11, 0x11, 0x19, 0x15, 0x13, 0x11, 0x11],
        'O' => [0x0E, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0E],
        'P' => [0x1E, 0x11, 0x11, 0x1E, 0x10, 0x10, 0x10],
        'Q' => [0x0E, 0x11, 0x11, 0x11, 0x15, 0x12, 0x0D],
        'R' => [0x1E, 0x11, 0x11, 0x1E, 0x14, 0x12, 0x11],
        'S' => [0x0F, 0x10, 0x10, 0x0E, 0x01, 0x01, 0x1E],
        'T' => [0x1F, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04],
        'U' => [0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0E],
        'V' => [0x11, 0x11, 0x11, 0x11, 0x11, 0x0A, 0x04],
        'W' => [0x11, 0x11, 0x11, 0x15, 0x15, 0x15, 0x0A],
        'X' => [0x11, 0x11, 0x0A, 0x04, 0x0A, 0x11, 0x11],
        'Y' => [0x11, 0x11, 0x11, 0x0A, 0x04, 0x04, 0x04],
        'Z' => [0x1F, 0x01, 0x02, 0x04, 0x08, 0x10, 0x1F],
        _ => [0; 7],
    }
}

/// Fill a rectangle (clipped) with an opaque colour.
fn rect(v: &View, x0: i64, y0: i64, x1: i64, y1: i64, c: [f32; 4]) {
    let (x0, x1) = (x0.clamp(0, v.w as i64) as usize, x1.clamp(0, v.w as i64) as usize);
    for y in y0.max(0)..y1.min(v.h as i64) {
        if x1 > x0 {
            // SAFETY: single-threaded
            for p in unsafe { v.span_mut(y as usize, x0, x1) }.chunks_mut(4) {
                p.copy_from_slice(&c);
            }
        }
    }
}

/// Lines of text at the top left, `px` screen pixels per font pixel, on a dark band.
pub fn text(v: &View, lines: &[String], px: usize) {
    let (cw, lh, m) = (6 * px, 9 * px, 2 * px);
    let wide = lines.iter().map(|l| l.chars().count()).max().unwrap_or(0);
    rect(v, 0, 0, (m + wide * cw + m) as i64, (m + lines.len() * lh + m) as i64, [0.0, 0.0, 0.0, 0.75]);
    for (li, l) in lines.iter().enumerate() {
        for (ci, ch) in l.chars().enumerate() {
            let g = glyph(ch);
            for (gy, row) in g.iter().enumerate() {
                for gx in 0..5 {
                    if row & (0x10 >> gx) != 0 {
                        let (x, y) = (m + ci * cw + gx * px, m + li * lh + gy * px);
                        rect(v, x as i64, y as i64, (x + px) as i64, (y + px) as i64, [TEXT[0], TEXT[1], TEXT[2], 1.0]);
                    }
                }
            }
        }
    }
}

/// The skeleton of the landmarks.
pub fn skeleton(v: &View, lm: &Landmarks) {
    let r = (v.h as f64 / 400.0).max(1.0);
    let ok = |i: usize| lm[i][0].is_finite() && lm[i][1].is_finite();
    for &(a, b) in BONES.iter() {
        if ok(a) && ok(b) {
            let c = if side(a) == side(b) { side(a) } else { MID };
            line(v, (lm[a][0], lm[a][1]), (lm[b][0], lm[b][1]), r, c);
        }
    }
    for (i, p) in lm.iter().enumerate() {
        if ok(i) && !(1..=10).contains(&i) {
            dot(v, p[0], p[1], 2.2 * r, side(i));
        }
    }
}

/// The 3D camera as a ring around the body (front half bright, back half dim), an arrow where
/// the subject faces and a line for its up direction, through the same projection the planes
/// use, then turned by the roll.
pub fn gizmo(v: &View, sp: &Space3D, bbox: (i64, i64, i64, i64)) {
    let (x0, y0, x1, y1) = bbox;
    let (bw, bh) = ((x1 - x0) as f64, (y1 - y0) as f64);
    let (cx, cy) = ((x0 + x1) as f64 / 2.0, (y0 + y1) as f64 / 2.0);
    let roll = Roll::new(sp.roll.to_degrees(), cx, cy);
    let d = sp.cam_dist * bh;
    let dotp = |a: &[f64; 3], b: &[f64; 3]| a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    // subject-space point -> screen, and its depth toward the camera
    let proj = |p: [f64; 3]| -> ((f64, f64), f64) {
        let (x, y, z) = (dotp(&p, &sp.right), dotp(&p, &sp.up), dotp(&p, &sp.cam));
        let s = d / (d - z).max(0.25 * d);
        (roll.to_screen(cx + x * s, cy + y * s), z)
    };
    let r = 0.6 * bw.max(0.25 * bh);
    let t = (v.h as f64 / 500.0).max(1.0);
    let n = 96;
    for i in 0..n {
        let (a0, a1) = (std::f64::consts::TAU * i as f64 / n as f64, std::f64::consts::TAU * (i + 1) as f64 / n as f64);
        let (pa, za) = proj([r * a0.sin(), 0.0, r * a0.cos()]);
        let (pb, _) = proj([r * a1.sin(), 0.0, r * a1.cos()]);
        line(v, pa, pb, t, if za >= 0.0 { RING_FRONT } else { RING_BACK });
    }
    let (c0, _) = proj([0.0, 0.0, 0.0]);
    let (up, _) = proj([0.0, -0.45 * bh, 0.0]);
    line(v, c0, up, t, UP);
    // facing arrow: the subject's front (+z in its space, toward the camera at yaw 0)
    let (tip, _) = proj([0.0, 0.0, 1.25 * r]);
    line(v, c0, tip, 1.6 * t, FACE);
    let (l, _) = proj([0.12 * r, 0.0, 1.05 * r]);
    let (rr, _) = proj([-0.12 * r, 0.0, 1.05 * r]);
    line(v, tip, l, 1.6 * t, FACE);
    line(v, tip, rr, 1.6 * t, FACE);
    dot(v, c0.0, c0.1, 3.0 * t, FACE);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::image::Frame;

    #[test]
    fn draws_inside_the_frame_only() {
        let mut f = Frame::new(320, 180);
        let v = f.view();
        let sp = Space3D::new(0.7, 3.0, 0.35, 0.3, 0.4, 0.25);
        gizmo(&v, &sp, (-200, -50, 600, 400));                      // box past the edges: clipped
        text(&v, &["YAW 40.1  PITCH -3".into()], 2);
        let lit = f.data.chunks(4).filter(|p| p[3] == 1.0).count();
        assert!(lit > 500);
        assert!(f.data.iter().all(|x| x.is_finite() && *x >= 0.0 && *x <= 1.0));
    }

    #[test]
    fn the_facing_arrow_turns_with_yaw() {
        // yaw +90: the front points to screen right, so pixels right of the centre light up
        let lit_side = |yaw: f64| {
            let mut f = Frame::new(400, 400);
            gizmo(&f.view(), &Space3D::new(yaw, 3.0, 0.35, 0.0, 0.0, 0.0), (150, 100, 250, 300));
            let face = |p: &[f32]| p[0] == FACE[0] && p[2] == FACE[2] && p[3] == 1.0;
            let (mut l, mut r) = (0, 0);
            for (i, p) in f.data.chunks(4).enumerate() {
                if face(p) {
                    if i % 400 < 190 { l += 1 } else if i % 400 > 210 { r += 1 }
                }
            }
            (l, r)
        };
        let (l, r) = lit_side(std::f64::consts::FRAC_PI_2);
        assert!(r > 3 * l.max(1), "{l} {r}");
        let (l, r) = lit_side(-std::f64::consts::FRAC_PI_2);
        assert!(l > 3 * r.max(1), "{l} {r}");
    }
}
