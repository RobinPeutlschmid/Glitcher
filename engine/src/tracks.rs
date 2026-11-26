//! Per-frame data from estimate_pose.py, read by the engine itself so a host (the Fusion node)
//! only passes file names: the pose (yaw / pitch / roll / cam_dist curves, glitch.py's Pose) and
//! the body landmarks (glitch.py's Landmarks). Both are plain text: estimate_pose.py writes the
//! landmarks as `<pose>_landmarks.csv` next to its .npz.

use crate::detmath::exp;
use crate::masks::Landmarks;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::SystemTime;

/// Split a line of numbers: comma, semicolon or whitespace separated; '#' starts a comment.
fn cells(line: &str) -> Vec<String> {
    let ln = line.split('#').next().unwrap_or("").trim();
    if ln.is_empty() {
        return vec![];
    }
    if ln.contains(',') || ln.contains(';') {
        let sep = if ln.contains(',') { ',' } else { ';' };
        ln.split(sep).map(|s| s.trim().to_string()).collect()
    } else {
        ln.split_whitespace().map(str::to_string).collect()
    }
}

fn num(s: &str) -> Option<f64> {
    if s.is_empty() {
        return Some(f64::NAN);
    }
    s.parse::<f64>().ok()
}

/// np.interp: piecewise linear through (xs, ys) (xs ascending), held beyond the ends.
fn interp(x: f64, xs: &[f64], ys: &[f64]) -> f64 {
    let n = xs.len();
    if n == 0 {
        return 0.0;
    }
    if x <= xs[0] {
        return ys[0];
    }
    if x >= xs[n - 1] {
        return ys[n - 1];
    }
    let j = xs.partition_point(|&v| v <= x);              // xs[j-1] <= x < xs[j]
    let (x0, x1) = (xs[j - 1], xs[j]);
    if x1 <= x0 {
        return ys[j];
    }
    let w = (x - x0) / (x1 - x0);
    ys[j - 1] + (ys[j] - ys[j - 1]) * w
}

/// Gaussian smoothing (sigma in samples) of an edge-padded curve, same length.
fn smooth(d: &[f64], sig: f64) -> Vec<f64> {
    if sig <= 0.3 || d.len() < 2 {
        return d.to_vec();
    }
    let r = (3.0 * sig) as i64 + 1;
    let k: Vec<f64> = (-r..=r).map(|i| exp(-0.5 * (i as f64 / sig).powi(2))).collect();
    let ks: f64 = k.iter().sum();
    let n = d.len() as i64;
    (0..n)
        .map(|i| {
            (-r..=r).map(|o| d[(i + o).clamp(0, n - 1) as usize] * k[(o + r) as usize]).sum::<f64>() / ks
        })
        .collect()
}

/// np.unwrap of radians: jumps over pi between samples become the short way round.
fn unwrap(y: &[f64]) -> Vec<f64> {
    use std::f64::consts::{PI, TAU};
    let mut out = Vec::with_capacity(y.len());
    let mut off = 0.0;
    for (i, &v) in y.iter().enumerate() {
        if i > 0 {
            let d = v - y[i - 1];
            let mut m = (d + PI).rem_euclid(TAU) - PI;
            if m == -PI && d > 0.0 {
                m = PI;
            }
            if d.abs() >= PI {
                off += m - d;
            }
        }
        out.push(v + off);
    }
    out
}

/// Pose curves on every whole frame between the first and last key.
#[derive(Debug)]
pub struct Pose {
    grid: Vec<f64>,
    yaw: Vec<f64>,       // radians
    pitch: Vec<f64>,
    roll: Vec<f64>,
    cam: Option<Vec<f64>>,
    scale: Option<Vec<f64>>,
}

impl Pose {
    /// Parse a pose CSV (glitch.py's Pose.from_file): with a header line, columns by name
    /// (frame, yaw, pitch, roll, cam_dist; others ignored); without one: frame, yaw[, pitch[, roll]].
    /// `sig` = smoothing in frames (smooth seconds x fps).
    pub fn parse(text: &str, sig: f64, unwrap_yaw: bool) -> Result<Pose, String> {
        let names = ["frame", "yaw", "pitch", "roll", "cam_dist", "scale"];
        let mut cols: Option<[Option<usize>; 6]> = None;
        let mut rows: Vec<[f64; 6]> = Vec::new();
        for (no, line) in text.trim_start_matches('\u{feff}').lines().enumerate() {
            let parts = cells(line);
            if parts.is_empty() {
                continue;
            }
            let nums: Option<Vec<f64>> = parts.iter().map(|p| num(p)).collect();
            let nums = match nums {
                Some(v) if !v.iter().all(|x| x.is_nan()) => v,
                _ => {
                    if !rows.is_empty() || cols.is_some() {
                        return Err(format!("line {}: expected numbers but got '{}'", no + 1, line.trim()));
                    }
                    let lower: Vec<String> = parts.iter().map(|p| p.to_lowercase()).collect();
                    let mut c = [None; 6];
                    for (k, n) in names.iter().enumerate() {
                        c[k] = lower.iter().position(|x| x == n);
                    }
                    if c[0].is_none() || c[1].is_none() {
                        return Err(format!("line {}: the header needs 'frame' and 'yaw' columns", no + 1));
                    }
                    cols = Some(c);
                    continue;
                }
            };
            let c = cols.unwrap_or([Some(0), Some(1), Some(2), Some(3), None, None]);
            let get = |k: usize| c[k].and_then(|i| nums.get(i).copied()).unwrap_or(f64::NAN);
            if !(get(0).is_finite() && get(1).is_finite()) {
                continue;                                   // a key without frame or yaw is no key
            }
            rows.push([get(0), get(1), get(2), get(3), get(4), get(5)]);
        }
        if rows.is_empty() {
            return Err("no 'frame, yaw' rows found".into());
        }
        rows.sort_by(|a, b| a[0].total_cmp(&b[0]));        // stable
        let f: Vec<f64> = rows.iter().map(|r| r[0]).collect();
        let (f0, f1) = (f[0].floor() as i64, f[f.len() - 1].ceil() as i64);
        let grid: Vec<f64> = (f0..=f1).map(|v| v as f64).collect();
        let dense = |k: usize, rad: bool, unw: bool| -> Option<Vec<f64>> {
            let (fk, mut y): (Vec<f64>, Vec<f64>) = rows
                .iter()
                .filter(|r| r[k].is_finite())
                .map(|r| (r[0], if rad { r[k].to_radians() } else { r[k] }))
                .unzip();
            if y.is_empty() {
                return None;
            }
            if unw {
                y = unwrap(&y);
            }
            let d: Vec<f64> = grid.iter().map(|&g| interp(g, &fk, &y)).collect();
            Some(smooth(&d, sig))
        };
        let zeros = || vec![0.0; grid.len()];
        Ok(Pose {
            yaw: dense(1, true, unwrap_yaw).unwrap_or_else(zeros),
            pitch: dense(2, true, false).unwrap_or_else(zeros),
            roll: dense(3, true, false).unwrap_or_else(zeros),
            cam: dense(4, false, false),
            scale: dense(5, false, false),
            grid,
        })
    }

    /// The subject's apparent size relative to the clip's median (the file's scale column).
    pub fn scale(&self, frame: f64) -> Option<f64> {
        self.scale.as_ref().map(|s| interp(frame, &self.grid, s))
    }

    /// (yaw, pitch, roll) in degrees and the camera distance (if the file has it) at a frame.
    pub fn at(&self, frame: f64) -> ([f64; 3], Option<f64>) {
        let g = &self.grid;
        let a = |v: &[f64]| interp(frame, g, v).to_degrees();
        ([a(&self.yaw), a(&self.pitch), a(&self.roll)], self.cam.as_ref().map(|c| interp(frame, g, c)))
    }
}

/// Body landmarks per frame: (frame, 33 x (x, y, visibility)), the image size they were found in.
#[derive(Debug)]
pub struct LandmarkTrack {
    frames: Vec<f64>,
    lm: Vec<Landmarks>,
    pub size: Option<(f64, f64)>,
}

impl LandmarkTrack {
    /// Parse estimate_pose.py's landmarks CSV: a `size, W, H` line, a header, then rows of
    /// `frame, x0, y0, v0, ..., x32, y32, v32` (empty cells = no person that frame). Frames
    /// without a person are dropped (filled from their neighbours) and positions are smoothed
    /// [1 2 1] (MediaPipe jitters a few pixels from frame to frame).
    pub fn parse(text: &str) -> Result<LandmarkTrack, String> {
        let mut size = None;
        let mut rows: Vec<(f64, Landmarks)> = Vec::new();
        for (no, line) in text.trim_start_matches('\u{feff}').lines().enumerate() {
            let parts = cells(line);
            if parts.is_empty() {
                continue;
            }
            let head = parts[0].to_lowercase();
            if head == "size" {
                let v: Vec<f64> = parts[1..].iter().filter_map(|p| p.parse().ok()).collect();
                if v.len() >= 2 && v[0] > 0.0 && v[1] > 0.0 {
                    size = Some((v[0], v[1]));
                }
                continue;
            }
            if head == "frame" {
                continue;
            }
            let nums: Option<Vec<f64>> = parts.iter().map(|p| num(p)).collect();
            let Some(nums) = nums else {
                return Err(format!("line {}: expected numbers", no + 1));
            };
            if !nums.is_empty() && nums[0].is_finite() && nums[1..].iter().all(|v| v.is_nan()) {
                continue;                                   // no person this frame
            }
            if nums.len() < 1 + 33 * 2 || !nums[0].is_finite() {
                return Err(format!("line {}: expected frame and 33 x (x, y, visibility)", no + 1));
            }
            let mut lm = [[f64::NAN; 3]; 33];
            for (i, p) in lm.iter_mut().enumerate() {
                for (c, v) in p.iter_mut().enumerate() {
                    *v = nums.get(1 + 3 * i + c).copied().unwrap_or(if c == 2 { 1.0 } else { f64::NAN });
                }
            }
            if lm[0][0].is_finite() && lm.iter().all(|p| p[0].is_finite() && p[1].is_finite()) {
                rows.push((nums[0], lm));
            }
        }
        if rows.is_empty() {
            return Err("no person in any frame".into());
        }
        rows.sort_by(|a, b| a.0.total_cmp(&b.0));
        let frames: Vec<f64> = rows.iter().map(|r| r.0).collect();
        let raw: Vec<Landmarks> = rows.into_iter().map(|r| r.1).collect();
        let mut lm = raw.clone();
        if raw.len() >= 3 {
            for i in 1..raw.len() - 1 {
                for p in 0..33 {
                    for c in 0..2 {
                        lm[i][p][c] = (raw[i - 1][p][c] + 2.0 * raw[i][p][c] + raw[i + 1][p][c]) / 4.0;
                    }
                }
            }
        }
        Ok(LandmarkTrack { frames, lm, size })
    }

    /// The landmarks at a frame, interpolated between known frames, held beyond them; scaled
    /// from the size they were found in to a w x h canvas.
    pub fn at(&self, frame: f64, w: usize, h: usize) -> Landmarks {
        let f = &self.frames;
        let n = f.len();
        let mut out = if frame <= f[0] {
            self.lm[0]
        } else if frame >= f[n - 1] {
            self.lm[n - 1]
        } else {
            let j = f.partition_point(|&v| v < frame);       // searchsorted (left)
            let wgt = (frame - f[j - 1]) / (f[j] - f[j - 1]).max(1e-9);
            let mut o = self.lm[j];
            for (p, (a, b)) in o.iter_mut().zip(self.lm[j - 1].iter().zip(self.lm[j].iter())) {
                for c in 0..3 {
                    p[c] = (1.0 - wgt) * a[c] + wgt * b[c];
                }
            }
            o
        };
        if let Some((sw, sh)) = self.size {
            let (kx, ky) = (w as f64 / sw, h as f64 / sh);
            for p in out.iter_mut() {
                p[0] *= kx;
                p[1] *= ky;
            }
        }
        out
    }
}

/// A parsed file and the (modified time, length) it was parsed at.
type Cached<T> = (Option<SystemTime>, u64, Arc<T>);

/// Parsed files, re-read when they change on disk.
#[derive(Default)]
pub struct TrackCache {
    pose: Mutex<HashMap<(PathBuf, u64, u64), Cached<Pose>>>,
    lm: Mutex<HashMap<PathBuf, Cached<LandmarkTrack>>>,
}

fn stamp(p: &Path) -> Result<(Option<SystemTime>, u64), String> {
    let m = std::fs::metadata(p).map_err(|e| format!("{}: {e}", p.display()))?;
    Ok((m.modified().ok(), m.len()))
}

fn read(p: &Path) -> Result<String, String> {
    let b = std::fs::read(p).map_err(|e| format!("{}: {e}", p.display()))?;
    String::from_utf8(b).map_err(|_| format!("{}: not a text file", p.display()))
}

impl TrackCache {
    pub fn pose(&self, path: &Path, sig: f64, unwrap_yaw: bool) -> Result<Arc<Pose>, String> {
        let st = stamp(path)?;
        let key = (path.to_path_buf(), sig.to_bits(), unwrap_yaw as u64);
        let mut c = self.pose.lock().unwrap_or_else(|e| e.into_inner());
        if let Some((m, l, p)) = c.get(&key) {
            if (*m, *l) == st {
                return Ok(p.clone());
            }
        }
        let p = Arc::new(Pose::parse(&read(path)?, sig, unwrap_yaw).map_err(|e| format!("{}: {e}", path.display()))?);
        if c.len() > 16 {
            c.clear();
        }
        c.insert(key, (st.0, st.1, p.clone()));
        Ok(p)
    }

    pub fn landmarks(&self, path: &Path) -> Result<Arc<LandmarkTrack>, String> {
        let st = stamp(path)?;
        let mut c = self.lm.lock().unwrap_or_else(|e| e.into_inner());
        if let Some((m, l, t)) = c.get(path) {
            if (*m, *l) == st {
                return Ok(t.clone());
            }
        }
        let t = Arc::new(LandmarkTrack::parse(&read(path)?).map_err(|e| format!("{}: {e}", path.display()))?);
        if c.len() > 16 {
            c.clear();
        }
        c.insert(path.to_path_buf(), (st.0, st.1, t.clone()));
        Ok(t)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pose_reads_headers_interpolates_and_holds() {
        let p = Pose::parse("# comment\nframe,yaw,pitch,roll,scale,confidence\n10,0,5,,1,0.9\n20,90,15,,1,0.9\n", 0.0, false).unwrap();
        let (a, cam) = p.at(15.0);
        assert!((a[0] - 45.0).abs() < 1e-9 && (a[1] - 10.0).abs() < 1e-9 && a[2] == 0.0);
        assert!(cam.is_none());
        assert!((p.at(0.0).0[0]).abs() < 1e-9);           // held before the first key
        assert!((p.at(99.0).0[0] - 90.0).abs() < 1e-9);
        // no header: frame, yaw, pitch, roll
        let q = Pose::parse("0 10 0 5\n4 30 0 5\n", 0.0, false).unwrap();
        assert!((q.at(2.0).0[0] - 20.0).abs() < 1e-9 && (q.at(2.0).0[2] - 5.0).abs() < 1e-9);
        assert!(Pose::parse("frame,yaw\nx,y\n", 0.0, false).is_err());
    }

    #[test]
    fn pose_unwrap_and_smoothing() {
        let p = Pose::parse("frame,yaw\n0,170\n2,-170\n", 0.0, true).unwrap();
        assert!((p.at(1.0).0[0] - 180.0).abs() < 1e-6);   // took the short way round
        let lit = Pose::parse("frame,yaw\n0,170\n2,-170\n", 0.0, false).unwrap();
        assert!(lit.at(1.0).0[0].abs() < 1e-6);
        let s = Pose::parse("frame,yaw\n0,0\n10,0\n11,100\n20,100\n", 2.0, false).unwrap();
        let v = s.at(10.0).0[0];
        assert!(v > 10.0 && v < 50.0, "{v}");
    }

    #[test]
    fn landmarks_parse_scale_and_interpolate() {
        let row = |f: i32, x: f64| {
            let mut s = format!("{f}");
            for _ in 0..33 {
                s += &format!(",{x},{},1", x / 2.0);
            }
            s
        };
        let text = format!("size,100,50\nframe,x0,y0,v0\n{}\n{}\n12,,,\n", row(0, 10.0), row(10, 30.0));
        let t = LandmarkTrack::parse(&text).unwrap();
        let a = t.at(5.0, 200, 100);
        assert!((a[0][0] - 40.0).abs() < 1e-9);           // 20 at the file's size, x2
        assert!((a[3][1] - 20.0).abs() < 1e-9);
        assert!((t.at(-5.0, 100, 50)[0][0] - 10.0).abs() < 1e-9);
    }
}
