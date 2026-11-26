//! Pose estimation for a host that only has frames in memory (the Fusion node): the engine
//! collects the shot's frames as PNGs in a job folder, then runs estimate_pose.py - or the
//! Glitcher app, which carries it and MediaPipe - in the background, and reports its progress.
//!
//! Estimation needs the whole shot (MediaPipe tracks from frame to frame, then yaw is solved
//! over the clip), so it can't run inside one frame's render. The host feeds frames as it renders
//! them and fetches a few missing ones per render; once all are there the estimator starts.
//!
//! The folder holds `frames/f_000000.png ...` (numbered from the range start, passed as
//! `--offset`, so the pose's frame numbers are the host's), `pose.csv`, `pose_landmarks.npz` and
//! `.csv`, `pose.png` (a plot of the curves), `landmarks_cache.npz` (re-solving with new hints
//! skips MediaPipe) and `estimate.log`. A finished run writes `done_<id>`, so a host that
//! reopens a project doesn't run it again.

use crate::image::View;
use rayon::prelude::*;
use std::fs;
use std::io::BufWriter;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

/// What to estimate and how.
#[derive(Clone, Debug, PartialEq)]
pub struct PoseSpec {
    pub dir: PathBuf,
    pub first: i64,
    pub last: i64,
    /// Longest side of the stored frames (MediaPipe works at ~256 px; smaller files, faster).
    pub max_side: usize,
    /// The host's pixels are linear light: encode to sRGB for the pose model.
    pub linear: bool,
    /// Glitcher app (run with --estimate-pose) or estimate_pose.py (run with Python).
    pub estimator: String,
    pub fps: f64,
    pub variant: String,
    pub focal_mm: f64,
    pub sensor_mm: f64,
    pub yaw_hints: String,
    pub pitch_hints: String,
    pub roll_hints: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum State {
    Idle = 0,
    Capturing = 1,
    Estimating = 2,
    Done = 3,
    Failed = 4,
}

#[derive(Clone, Debug)]
pub struct Status {
    pub state: State,
    pub captured: usize,
    pub total: usize,
    /// Landmark frames done by the estimator (its --progress lines).
    pub progress: usize,
    pub message: String,
}

pub struct PoseJob {
    pub spec: PoseSpec,
    pub run: i64,
    have: Vec<bool>,
    /// Frames the host had nothing for (written as empty frames before the run).
    blank: Vec<bool>,
    size: Option<(u32, u32)>,
    pending: Arc<AtomicUsize>,
    write_err: Arc<Mutex<Option<String>>>,
    child: Option<Child>,
    state: State,
    message: String,
}

fn frame_path(dir: &Path, i: usize) -> PathBuf {
    dir.join("frames").join(format!("f_{i:06}.png"))
}

/// "0:0, 120:180" -> ["0:0", "120:180"]; an error names the bad item.
pub fn parse_hints(text: &str) -> Result<Vec<String>, String> {
    let mut out = Vec::new();
    for it in text.split([',', ';', ' ', '\t', '\n']).map(str::trim).filter(|s| !s.is_empty()) {
        let ok = it.split_once(':').is_some_and(|(f, v)| f.trim().parse::<f64>().is_ok() && v.trim().parse::<f64>().is_ok());
        if !ok {
            return Err(format!("hint '{it}': expected FRAME:DEGREES, e.g. 0:0, 120:180"));
        }
        out.push(it.replace(' ', ""));
    }
    Ok(out)
}

impl PoseJob {
    /// A job for run id `run`. `restart` deletes earlier frames and landmarks first. Frames
    /// already in the folder from an earlier run of the same range are kept. If this run id
    /// already finished (its marker exists), the job is Done without doing anything.
    pub fn new(spec: PoseSpec, run: i64, restart: bool) -> Result<PoseJob, String> {
        if spec.last < spec.first {
            return Err("empty frame range".into());
        }
        for h in [&spec.yaw_hints, &spec.pitch_hints, &spec.roll_hints] {
            parse_hints(h)?;
        }
        let n = (spec.last - spec.first + 1) as usize;
        if n > 100_000 {
            return Err(format!("{n} frames: set the comp's render range to the shot"));
        }
        let frames = spec.dir.join("frames");
        if restart {
            let _ = fs::remove_dir_all(&frames);
            let _ = fs::remove_file(spec.dir.join("landmarks_cache.npz"));
        }
        fs::create_dir_all(&frames).map_err(|e| format!("{}: {e}", frames.display()))?;
        // the range is part of what the frames mean: a different range starts over
        let range = format!("{} {}", spec.first, spec.last);
        let rp = spec.dir.join("range.txt");
        if fs::read_to_string(&rp).ok().as_deref() != Some(range.as_str()) {
            let _ = fs::remove_dir_all(&frames);
            fs::create_dir_all(&frames).map_err(|e| format!("{}: {e}", frames.display()))?;
            fs::write(&rp, &range).map_err(|e| format!("{}: {e}", rp.display()))?;
        }
        let have: Vec<bool> = (0..n).map(|i| frame_path(&spec.dir, i).is_file()).collect();
        let mut job = PoseJob { have, blank: vec![false; n], size: None, pending: Arc::new(AtomicUsize::new(0)),
                                write_err: Arc::new(Mutex::new(None)), child: None, state: State::Capturing,
                                message: String::new(), spec, run };
        if job.spec.dir.join(job.done_marker()).is_file() && job.spec.dir.join("pose.csv").is_file() {
            job.state = State::Done;
            job.message = format!("pose ready: {}", job.spec.dir.join("pose.csv").display());
        }
        Ok(job)
    }

    /// The file that says this run (with this range) finished.
    fn done_marker(&self) -> String {
        format!("done_{}_{}_{}", self.run, self.spec.first, self.spec.last)
    }

    /// Frame numbers still needed, in order, at most `max`.
    pub fn missing(&self, max: usize) -> Vec<i64> {
        if self.state != State::Capturing {
            return vec![];
        }
        (0..self.have.len()).filter(|&i| !self.have[i]).take(max).map(|i| self.spec.first + i as i64).collect()
    }

    /// The host's frame `frame` (premultiplied, any size) - stored if the job still needs it.
    /// None = the host has no picture at that frame. Returns whether it was taken.
    pub fn feed(&mut self, frame: i64, img: Option<&View>) -> bool {
        if self.state != State::Capturing || frame < self.spec.first || frame > self.spec.last {
            return false;
        }
        let i = (frame - self.spec.first) as usize;
        if self.have[i] {
            return false;
        }
        self.have[i] = true;
        let Some(v) = img else {
            self.blank[i] = true;
            return true;
        };
        let (w, h, rgba) = to_rgba8(v, self.spec.max_side, self.spec.linear);
        self.size.get_or_insert((w, h));
        let path = frame_path(&self.spec.dir, i);
        let (pending, err) = (self.pending.clone(), self.write_err.clone());
        pending.fetch_add(1, Ordering::SeqCst);
        rayon::spawn(move || {
            if let Err(e) = write_png(&path, w, h, &rgba) {
                *err.lock().unwrap_or_else(|e| e.into_inner()) = Some(e);
            }
            pending.fetch_sub(1, Ordering::SeqCst);
        });
        true
    }

    /// Advance: start the estimator once every frame is stored, notice when it ends.
    pub fn poll(&mut self) -> Status {
        if let Some(e) = self.write_err.lock().unwrap_or_else(|e| e.into_inner()).take() {
            self.state = State::Failed;
            self.message = format!("could not store a frame: {e}");
        }
        if self.state == State::Capturing && self.have.iter().all(|&h| h) && self.pending.load(Ordering::SeqCst) == 0 {
            match self.launch() {
                Ok(()) => {
                    self.state = State::Estimating;
                    self.message = "estimating (loading MediaPipe)".into();
                }
                Err(m) => {
                    self.state = State::Failed;
                    self.message = m;
                }
            }
        }
        let mut progress = 0;
        if self.state == State::Estimating {
            let log = fs::read_to_string(self.spec.dir.join("estimate.log")).unwrap_or_default();
            progress = log.lines().filter_map(|l| l.trim().strip_prefix("landmarks ")).filter_map(|r| r.split('/').next()?.parse().ok()).max().unwrap_or(0);
            if progress > 0 {
                self.message = format!("estimating: landmarks {progress}/{}", self.have.len());
            }
            if log.contains("using cached landmarks") {
                self.message = "estimating: re-solving from the cached landmarks".into();
            }
            if let Some(child) = self.child.as_mut() {
                if let Ok(Some(st)) = child.try_wait() {
                    self.child = None;
                    let csv = self.spec.dir.join("pose.csv");
                    if st.success() && csv.is_file() {
                        self.state = State::Done;
                        let _ = fs::write(self.spec.dir.join(self.done_marker()), "");
                        // estimate_pose.py's summary: "N frames -> path  (person found in ...; yaw ...)"
                        let summary = log.lines().rev().find(|l| l.contains(" -> ")).unwrap_or("");
                        let detail = summary.find('(').map(|i| summary[i..].trim()).unwrap_or("");
                        self.message = format!("pose ready {detail}: {}", csv.display());
                    } else {
                        self.state = State::Failed;
                        // its own error line if there is one, else the last lines
                        let tail = match log.lines().rev().find(|l| l.contains("error:") || l.starts_with("Traceback")) {
                            Some(l) => l.trim().to_string(),
                            None => {
                                let t: Vec<&str> = log.lines().filter(|l| !l.trim().is_empty() && !l.starts_with("landmarks ")).collect();
                                t[t.len().saturating_sub(3)..].join(" | ")
                            }
                        };
                        self.message = format!("the estimator failed ({st}); see {}: {tail}", self.spec.dir.join("estimate.log").display());
                    }
                }
            }
        }
        Status { state: self.state, captured: self.have.iter().filter(|&&h| h).count(), total: self.have.len(),
                 progress, message: self.message.clone() }
    }

    /// The estimator's command line (program, arguments).
    pub fn command(&self) -> Result<(String, Vec<String>), String> {
        let s = &self.spec;
        let est = s.estimator.trim().trim_matches('"').to_string();
        if est.is_empty() {
            return Err("no estimator: set Pose estimation > Estimator to the Glitcher app or estimate_pose.py".into());
        }
        if !Path::new(&est).exists() {
            return Err(format!("estimator not found: {est}"));
        }
        let (prog, mut args) = if est.to_lowercase().ends_with(".py") {
            // the Python with MediaPipe: GLITCHER_PYTHON, a virtual environment next to the
            // script, else the system's
            let here = Path::new(&est).parent().unwrap_or(Path::new("."));
            let venv = ["venv", ".venv"].iter().flat_map(|v| {
                [here.join(v).join("Scripts").join("python.exe"), here.join(v).join("bin").join("python3")]
            }).find(|p| p.is_file());
            match (std::env::var("GLITCHER_PYTHON").ok().filter(|p| !p.trim().is_empty()), venv) {
                (Some(py), _) => (py, vec![est]),
                (None, Some(v)) => (v.to_string_lossy().into_owned(), vec![est]),
                (None, None) if cfg!(windows) => ("py".to_string(), vec!["-3".to_string(), est]),
                (None, None) => ("python3".to_string(), vec![est]),
            }
        } else {
            (est, vec!["--estimate-pose".to_string()])
        };
        let d = |f: &str| s.dir.join(f).to_string_lossy().into_owned();
        args.extend([d("frames"), "--offset".into(), s.first.to_string(), "-o".into(), d("pose.csv"),
                     "--fps".into(), format!("{}", s.fps), "--cache".into(), d("landmarks_cache.npz"),
                     "--landmarks-out".into(), d("pose_landmarks.npz"), "--plot".into(), d("pose.png"),
                     "--progress".into(), "--variant".into(), s.variant.clone()]);
        if s.focal_mm > 0.0 {
            args.extend(["--focal-mm".into(), format!("{}", s.focal_mm), "--sensor-mm".into(), format!("{}", s.sensor_mm)]);
        }
        for (flag, text) in [("--hint", &s.yaw_hints), ("--pitch-hint", &s.pitch_hints), ("--roll-hint", &s.roll_hints)] {
            for h in parse_hints(text)? {
                args.push(format!("{flag}={h}"));            // = keeps a negative value an argument
            }
        }
        Ok((prog, args))
    }

    fn launch(&mut self) -> Result<(), String> {
        // frames the host had nothing for: empty (transparent) frames of the same size
        let (w, h) = self.size.unwrap_or((64, 64));
        for i in (0..self.blank.len()).filter(|&i| self.blank[i]) {
            write_png(&frame_path(&self.spec.dir, i), w, h, &vec![0u8; 4 * (w * h) as usize])?;
        }
        let (prog, args) = self.command()?;
        let logp = self.spec.dir.join("estimate.log");
        let log = fs::File::create(&logp).map_err(|e| format!("{}: {e}", logp.display()))?;
        let log2 = log.try_clone().map_err(|e| e.to_string())?;
        let _ = fs::remove_file(self.spec.dir.join("pose.csv"));       // so success means a new one
        let mut cmd = Command::new(&prog);
        cmd.args(&args).stdin(Stdio::null()).stdout(log).stderr(log2).current_dir(&self.spec.dir)
            .env("GLITCHER_LOG", &logp);                  // a windowed app without std handles logs there
        #[cfg(windows)]
        {
            use std::os::windows::process::CommandExt;
            cmd.creation_flags(0x0800_0000);                             // CREATE_NO_WINDOW
        }
        let child = cmd.spawn().map_err(|e| format!("could not start {prog}: {e}"))?;
        self.child = Some(child);
        Ok(())
    }
}

/// Premultiplied float -> straight 8-bit RGBA, box-downscaled so the longest side is at most
/// `max_side`. Linear input is encoded to sRGB.
pub fn to_rgba8(v: &View, max_side: usize, linear: bool) -> (u32, u32, Vec<u8>) {
    let k = v.w.max(v.h).div_ceil(max_side.max(16)).max(1);
    let (w, h) = ((v.w / k).max(1), (v.h / k).max(1));
    let enc = |x: f32| -> f32 {
        let x = x.clamp(0.0, 1.0);
        if !linear {
            x
        } else if x <= 0.003_130_8 {
            12.92 * x
        } else {
            1.055 * x.powf(1.0 / 2.4) - 0.055
        }
    };
    let mut out = vec![0u8; 4 * w * h];
    out.par_chunks_mut(4 * w).enumerate().for_each(|(y, row)| {
        for x in 0..w {
            let mut acc = [0f32; 4];
            for yy in y * k..((y + 1) * k).min(v.h) {
                for xx in x * k..((x + 1) * k).min(v.w) {
                    let p = v.px(xx as i64, yy as i64);
                    for c in 0..4 {
                        acc[c] += p[c];
                    }
                }
            }
            let n = (k * k) as f32;
            let a = (acc[3] / n).clamp(0.0, 1.0);
            for c in 0..3 {
                let s = if a > 0.0 { acc[c] / n / a } else { 0.0 };
                row[4 * x + c] = (enc(s) * 255.0 + 0.5) as u8;
            }
            row[4 * x + 3] = (a * 255.0 + 0.5) as u8;
        }
    });
    (w as u32, h as u32, out)
}

fn write_png(path: &Path, w: u32, h: u32, rgba: &[u8]) -> Result<(), String> {
    let tmp = path.with_extension("png.tmp");
    let f = fs::File::create(&tmp).map_err(|e| format!("{}: {e}", tmp.display()))?;
    let mut enc = png::Encoder::new(BufWriter::new(f), w, h);
    enc.set_color(png::ColorType::Rgba);
    enc.set_depth(png::BitDepth::Eight);
    enc.set_compression(png::Compression::Fast);
    let mut wr = enc.write_header().map_err(|e| e.to_string())?;
    wr.write_image_data(rgba).map_err(|e| e.to_string())?;
    wr.finish().map_err(|e| e.to_string())?;
    fs::rename(&tmp, path).map_err(|e| format!("{}: {e}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::image::Frame;

    fn spec(dir: &Path, est: &str) -> PoseSpec {
        PoseSpec { dir: dir.to_path_buf(), first: 10, last: 14, max_side: 64, linear: false, estimator: est.into(),
                   fps: 25.0, variant: "lite".into(), focal_mm: 0.0, sensor_mm: 36.0, yaw_hints: "10:0, 14:-90".into(),
                   pitch_hints: String::new(), roll_hints: String::new() }
    }

    fn tmpdir(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("glitch_core_test_{name}_{}", std::process::id()));
        let _ = fs::remove_dir_all(&d);
        d
    }

    #[test]
    fn hints_parse_and_reject() {
        assert_eq!(parse_hints("0:0, 120:180;7:-30").unwrap(), vec!["0:0", "120:180", "7:-30"]);
        assert!(parse_hints("0:0, x").is_err());
        assert!(parse_hints("").unwrap().is_empty());
    }

    #[test]
    fn downscale_and_unpremultiply() {
        let mut f = Frame::new(200, 100);
        for p in f.data.chunks_mut(4) {
            p.copy_from_slice(&[0.25, 0.1, 0.0, 0.5]);
        }
        let (w, h, px) = to_rgba8(&f.view(), 64, false);
        assert_eq!((w, h), (50, 25));                      // 200 / ceil(200 / 64) = 50
        assert_eq!(&px[..4], &[128, 51, 0, 128]);
    }

    #[test]
    fn capture_then_run_a_stand_in_estimator() {
        let dir = tmpdir("posejob");
        // a stand-in estimator: checks its arguments, writes the outputs
        let fake = std::env::temp_dir().join(format!("fake_estimate_{}.py", std::process::id()));
        fs::write(&fake, r#"
import sys, os
a = sys.argv[1:]
assert a[a.index("--offset") + 1] == "10", a
assert "--hint=14:-90" in a, a
frames = sorted(os.listdir(a[0]))
assert frames == [f"f_{i:06d}.png" for i in range(5)], frames
for i in range(5):
    print(f"landmarks {i + 1}/5", file=sys.stderr)
out = a[a.index("-o") + 1]
open(out, "w").write("frame,yaw\n10,0\n14,-90\n")
print(f"5 frames -> {out}  (person found in 5)")
"#).unwrap();
        if Command::new("python3").arg("-c").arg("1").status().is_err() {
            return;                                            // no Python here: nothing to run
        }
        // SAFETY (test only): no other thread reads the environment meanwhile
        unsafe { std::env::set_var("GLITCHER_PYTHON", "python3") };
        let mut job = PoseJob::new(spec(&dir, fake.to_str().unwrap()), 1, false).unwrap();
        assert_eq!(job.missing(10), vec![10, 11, 12, 13, 14]);
        let mut fr = Frame::new(120, 80);
        fr.data[3] = 1.0;
        for f in [10, 11, 13, 14] {
            assert!(job.feed(f, Some(&fr.view())));
        }
        assert!(!job.feed(11, Some(&fr.view())));                // already have it
        assert!(!job.feed(40, Some(&fr.view())));                // outside the range
        assert_eq!(job.poll().state, State::Capturing);
        assert!(job.feed(12, None));                             // the host had nothing there
        let mut st = job.poll();
        for _ in 0..200 {
            if st.state != State::Capturing && st.state != State::Estimating {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(50));
            st = job.poll();
        }
        assert_eq!(st.state, State::Done, "{}", st.message);
        assert!(st.message.contains("person found in 5"));
        // reopening: the same run id is not run again; a new one keeps the frames
        assert_eq!(PoseJob::new(spec(&dir, fake.to_str().unwrap()), 1, false).unwrap().poll().state, State::Done);
        let again = PoseJob::new(spec(&dir, fake.to_str().unwrap()), 2, false).unwrap();
        assert!(again.missing(10).is_empty());
        // start over: frames gone
        let fresh = PoseJob::new(spec(&dir, fake.to_str().unwrap()), 3, true).unwrap();
        assert_eq!(fresh.missing(10).len(), 5);
        let _ = fs::remove_dir_all(&dir);
        let _ = fs::remove_file(&fake);
    }

    #[test]
    fn a_missing_estimator_fails_with_a_message() {
        let dir = tmpdir("posejob_missing");
        let mut job = PoseJob::new(spec(&dir, "/nope/Glitcher"), 1, false).unwrap();
        let fr = Frame::new(32, 32);
        let mut f = fr.clone();
        for t in 10..=14 {
            job.feed(t, Some(&f.view()));
        }
        let mut st = job.poll();
        for _ in 0..100 {
            if st.state != State::Capturing {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(20));
            st = job.poll();
        }
        assert_eq!(st.state, State::Failed);
        assert!(st.message.contains("not found"), "{}", st.message);
        let _ = fs::remove_dir_all(&dir);
    }
}
