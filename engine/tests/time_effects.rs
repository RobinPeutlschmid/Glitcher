//! The time effects through the public engine API, on a synthetic moving subject.

use glitch_core::engine::Engine;
use glitch_core::image::{Frame, View};
use glitch_core::sched::{ECHO, MAX_BACK, MOSH, Params, STUTTER};

const W: usize = 320;
const H: usize = 180;

/// A textured soft ellipse moving right (premultiplied), on transparent.
fn frame(i: usize) -> Frame {
    let mut f = Frame::new(W, H);
    let (cx, cy) = (90.0 + 3.0 * i as f32, 95.0);
    for y in 0..H {
        for x in 0..W {
            let (dx, dy) = ((x as f32 - cx) / 34.0, (y as f32 - cy) / 70.0);
            let a = ((1.0 - dx * dx - dy * dy) * 8.0).clamp(0.0, 1.0);
            if a > 0.0 {
                let t = ((x as f32 * 0.11).sin() * 0.5 + 0.5) * 0.6 + 0.2;   // test data only
                let k = 4 * (y * W + x);
                f.data[k..k + 4].copy_from_slice(&[t * a, 0.5 * a, (1.0 - t) * a, a]);
            }
        }
    }
    f
}

struct Seq {
    frames: Vec<Frame>,
}

impl Seq {
    fn new(n: usize) -> Seq {
        Seq { frames: (0..n).map(frame).collect() }
    }
    /// Render frame i with the earlier frames given by `give(k)`.
    fn render(&mut self, e: &Engine, p: &Params, i: usize, give: &dyn Fn(usize) -> bool) -> (Frame, [usize; 3]) {
        let mut views: Vec<View> = self.frames.iter_mut().map(|f| f.view()).collect();
        let back: Vec<Option<View>> = (1..=MAX_BACK).map(|k| if k <= i && give(k) { Some(views[i - k]) } else { None }).collect();
        let mut out = Frame::new(W, H);
        let st = e.render(p, i as i64, views.remove(i), &back, out.view());
        (out, st.events)
    }
}

fn params(seed: u32) -> Params {
    let mut p = Params { seed, intensity: 0.9, bursts: 0.2, ..Default::default() };
    for g in p.gains.iter_mut() {
        *g = 0.0;                                   // only the time effects exist in the engine so far
    }
    p.gains[ECHO] = 2.5;
    p.gains[STUTTER] = 2.5;
    p.gains[MOSH] = 2.5;
    p.sane()
}

#[test]
fn every_effect_fires_and_output_stays_valid() {
    let mut s = Seq::new(160);
    let e = Engine::new();
    let p = params(3);
    let mut fired = [0usize; 3];
    let mut changed = 0;
    for i in 0..160 {
        let (out, ev) = s.render(&e, &p, i, &|_| true);
        for k in 0..3 {
            fired[k] += (ev[k] > 0) as usize;
        }
        if out.data != s.frames[i].data {
            changed += 1;
        }
        for px in out.data.chunks(4) {
            assert!(px[3] <= 1.0 + 1e-5 && px[3] >= 0.0, "alpha {px:?}");
            assert!(px[..3].iter().all(|&c| c >= 0.0 && c <= px[3] + 1e-5), "not premultiplied: {px:?}");
        }
    }
    assert!(fired.iter().all(|&n| n > 0), "fired {fired:?}");
    assert!(changed > 20, "{changed} frames changed");
}

#[test]
fn no_history_or_no_strength_changes_nothing() {
    let mut s = Seq::new(40);
    let e = Engine::new();
    let p = params(5);
    for i in [0usize, 20, 39] {
        let (out, _) = s.render(&e, &p, i, &|_| false);
        assert_eq!(out.data, s.frames[i].data, "frame {i} without history");
    }
    let mut off = params(5);
    off.gains = [0.0; 13];
    for i in 0..40 {
        let (out, _) = s.render(&e, &off, i, &|_| true);
        assert_eq!(out.data, s.frames[i].data);
    }
}

#[test]
fn needed_frames_are_exactly_enough() {
    // rendering with only the frames `needed` names must give the same image as with all 24
    let mut s = Seq::new(120);
    let e = Engine::new();
    for seed in [1u32, 8, 21] {
        let p = params(seed);
        for i in 0..120 {
            let need = e.needed(&p, i as i64, W, H, i);
            assert!(need.iter().all(|&k| k >= 1 && k <= i.min(MAX_BACK)));
            let (all, _) = s.render(&e, &p, i, &|_| true);
            let (some, _) = s.render(&e, &p, i, &|k| need.contains(&k));
            assert_eq!(all.data, some.data, "seed {seed} frame {i}: needed {need:?}");
        }
    }
}

#[test]
fn deterministic_across_runs_threads_and_caches() {
    let p = params(11);
    let run = |threads: usize| {
        let pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap();
        pool.install(|| {
            let mut s = Seq::new(80);
            let e = Engine::new();
            (0..80).map(|i| s.render(&e, &p, i, &|_| true).0.data).collect::<Vec<_>>()
        })
    };
    let a = run(1);
    assert_eq!(a, run(4));
    // a fresh engine (cold caches) rendering frames out of order gives the same frames
    let mut s = Seq::new(80);
    let e = Engine::new();
    for i in (0..80).rev().step_by(3) {
        assert_eq!(s.render(&e, &p, i, &|_| true).0.data, a[i], "frame {i}");
    }
}

#[test]
fn bottom_up_memory_and_in_place_give_the_same_picture() {
    let mut s = Seq::new(60);
    let e = Engine::new();
    let p = params(2);
    let i = 50;
    let (want, _) = s.render(&e, &p, i, &|_| true);
    // the same frames stored bottom-up, like Fusion: views start at the last row, stride < 0
    let flip = |f: &Frame| {
        let mut g = f.clone();
        for y in 0..H {
            g.data[4 * W * y..4 * W * (y + 1)].copy_from_slice(&f.data[4 * W * (H - 1 - y)..4 * W * (H - y)]);
        }
        g
    };
    let mut up: Vec<Frame> = s.frames.iter().map(flip).collect();
    let views: Vec<View> = up
        .iter_mut()
        .map(|f| unsafe { View::from_raw(f.data.as_mut_ptr().add(4 * W * (H - 1)), W, H, -(4 * W as isize)) }.unwrap())
        .collect();
    let back: Vec<Option<View>> = (1..=MAX_BACK).map(|k| Some(views[i - k])).collect();
    let mut out = Frame::new(W, H);
    let ov = unsafe { View::from_raw(out.data.as_mut_ptr().add(4 * W * (H - 1)), W, H, -(4 * W as isize)) }.unwrap();
    e.render(&p, i as i64, views[i], &back, ov);
    assert_eq!(flip(&out).data, want.data);
    // in place (out = cur): the effects must still read the clean frame
    let mut inplace = s.frames[i].clone();
    let v = inplace.view();
    let mut views: Vec<View> = s.frames.iter_mut().map(|f| f.view()).collect();
    let back: Vec<Option<View>> = (1..=MAX_BACK).map(|k| Some(views[i - k])).collect();
    let _ = views.remove(i);
    e.render(&p, i as i64, v, &back, v);
    assert_eq!(inplace.data, want.data);
}
