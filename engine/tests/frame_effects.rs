//! The frame effects, one by one and together, on a synthetic subject.

use glitch_core::engine::Engine;
use glitch_core::image::{Frame, boxes};
use glitch_core::sched::{Ctx, EFFECTS, Params};

const W: usize = 320;
const H: usize = 200;

fn subject(i: usize, opaque_bg: bool) -> Frame {
    let mut f = Frame::new(W, H);
    let cx = 120.0 + 2.0 * i as f32;
    for y in 0..H {
        for x in 0..W {
            let (dx, dy) = ((x as f32 - cx) / 45.0, (y as f32 - 100.0) / 80.0);
            let a = if opaque_bg { 1.0 } else { ((1.0 - dx * dx - dy * dy) * 6.0).clamp(0.0, 1.0) };
            if a > 0.0 {
                let v = ((x * 13 + y * 7) % 31) as f32 / 31.0;
                let k = 4 * (y * W + x);
                f.data[k..k + 4].copy_from_slice(&[v * a, (0.3 + 0.4 * (y as f32 / H as f32)) * a, (1.0 - v) * 0.8 * a, a]);
            }
        }
    }
    f
}

fn only(fx: usize, seed: u32) -> Params {
    let mut p = Params { seed, intensity: 0.9, bursts: 0.1, tint: 0.6, ..Default::default() };
    p.gains = [0.0; 13];
    p.gains[fx] = 2.5;
    p.sane()
}

fn render(e: &Engine, p: &Params, i: usize, src: &mut Frame) -> Frame {
    let mut out = Frame::new(W, H);
    e.render(p, i as i64, src.view(), &[], out.view());
    out
}

fn assert_valid(f: &Frame, what: &str) {
    for px in f.data.chunks(4) {
        assert!(px[3] >= 0.0 && px[3] <= 1.0 + 1e-5, "{what}: alpha {px:?}");
        assert!(px[..3].iter().all(|&c| c >= 0.0 && c <= px[3] + 1e-4), "{what}: not premultiplied {px:?}");
    }
}

#[test]
fn every_frame_effect_changes_the_picture_and_stays_valid() {
    let e = Engine::new();
    #[allow(clippy::needless_range_loop)]
    for fx in 0..10 {
        let mut changed = 0;
        for seed in 1..4u32 {
            let p = only(fx, seed);
            for i in (0..120).step_by(3) {
                let mut src = subject(i, false);
                let out = render(&e, &p, i, &mut src);
                assert_valid(&out, EFFECTS[fx]);
                changed += (out.data != src.data) as usize;
            }
        }
        assert!(changed > 3, "{}: changed only {changed} frames", EFFECTS[fx]);
    }
}

#[test]
fn content_tracking_and_dirty_areas_never_miss_pixels() {
    // every effect must keep the tracked content area around everything non-transparent (the
    // next effects only look inside it), and report every pixel it changed as dirty (3D layering
    // and part masks only revisit those) - in 2-D and for both 3-D passes
    use glitch_core::fx_frame::{Area, run_one};
    use glitch_core::sched::{Layer, Space3D};
    for seed in 1..6u32 {
        let mut p = Params { seed, intensity: 1.0, bursts: 0.0, tint: 0.5, ..Default::default() };
        p.gains = [2.5; 13];
        let p = p.sane();
        for mode in 0..3 {
            for i in (0..90).step_by(4) {
                let mut f = subject(i, false);
                let v = f.view();
                let (lb, cbox) = boxes(&v, 8.0 / 255.0);
                let (x0, y0, x1, y1) = lb.unwrap();
                let mut c = Ctx::new(p.clone(), i as f64 / 25.0, i as i64, (x0, y0, x1, y1));
                if mode > 0 {
                    c = c.with_space(Some(Space3D::new(0.5 + 0.02 * i as f64, 2.5, 0.35, 0.3, 0.0, 0.5)))
                        .layered(if mode == 1 { Layer::Front } else { Layer::Back });
                }
                let mut area = Area::new(cbox);
                for (fx, name) in EFFECTS.iter().enumerate().take(10) {
                    let before = f.data.clone();
                    area.dirty.clear();
                    let v = f.view();
                    run_one(fx, &c, &v, &mut area);
                    let real = boxes(&v, 0.0).1;
                    let cb = area.cbox;
                    if !real.is_empty() {
                        assert!(real.y0 >= cb.y0 && real.y1 <= cb.y1 && real.x0 >= cb.x0 && real.x1 <= cb.x1,
                                "seed {seed} mode {mode} frame {i}: after {name} content {real:?} outside tracked {cb:?}");
                    }
                    for (k, (a, b)) in before.chunks(4).zip(f.data.chunks(4)).enumerate() {
                        if a != b {
                            let (x, y) = (k % W, k / W);
                            assert!(area.dirty.iter().any(|r| y >= r.y0 && y < r.y1 && x >= r.x0 && x < r.x1),
                                    "seed {seed} mode {mode} frame {i}: {name} changed ({x}, {y}) outside its dirty areas");
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn all_effects_together_deterministic_across_threads() {
    let mut p = Params { seed: 9, intensity: 0.85, tint: 0.3, ..Default::default() };
    p.gains = [1.5; 13];
    let p = p.sane();
    let run = |threads: usize| {
        rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap().install(|| {
            let e = Engine::new();
            (0..60).map(|i| render(&e, &p, i, &mut subject(i, false)).data).collect::<Vec<_>>()
        })
    };
    let a = run(1);
    assert_eq!(a, run(3));
    assert!(a.iter().enumerate().filter(|(i, f)| **f != subject(*i, false).data).count() > 30);
}

#[test]
fn edge_cases_dont_break() {
    let e = Engine::new();
    let mut p = Params { seed: 4, intensity: 1.0, tint: 0.5, ..Default::default() };
    p.gains = [3.0; 13];
    let p = p.sane();
    // empty frame: stays empty
    let mut empty = Frame::new(W, H);
    assert!(render(&e, &p, 10, &mut empty).data.iter().all(|&v| v == 0.0));
    // full opaque frame (no alpha anywhere) and tiny frames
    for i in 0..30 {
        assert_valid(&render(&e, &p, i, &mut subject(i, true)), "opaque");
    }
    for (w, h) in [(1usize, 1usize), (3, 2), (9, 17), (64, 8)] {
        let mut f = Frame::new(w, h);
        f.data.iter_mut().enumerate().for_each(|(k, v)| *v = if k % 4 == 3 { 1.0 } else { 0.5 });
        let mut out = Frame::new(w, h);
        for i in 0..20 {
            e.render(&p, i, f.view(), &[], out.view());
        }
    }
    // HDR values outside the glitched areas survive (no clamp to 1)
    let mut hdr = subject(5, false);
    for v in hdr.data.chunks_mut(4) {
        if v[3] >= 1.0 {
            v[0] = 3.0;
        }
    }
    let mut calm = Params { seed: 1, intensity: 0.2, ..Default::default() }.sane();
    calm.gains = [0.0; 13];
    let out = render(&e, &calm, 5, &mut hdr);
    assert!(out.data.chunks(4).any(|v| v[0] == 3.0));
}
