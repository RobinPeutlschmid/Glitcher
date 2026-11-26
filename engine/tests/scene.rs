//! 3D mode, body parts and plates, end to end through Engine::render_scene: the guarantees the
//! node relies on, checked on a synthetic figure.

use glitch_core::engine::{Engine, FLAG_3D, FLAG_PARTS, FLAG_PLATE, FLAG_ROLLED, Scene};
use glitch_core::image::Frame;
use glitch_core::layers::Mask;
use glitch_core::masks::{Landmarks, PartMask, Spill};
use glitch_core::sched::{EFFECTS, Params};

const W: usize = 360;
const H: usize = 420;

/// A standing figure (soft-edged capsules) and its landmarks; it sways with i.
fn figure(i: usize) -> (Frame, Landmarks) {
    let cx = 180.0 + 6.0 * ((i as f64) * 0.3).sin();
    let (top, s) = (30.0, 90.0);
    let mut lm = [[0.0, 0.0, 1.0]; 33];
    for (k, x, y) in [(0, 0.0, 0.3), (2, -0.1, 0.25), (5, 0.1, 0.25), (7, -0.25, 0.3), (8, 0.25, 0.3),
                      (9, -0.05, 0.4), (10, 0.05, 0.4), (11, -0.5, 0.8), (12, 0.5, 0.8), (13, -0.6, 1.4),
                      (14, 0.6, 1.4), (15, -0.65, 1.9), (16, 0.65, 1.9), (17, -0.7, 2.0), (18, 0.7, 2.0),
                      (19, -0.65, 2.05), (20, 0.65, 2.05), (21, -0.6, 2.0), (22, 0.6, 2.0), (23, -0.3, 2.0),
                      (24, 0.3, 2.0), (25, -0.3, 2.9), (26, 0.3, 2.9), (27, -0.3, 3.8), (28, 0.3, 3.8),
                      (29, -0.35, 3.9), (30, 0.35, 3.9), (31, -0.2, 3.95), (32, 0.2, 3.95)] {
        lm[k] = [cx + x * s, top + y * s, 1.0];
    }
    // the body: the union of the torso, head, arm and leg part masks, as alpha
    let body = PartMask::new(&lm, 127, W, H, 0.5);
    let mut f = Frame::new(W, H);
    for y in 0..H {
        for x in 0..W {
            let a = body.at(x, y);
            if a > 0.0 {
                let v = ((x * 7 + y * 3) % 23) as f32 / 23.0;
                let k = 4 * (y * W + x);
                f.data[k..k + 4].copy_from_slice(&[v * a, (0.3 + 0.4 * (y as f32 / H as f32)) * a, (1.0 - v) * a, a]);
            }
        }
    }
    (f, lm)
}

fn plate() -> Frame {
    let mut f = Frame::new(W, H);
    for y in 0..H {
        for x in 0..W {
            let k = 4 * (y * W + x);
            f.data[k..k + 4].copy_from_slice(&[((x / 20 + y / 20) % 2) as f32 * 0.6 + 0.2, 0.35, 0.5, 1.0]);
        }
    }
    f
}

fn loud(seed: u32) -> Params {
    let mut p = Params { seed, intensity: 1.0, bursts: 0.0, tint: 0.4, ..Default::default() };
    p.gains = [2.0; 13];
    p.sane()
}

/// Render frame i (with history) of the figure.
fn render(e: &Engine, p: &Params, i: usize, plate: Option<&mut Frame>, sc: &Scene) -> (Frame, glitch_core::engine::Stats) {
    let (mut cur, _) = figure(i);
    let mut back: Vec<Frame> = (1..=24).map(|k| figure(i.saturating_sub(k)).0).collect();
    let bv: Vec<_> = back.iter_mut().enumerate().map(|(k, f)| (k < i).then(|| f.view())).collect();
    let mut out = Frame::new(W, H);
    let pv = plate.map(|f| f.view());
    let st = e.render_scene(p, i as i64, cur.view(), &bv, out.view(), pv, sc);
    (out, st)
}

fn valid(f: &Frame) -> bool {
    f.data.chunks(4).all(|p| p.iter().all(|v| v.is_finite() && *v >= 0.0) && p[3] <= 1.0 + 1e-6
        && p[0].max(p[1]).max(p[2]) <= p[3] * 2.0 + 1e-4)
}

fn view3d(yaw: f64, pitch: f64, roll: f64) -> Option<[f64; 6]> {
    Some([yaw, pitch, roll, 3.0, 0.35, 0.25])
}

#[test]
fn three_d_renders_valid_deterministic_frames_unlike_2d() {
    let p = loud(5);
    let sc = Scene { view3d: view3d(35.0, 10.0, 0.0), ..Default::default() };
    let run = |threads: usize| {
        rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap().install(|| {
            let e = Engine::new();
            (0..30).map(|i| render(&e, &p, i, None, &sc).0.data).collect::<Vec<_>>()
        })
    };
    let a = run(1);
    assert_eq!(a, run(4));
    let e = Engine::new();
    let mut differs = 0;
    for i in 0..30 {
        let (f, st) = render(&e, &p, i, None, &sc);
        assert!(valid(&f), "frame {i}");
        assert!(st.flags & FLAG_3D != 0 && st.flags & FLAG_ROLLED == 0);
        let (flat, _) = render(&e, &p, i, None, &Scene::default());
        differs += (flat.data != f.data) as usize;
    }
    assert!(differs > 20, "{differs}");
    // planes in front and behind both happen: front ones change pixels over the body
    let any_glitch = (0..30).filter(|&i| render(&e, &p, i, None, &sc).0.data != figure(i).0.data).count();
    assert!(any_glitch > 20);
}

#[test]
fn roll_leaves_untouched_pixels_exact() {
    // with the plane effects off, a rolled 3D frame must come out exactly as it went in: only
    // what the glitch changes is rotated back
    let mut p = loud(2);
    p.gains = [0.0; 13];
    let p = p.sane();
    let e = Engine::new();
    let sc = Scene { view3d: view3d(20.0, 0.0, 25.0), ..Default::default() };
    for i in [3usize, 11] {
        let (f, st) = render(&e, &p, i, None, &sc);
        assert!(st.flags & FLAG_ROLLED != 0);
        assert_eq!(f.data, figure(i).0.data);
    }
    // and with them on: valid, and about as much changes as without the roll (the rotation
    // back resamples only the changed areas, it doesn't smear the rest)
    let p = loud(2);
    let flat = Scene { view3d: view3d(20.0, 0.0, 0.0), ..Default::default() };
    let (mut changed, mut changed_flat) = (0, 0);
    for i in 0..20 {
        let (f, _) = render(&e, &p, i, None, &sc);
        assert!(valid(&f));
        let src = figure(i).0;
        changed += f.data.chunks(4).zip(src.data.chunks(4)).filter(|(a, b)| a != b).count();
        let (g, _) = render(&e, &p, i, None, &flat);
        changed_flat += g.data.chunks(4).zip(src.data.chunks(4)).filter(|(a, b)| a != b).count();
    }
    let r = changed as f64 / changed_flat as f64;
    eprintln!("changed rolled {changed} vs upright {changed_flat}");
    assert!(r > 0.6 && r < 1.6, "{changed} vs {changed_flat}");
}

#[test]
fn protect_and_only_keep_their_areas_clean() {
    let p = loud(8);
    let e = Engine::new();
    for (mode, view) in [(0, None), (1, view3d(30.0, 5.0, 0.0)), (2, view3d(30.0, 5.0, 15.0))] {
        for i in (2..40).step_by(3) {
            let (src, lm) = figure(i);
            // protect the head: wherever its mask is fully on, the pixel is the source's
            let sc = Scene { view3d: view, lm: Some(lm), protect: 2, ..Default::default() };
            let (f, st) = render(&e, &p, i, None, &sc);
            assert!(st.flags & FLAG_PARTS != 0);
            let head = PartMask::new(&lm, 2, W, H, 1.0);
            let only_legs = PartMask::new(&lm, 32, W, H, 1.0);
            for y in 0..H {
                for x in 0..W {
                    let k = 4 * (y * W + x);
                    if head.at(x, y) >= 0.999 {
                        assert_eq!(&f.data[k..k + 4], &src.data[k..k + 4], "mode {mode} frame {i} ({x},{y})");
                    }
                }
            }
            // only the legs: outside their mask nothing changes
            let sc = Scene { view3d: view, lm: Some(lm), only: 32, ..Default::default() };
            let (f, _) = render(&e, &p, i, None, &sc);
            for y in 0..H {
                for x in 0..W {
                    let k = 4 * (y * W + x);
                    if only_legs.at(x, y) <= 0.001 {
                        assert_eq!(&f.data[k..k + 4], &src.data[k..k + 4], "mode {mode} frame {i} ({x},{y})");
                    }
                }
            }
        }
    }
}

#[test]
fn a_target_limits_one_effect() {
    // only jpeg on, targeted at the torso: everything off the torso mask stays the source
    let mut p = Params { seed: 4, intensity: 1.0, bursts: 0.0, ..Default::default() };
    p.gains = [0.0; 13];
    p.gains[0] = 3.0;
    let p = p.sane();
    let e = Engine::new();
    let jpeg = EFFECTS.iter().position(|f| *f == "jpeg").unwrap();
    let mut hit = 0;
    for i in 0..40 {
        let (src, lm) = figure(i);
        let mut targets = [0u32; 13];
        targets[jpeg] = 4;
        let sc = Scene { lm: Some(lm), targets, ..Default::default() };
        let (f, _) = render(&e, &p, i, None, &sc);
        let torso = PartMask::new(&lm, 4, W, H, 1.0);
        for y in 0..H {
            for x in 0..W {
                let k = 4 * (y * W + x);
                if torso.at(x, y) <= 0.001 {
                    assert_eq!(&f.data[k..k + 4], &src.data[k..k + 4]);
                }
            }
        }
        hit += (f.data != src.data) as usize;
    }
    assert!(hit > 5, "{hit}");
}

#[test]
fn plates_composite_and_spill_stays_near_the_subject() {
    let p = loud(3);
    let e = Engine::new();
    let mut pl = plate();
    for (view, spill) in [(None, 0.0), (None, 0.3), (view3d(40.0, 0.0, 0.0), 0.0), (view3d(40.0, 0.0, 20.0), 0.3)] {
        let mut damaged_plate = 0;
        for i in (1..40).step_by(3) {
            let (src, _) = figure(i);
            let sc = Scene { view3d: view, spill, ..Default::default() };
            let (f, st) = render(&e, &p, i, Some(&mut pl), &sc);
            assert!(st.flags & FLAG_PLATE != 0);
            let mut sv = src.clone();
            let sp = Spill::new(sv.view(), spill, st.bbox.unwrap());
            for y in 0..H {
                for x in 0..W {
                    let k = 4 * (y * W + x);
                    let o = &f.data[k..k + 4];
                    assert!((o[3] - 1.0).abs() < 1e-5, "not opaque at ({x},{y}): {o:?}");
                    let a = src.data[k + 3];
                    let clean: Vec<f32> = (0..4).map(|c| src.data[k + c] + pl.data[k + c] * (1.0 - a)).collect();
                    if sp.at(x, y) <= 0.001 {
                        assert_eq!(o, &clean[..], "spill {spill}: plate damaged at ({x},{y})");
                    } else if src.data[k + 3] == 0.0 && o != &pl.data[k..k + 4] {
                        damaged_plate += 1;
                    }
                }
            }
        }
        if spill > 0.0 {
            assert!(damaged_plate > 0, "{view:?}: spill {spill} never touched the plate");
        }
    }
    // an empty frame is the clean plate
    let mut empty = Frame::new(W, H);
    let mut out = Frame::new(W, H);
    let st = e.render_scene(&p, 5, empty.view(), &[], out.view(), Some(pl.view()), &Scene::default());
    assert!(st.flags & FLAG_PLATE != 0);
    assert_eq!(out.data, pl.data);
}

#[test]
fn rendering_in_place_matches_separate_output() {
    let p = loud(6);
    let e = Engine::new();
    let sc = Scene { view3d: view3d(25.0, 10.0, 12.0), ..Default::default() };
    for i in [4usize, 9] {
        let (sep, _) = render(&e, &p, i, None, &sc);
        let (mut cur, _) = figure(i);
        let mut back: Vec<Frame> = (1..=24).map(|k| figure(i.saturating_sub(k)).0).collect();
        let bv: Vec<_> = back.iter_mut().enumerate().map(|(k, f)| (k < i).then(|| f.view())).collect();
        let v = cur.view();
        e.render_scene(&p, i as i64, v, &bv, v, None, &sc);
        assert_eq!(cur.data, sep.data);
    }
}

#[test]
fn appear_and_vanish() {
    use glitch_core::reveal::Reveal;
    let p = loud(5);
    let e = Engine::new();
    let mut pl = plate();
    // hidden: the clean plate, or nothing
    let hidden = Scene { reveal: Reveal::Hidden, ..Default::default() };
    assert_eq!(render(&e, &p, 9, Some(&mut pl), &hidden).0.data, pl.data);
    assert!(render(&e, &p, 9, None, &hidden).0.data.iter().all(|&v| v == 0.0));
    // visible again after a transition: the same as never having had one
    let a = render(&e, &p, 9, None, &Scene::default()).0;
    let b = render(&e, &p, 9, None, &Scene { reveal: Reveal::Visible, ..Default::default() }).0;
    assert_eq!(a.data, b.data);
    // mid transition: valid, partly there; appear grows with progress; on a plate opaque
    let cover = |f: &Frame| f.data.chunks(4).map(|q| q[3] as f64).sum::<f64>();
    let full = cover(&figure(9).0);
    let mut last = 0.0;
    for q in [0.1, 0.4, 0.7] {
        let sc = Scene { reveal: Reveal::Moving(true, q, 0), ..Default::default() };
        let (f, _) = render(&e, &p, 9, None, &sc);
        assert!(valid(&f));
        let c = cover(&f);
        assert!(c > last && c < 1.3 * full, "q {q}: {c} vs {full}");
        last = c;
        let (g, _) = render(&e, &p, 9, Some(&mut pl), &sc);
        assert!(g.data.chunks(4).all(|q| (q[3] - 1.0).abs() < 1e-5));
    }
    // vanishing at the very end: almost nothing left
    let (f, _) = render(&e, &p, 9, None, &Scene { reveal: Reveal::Moving(false, 1.0, 1), ..Default::default() });
    assert!(cover(&f) < 0.1 * full, "{} vs {full}", cover(&f));
}
