//! The C interface as a host uses it: bottom-up rows, mixed pixel formats (Resolve hands a node
//! float16 during playback and float32 when scrubbing), bad arguments.

use glitch_core::pixfmt::{f16_to_f32, f32_to_f16};
use glitch_core::sched::{ECHO, MOSH, STUTTER};
use glitch_core::*;
use std::ffi::c_void;

const W: usize = 200;
const H: usize = 120;

fn frame(i: usize) -> Vec<f32> {
    let mut f = vec![0f32; 4 * W * H];
    let cx = 60.0 + 3.0 * i as f32;
    for y in 10..110 {
        for x in 0..W {
            let a = (1.0 - ((x as f32 - cx) / 25.0).powi(2)).clamp(0.0, 1.0);
            let v = ((x * 5 + y * 3) % 17) as f32 / 17.0;
            let k = 4 * (y * W + x);
            f[k..k + 4].copy_from_slice(&[v * a, 0.5 * a, (1.0 - v) * a, a]);
        }
    }
    f
}

/// Top-down rows -> bottom-up memory (Fusion's layout).
fn flip<T: Copy>(v: &[T]) -> Vec<T> {
    let mut o = v.to_vec();
    for y in 0..H {
        o[4 * W * y..4 * W * (y + 1)].copy_from_slice(&v[4 * W * (H - 1 - y)..4 * W * (H - y)]);
    }
    o
}

fn img<T>(v: &mut [T], format: i32) -> GcImage {
    // pointer to the top row = the last row in memory; stride negative, in channel values
    let top = unsafe { v.as_mut_ptr().add(4 * W * (H - 1)) } as *mut c_void;
    GcImage { ptr: top, width: W as i32, height: H as i32, stride: -(4 * W as i64), format, _reserved: 0 }
}

fn params() -> GcParams {
    let mut g = [0.0; 13];
    g[ECHO] = 3.0;
    g[STUTTER] = 3.0;
    g[MOSH] = 3.0;
    GcParams { seed: 6, intensity: 1.0, bursts: 0.0, gains: g, ..Default::default() }
}

#[test]
fn float16_playback_frames_render_like_float32() {
    let e = gc_engine_new();
    let p = params();
    let frames: Vec<Vec<f32>> = (0..40).map(frame).collect();
    let mut f32s: Vec<Vec<f32>> = frames.iter().map(|f| flip(f)).collect();
    let mut f16s: Vec<Vec<u16>> = f32s.iter().map(|f| f.iter().map(|&v| f32_to_f16(v)).collect()).collect();
    let mut max_err = 0f32;
    let mut glitched = 0;
    for i in 0..40usize {
        let render = |cur: GcImage, back: &[GcImage], out: GcImage| unsafe {
            let mut st = GcStats::default();
            assert_eq!(gc_render(e, &p, i as i64, &cur, back.as_ptr(), std::ptr::null(), &out, std::ptr::null(), &mut st), 0);
            st
        };
        // all float32
        let back32: Vec<GcImage> = (1..=24).map(|k| if k <= i { img(&mut f32s[i - k], 0) } else { GcImage { ptr: std::ptr::null_mut(), width: 0, height: 0, stride: 0, format: 0, _reserved: 0 } }).collect();
        let mut out32 = vec![0f32; 4 * W * H];
        let cur32 = img(&mut f32s[i], 0);
        let st = render(cur32, &back32, img(&mut out32, 0));
        glitched += (st.echo + st.stutter + st.mosh > 0) as usize;
        // all float16, and a mix (current float32, history float16) like the log showed
        let back16: Vec<GcImage> = (1..=24).map(|k| if k <= i { img(&mut f16s[i - k], 1) } else { GcImage { ptr: std::ptr::null_mut(), width: 0, height: 0, stride: 0, format: 1, _reserved: 0 } }).collect();
        let mut out16 = vec![0u16; 4 * W * H];
        let cur16 = img(&mut f16s[i], 1);
        render(cur16, &back16, img(&mut out16, 1));
        let mut outmix = vec![0f32; 4 * W * H];
        render(cur32, &back16, img(&mut outmix, 0));
        for ((a, b), c) in out32.iter().zip(out16.iter()).zip(outmix.iter()) {
            max_err = max_err.max((a - f16_to_f32(*b)).abs()).max((a - c).abs());
        }
    }
    unsafe { gc_engine_free(e) };
    assert!(glitched > 10, "only {glitched} frames glitched");
    assert!(max_err < 0.01, "float16 result differs by {max_err}");
}

#[test]
fn bad_arguments_are_refused_not_crashed() {
    let e = gc_engine_new();
    let mut p = params();
    let mut a = flip(&frame(0));
    let mut o = vec![0f32; 4 * W * H];
    let cur = img(&mut a, 0);
    let mut st = GcStats::default();
    unsafe {
        let mut bad = img(&mut o, 7);                                  // unknown format
        assert_eq!(gc_render(e, &p, 0, &cur, std::ptr::null(), std::ptr::null(), &bad, std::ptr::null(), &mut st), -1);
        bad.format = 0;
        bad.stride = 3;                                                // stride shorter than a row
        assert_eq!(gc_render(e, &p, 0, &cur, std::ptr::null(), std::ptr::null(), &bad, std::ptr::null(), &mut st), -1);
        let mut small = img(&mut o, 0);
        small.width = 100;                                             // size mismatch
        assert_eq!(gc_render(e, &p, 0, &cur, std::ptr::null(), std::ptr::null(), &small, std::ptr::null(), &mut st), -1);
        p.abi = 2;                                                     // an older host
        assert_eq!(gc_render(e, &p, 0, &cur, std::ptr::null(), std::ptr::null(), &img(&mut o, 0), std::ptr::null(), &mut st), -3);
        assert_eq!(gc_render(std::ptr::null(), &p, 0, &cur, std::ptr::null(), std::ptr::null(), &img(&mut o, 0), std::ptr::null(), &mut st), -1);
        gc_engine_free(e);
    }
}

#[test]
fn stats_report_the_angles_used_and_the_overlay_draws() {
    let dir = std::env::temp_dir().join(format!("gc_view_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let pose = dir.join("pose.csv");
    std::fs::write(&pose, "frame,yaw,pitch,roll\n0,0,10,0\n10,100,10,20\n").unwrap();
    let pc = std::ffi::CString::new(pose.to_str().unwrap()).unwrap();
    let e = gc_engine_new();
    let mut f = flip(&frame(3));
    let cur = img(&mut f, 0);
    let render = |p: &GcParams| {
        let mut o = vec![0f32; 4 * W * H];
        let mut st = GcStats::default();
        assert_eq!(unsafe { gc_render(e, p, 5, &cur, std::ptr::null(), std::ptr::null(), &img(&mut o, 0), std::ptr::null(), &mut st) }, 0);
        (o, st)
    };
    // 3D with the pose file and a slider offset: frame 5 is halfway, yaw 50 + 15, roll 10
    let p = GcParams { mode3d: 1, yaw: 15.0, pose_path: pc.as_ptr(), gains: [0.0; 13], ..Default::default() };
    let (plain, st) = render(&p);
    assert_eq!(st.has_view, 1);
    assert!((st.view[0] - 65.0).abs() < 1e-6 && (st.view[1] - 10.0).abs() < 1e-6 && (st.view[2] - 10.0).abs() < 1e-6,
            "{:?}", st.view);
    assert_eq!(st.warn, 0);
    // without 3D: no view
    let (_, st2) = render(&GcParams { gains: [0.0; 13], ..Default::default() });
    assert_eq!(st2.has_view, 0);
    // Show pose: the same frame with the ring, arrow and text drawn over it
    let (shown, _) = render(&GcParams { overlay: 1, ..p });
    let drawn = plain.chunks(4).zip(shown.chunks(4)).filter(|(a, b)| a != b).count();
    assert!(drawn > 300, "{drawn}");
    unsafe { gc_engine_free(e) };
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn layout_box_and_hits() {
    let e = gc_engine_new();
    let mut f = flip(&frame(3));
    let cur = img(&mut f, 0);
    let render = |p: &GcParams, fr: i64| {
        let mut o = vec![0f32; 4 * W * H];
        let mut st = GcStats::default();
        assert_eq!(unsafe { gc_render(e, p, fr, &cur, std::ptr::null(), std::ptr::null(), &img(&mut o, 0), std::ptr::null(), &mut st) }, 0);
        (o, st)
    };
    // a given layout box is used as is, then grown by 10 % of its size each side
    let p = GcParams { layout_mode: 1, layout: [20.0, 10.0, 180.0, 110.0], box_grow: 0.1, ..Default::default() };
    let (_, st) = render(&p, 4);
    assert_eq!(st.bbox, [4, 0, 196, 120]);
    let (_, st) = render(&GcParams::default(), 4);
    assert!(st.bbox[2] - st.bbox[0] < 120);                 // this frame's subject: narrower
    // a hit: full strength on its frame, gone after its length, more glitch while it lasts
    let hits = std::ffi::CString::new("20:1:0.4").unwrap();
    let p = GcParams { intensity: 0.15, hits: hits.as_ptr(), ..Default::default() };
    let (calm, s0) = render(&GcParams { intensity: 0.15, ..Default::default() }, 20);
    let (hit, s1) = render(&p, 20);
    assert_eq!(s0.hit, 0.0);
    assert!((s1.hit - 1.0).abs() < 1e-9);
    assert_eq!(render(&p, 40).1.hit, 0.0);
    let changed = |o: &Vec<f32>| o.iter().zip(f.iter()).filter(|(a, b)| a != b).count();
    assert!(changed(&hit) > changed(&calm), "{} vs {}", changed(&hit), changed(&calm));
    // a malformed list is ignored, with a message
    let bad = std::ffi::CString::new("20:x").unwrap();
    let (_, st) = render(&GcParams { hits: bad.as_ptr(), ..Default::default() }, 20);
    assert_eq!(st.hit, 0.0);
    assert!(unsafe { std::ffi::CStr::from_ptr(gc_last_error(e)) }.to_str().unwrap().contains("20:x"));
    unsafe { gc_engine_free(e) };
}


#[test]
fn glitch_matte_marks_what_the_glitch_made() {
    let e = gc_engine_new();
    let mut f = flip(&frame(3));
    let cur = img(&mut f, 0);
    let p = GcParams { seed: 6, intensity: 1.0, bursts: 0.0, ..Default::default() };
    let mut o = vec![0f32; 4 * W * H];
    let mut m = vec![9f32; 4 * W * H];
    let mut st = GcStats::default();
    assert_eq!(unsafe { gc_render(e, &p, 4, &cur, std::ptr::null(), std::ptr::null(), &img(&mut o, 0), &img(&mut m, 0), &mut st) }, 0);
    // R = G = B = A, 0..1; white only where the output differs from the source
    for (k, px) in m.chunks(4).enumerate() {
        assert!(px[0] == px[1] && px[1] == px[2] && px[2] == px[3] && (0.0..=1.0).contains(&px[0]));
        let diff = (0..4).map(|c| (o[4 * k + c] - f[4 * k + c]).abs()).fold(0.0, f32::max);
        if diff * 255.0 < 1.0 {
            assert_eq!(px[0], 0.0, "untouched pixel {k} marked");
        }
    }
    let lit = m.chunks(4).filter(|px| px[0] > 0.5).count();
    assert!(lit > 50, "{lit}");
    // the effects off: an all-black matte
    let calm = GcParams { gains: [0.0; 13], ..p };
    assert_eq!(unsafe { gc_render(e, &calm, 4, &cur, std::ptr::null(), std::ptr::null(), &img(&mut o, 0), &img(&mut m, 0), &mut st) }, 0);
    assert!(m.iter().all(|&v| v == 0.0));
    unsafe { gc_engine_free(e) };
}
