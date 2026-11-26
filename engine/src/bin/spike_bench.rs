//! cargo run --release --bin spike_bench [WIDTH HEIGHT]
//! Times the spike effect and the row copy on a synthetic frame (default 3840x2160).

use glitch_core::{gc_copy_rows, gc_spike, gc_threads};

fn main() {
    let a: Vec<usize> = std::env::args().skip(1).filter_map(|s| s.parse().ok()).collect();
    let (w, h) = if a.len() >= 2 { (a[0], a[1]) } else { (3840, 2160) };
    let st = 4 * w;
    let mut cur = vec![0f32; st * h];
    for y in 0..h {                                           // a soft-edged "person" in the middle
        for x in 0..w {
            let (dx, dy) = ((x as f32 - w as f32 * 0.5) / (w as f32 * 0.15), (y as f32 - h as f32 * 0.5) / (h as f32 * 0.42));
            let al = ((1.0 - dx * dx - dy * dy) * 20.0).clamp(0.0, 1.0);
            let i = y * st + 4 * x;
            cur[i..i + 4].copy_from_slice(&[0.6 * al, 0.45 * al, 0.35 * al, al]);
        }
    }
    let prev = cur.clone();
    let mut out = vec![0f32; st * h];
    let mut tmp = vec![0f32; st * h];
    let run = |f: &mut dyn FnMut() -> f64| { f(); let mut v: Vec<f64> = (0..15).map(|_| f()).collect(); v.sort_by(f64::total_cmp); v[7] };
    let fx = run(&mut || unsafe { gc_spike(cur.as_ptr(), prev.as_ptr(), out.as_mut_ptr(), w as i32, h as i32, st as i64, 3, 7, 1.0, 40) });
    let cp = run(&mut || unsafe { gc_copy_rows(cur.as_ptr(), st as i64, tmp.as_mut_ptr(), st as i64, h as i32, st as i64) });
    println!("{w}x{h}, {} threads: echo + grain {fx:.1} ms, copying one frame {cp:.1} ms ({:.0} MB)",
             gc_threads(), (st * h * 4) as f64 / 1e6);
}
