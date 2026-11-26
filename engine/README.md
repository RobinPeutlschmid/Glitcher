# glitch_core

Glitcher's native engine (Rust). The Fusion node (`../fusion/Glitcher.fuse`), Python
(`../glitch_native.py`, through ctypes) and later an OpenFX plugin all call it through one C
interface (`src/lib.rs`).

**Ported so far:**
- the scheduler (activity, events, strength), from glitch.py;
- all 13 effects in their 2-D form (`fx_time.rs`: echo, stutter, mosh; `fx_frame.rs`: the
  other ten).

Still in glitch.py only: 3-D mode, body parts, plates and reveals.

```
cargo test --release              # unit + integration tests
cargo build --release             # -> target/release/glitch_core.dll / libglitch_core.dylib / .so
cargo run --release --bin spike_bench          # the first spike's speed at 3840x2160
python ../tools/compare_native.py              # side by side with glitch.py: stats + contact sheet
luajit ../fusion/test_glitcher_mock.lua ../fusion/Glitcher.fuse target/release/libglitch_core.so
```

## How it differs from glitch.py

**Pixels:**
- Float32 RGBA, **premultiplied**, nominally 0..1, which is what Fusion gives a node.
  Compositing ("under", "over") is then one multiply-add per channel.
- Images come as a pointer to the top row plus a stride. The stride is negative when rows are
  stored bottom-up, as Fusion stores them.

**Randomness:**
- It comes from hashing integers, not from numpy's generators. So the same seed gives
  different events than in glitch.py, but the same statistics: `tools/compare_native.py`
  measures how often each effect shows, how much of the subject it covers, and how strongly.
- Nothing in the scheduler or pixel code uses the platform's maths library (`exp`, `ln`,
  `sin` …). Each platform rounds those its own way. The engine uses only + − × ÷, sqrt and
  floor, plus its own `exp` built from them (`src/detmath.rs`), so **every OS and CPU renders
  the same frames**.

**Earlier frames:**
- The host asks `gc_needed` which earlier frames this frame will read, and fetches only
  those. A test checks that rendering with only them equals rendering with all 24.
- The motion between frames is cached by a fingerprint of each frame's content, not by memory
  address (hosts reuse memory).

**Mosh:**
- glitch.py replays the held frame through every motion step: up to 16 passes over a large
  area. The engine instead follows each output pixel's source back through the steps and
  reads the held frame once. That's ~4x faster at 4K, and identical to the literal replay
  (tested).

**Safety:** no panic crosses the C boundary. Every function catches panics and returns an
error code, because a crash would take Resolve down with it.
