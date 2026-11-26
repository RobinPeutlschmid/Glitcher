# What Resolve's Fusion gives a Fuse (measured with GlitcherProbe.fuse)

Resolve on Windows 11, RTX 4060 laptop (16 threads), 2026-11-25. Most of this isn't in the
Fuse SDK. It was found by probing, so re-run the probe after a Resolve update.

- **Lua:** LuaJIT 2.1.0-beta3, with the FFI. `ffi.load(path)` needs the full path, without
  quotes (Explorer's "Copy as path" adds them).
- **Image objects** are FFI `cdata<struct Image *>`. They have `GetScanLine`, `GetChanSize`,
  `ProcessPixels` and `MultiProcessPixels`. `GetData()` returns nil.
- **`img:GetScanLine(y)`** returns `unsigned char **`, a **table of row pointers**. Row `y` is
  `p[y]`, and it is the same table whichever `y` you pass.
- **Pixel layout:** in the clips tried, rows are contiguous with no padding (1920 px →
  30720 bytes), with the **bottom row first** (y = 0 at the lowest address; Fusion's origin is
  bottom-left). The whole frame is therefore one block, starting at `p[0]`, with a stride of
  `4 * width` floats.
- **Pixel format:** float32 RGBA, in R, G, B, A order (memory matched `GetPixel`).
  - `img.Depth` was 8 for these clips. An earlier clip reported 7, which is probably 16-bit
    float (not confirmed). The real node must check its bytes per pixel and handle 16-bit
    float (or ask Fusion for float32).
  - `GetChanSize(chan)` needs a channel argument (`CHAN_RED` = 0 … `CHAN_ALPHA` = 3) and
    returns the bytes per channel (4).
- **Earlier frames:** `InImage:GetSource(req.Time - 12)` returned a full frame in 0.2 ms
  (cached upstream) and 14 ms the first time.
- **Speed at 1920×1080:** the spike effect (echo + grain, 16 threads) took 4.2 ms on the real
  pixels in place. `CopyOf()` for the output took 3.4 ms, and the whole node per frame took
  8.6 ms.
  - The real node should create its output with `Image({IMG_Like = img})`, which allocates
    without copying, since the engine writes every pixel anyway.
  - Where the rows aren't one contiguous block, it should fall back to row by row.

## Pixel format changes with what Fusion is doing (debug log, 2026-11-25)

- **Scrubbing or paused:** frames arrive as `Depth 8`, 4-byte channels (float32).
- **Playback:** frames are rendered fresh as `Depth 7`, 2-byte channels, for the current
  frame *and* for `GetSource` of earlier frames. That is float16: the node checked it against
  `GetPixel` (2-byte channels could also be uint16).
- **Mixed:** formats mix within one frame. The current frame can be float32 (cached) while
  its history is float16 (freshly rendered), so the node checks each image on its own.
- **Requests** were all flagged `Interactive`, in playback and when scrubbing.
