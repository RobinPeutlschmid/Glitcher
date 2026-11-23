"""Sound-reactive mode: decoding and beat / loudness analysis on a track with known onsets."""
import numpy as np
import pytest

import glitch
import synth
from conftest import skip_or_fail

CLICKS = (0.5, 1.0, 1.5, 2.0)
FPS = 25.0


def frame_times(dur=2.6):
    return np.arange(int(dur * FPS)) / FPS


def test_wav_decodes_at_every_bit_depth(tmp_path):
    ref = None
    for width in (2, 3, 4):
        x, sr = glitch.load_audio(synth.make_clicks_wav(tmp_path / f"c{width}.wav", width=width))
        assert sr == 22050 and x.dtype == np.float32 and np.abs(x).max() <= 1.0
        if ref is None:
            ref = x
        else:
            assert len(x) == len(ref) and np.abs(x - ref).max() < 1e-3


def test_beats_found_where_the_clicks_are(clicks_wav):
    loud, hit, beats = glitch.audio_curves(clicks_wav, frame_times(), sens=0.5, fps=FPS)
    assert len(beats) == len(CLICKS), beats
    assert np.abs(np.asarray(beats) - np.asarray(CLICKS)).max() < 0.05
    for t in CLICKS:                                       # the frame holding the click is hit
        assert hit[int(t * FPS)] > 0.4
    assert hit[int(0.3 * FPS)] == 0 and loud[int(0.3 * FPS)] < 0.05     # silence stays calm


def test_passage_limits_the_analysis(clicks_wav):
    loud, hit, beats = glitch.audio_curves(clicks_wav, frame_times(), start=0.8, end=1.7, fps=FPS)
    assert len(beats) == 2 and 0.95 < beats[0] < 1.05 and 1.45 < beats[1] < 1.55
    outside = (frame_times() < 0.8) | (frame_times() >= 1.7)
    assert hit[outside].max() == 0 and loud[outside].max() == 0


def test_sensitivity_is_monotonic(tmp_path):
    # quiet clicks between the loud ones: higher sensitivity must find at least as many beats
    clicks = (0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.1)
    amps = (0.9, 0.01, 0.9, 0.01, 0.9, 0.01, 0.9)     # 3 clicks ~40 dB quieter
    p = synth.make_clicks_wav(tmp_path / "mixed.wav", clicks=clicks, amps=amps)
    n = [len(glitch.audio_curves(p, frame_times(), sens=s, fps=FPS)[2]) for s in (0.0, 0.5, 1.0)]
    assert n == sorted(n) and n[0] == 4 and n[-1] == 7, n   # 0: loud ones only, 1: all


def test_audio_drives_the_render(frames, clicks_wav, tmp_path):
    out = tmp_path / "a"
    assert glitch.main([str(frames), "-o", str(out), "-s", "7", "-j", "1", "--fps", "5",
                        "--audio", str(clicks_wav), "--audio-strength", "1"]) == 0
    assert len(list(out.glob("*.png"))) == 12


def test_float_wav_needs_miniaudio(tmp_path):
    try:
        import miniaudio  # noqa: F401
    except ImportError:
        skip_or_fail("MINIAUDIO", "miniaudio is not installed")
    import wave
    import struct
    p = tmp_path / "float.wav"                           # 32-bit float WAV: stdlib wave can't read it
    x = (0.5 * np.sin(np.arange(22050) * 2 * np.pi * 440 / 22050)).astype("<f4")
    data = x.tobytes()
    hdr = (b"RIFF" + struct.pack("<I", 36 + len(data)) + b"WAVEfmt " +
           struct.pack("<IHHIIHH", 16, 3, 1, 22050, 22050 * 4, 4, 32) + b"data" + struct.pack("<I", len(data)))
    p.write_bytes(hdr + data)
    with pytest.raises(wave.Error):
        wave.open(str(p)).close()
    y, sr = glitch.load_audio(p)
    assert abs(len(y) / sr - 1.0) < 0.02 and 0.4 < np.abs(y).max() <= 1.0
