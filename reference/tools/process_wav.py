"""Run an arbitrary mono WAV through the validated reference pipeline.

Reads 16/24/32-bit PCM mono WAV, runs it through `VoicePipeline`, and writes a
16-bit WAV next to the input. Prints per-stage stats.

Usage:
    python3 tools/process_wav.py INPUT.wav [--out OUT.wav] [--start S] [--dur S]
"""
from __future__ import annotations
import argparse, os, sys, time, wave
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from defeedback.pipeline import VoicePipeline, Config
from defeedback.noise_reducer import NoiseReducer
from defeedback.dereverb import Dereverb


def read_wav_mono_f32(path: str, start_s: float = 0.0, dur_s: float | None = None):
    with wave.open(path, "rb") as w:
        sr = w.getframerate(); ch = w.getnchannels(); sw = w.getsampwidth(); n = w.getnframes()
        start_f = int(start_s * sr)
        if dur_s is None:
            count = n - start_f
        else:
            count = min(int(dur_s * sr), n - start_f)
        w.setpos(start_f)
        raw = w.readframes(count)
    # Decode to int based on sample width.
    if sw == 2:
        arr = np.frombuffer(raw, dtype="<i2").astype(np.float32)
        scale = 1.0 / 32768.0
    elif sw == 3:
        # 24-bit little-endian -> int32 via byte triplets.
        b = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
        i32 = (b[:, 0].astype(np.int32)
               | (b[:, 1].astype(np.int32) << 8)
               | (b[:, 2].astype(np.int32) << 16))
        i32 = np.where(i32 & 0x800000, i32 - 0x1000000, i32)
        arr = i32.astype(np.float32)
        scale = 1.0 / (1 << 23)
    elif sw == 4:
        arr = np.frombuffer(raw, dtype="<i4").astype(np.float32)
        scale = 1.0 / (1 << 31)
    else:
        raise ValueError(f"Unsupported sample width: {sw}")
    if ch > 1:
        arr = arr.reshape(-1, ch).mean(axis=1)
    return arr * scale, sr


def write_wav_pcm16(path: str, x: np.ndarray, sr: int):
    peak = float(np.max(np.abs(x))) + 1e-12
    if peak > 0.999:
        x = x / peak * 0.999
    i16 = np.clip(np.round(x * 32767.0), -32768, 32767).astype("<i2")
    with wave.open(path, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr)
        w.writeframes(i16.tobytes())


def rms_db(x: np.ndarray) -> float:
    r = float(np.sqrt(np.mean(x.astype(np.float64) ** 2)) + 1e-12)
    return 20.0 * np.log10(r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("--out", default=None)
    ap.add_argument("--start", type=float, default=0.0, help="start (seconds)")
    ap.add_argument("--dur", type=float, default=None, help="duration (seconds, default: full)")
    ap.add_argument("--no-feedback", action="store_true")
    ap.add_argument("--no-denoise", action="store_true")
    ap.add_argument("--no-dereverb", action="store_true")
    # Gentleness knobs.
    ap.add_argument("--nr-over", type=float, default=1.5, help="noise over-subtraction (lower = gentler)")
    ap.add_argument("--nr-floor-db", type=float, default=-18.0, help="noise gain floor in dB (closer to 0 = gentler)")
    ap.add_argument("--dr-strength", type=float, default=1.0, help="dereverb strength 0..1")
    ap.add_argument("--dr-floor-db", type=float, default=-15.0, help="dereverb gain floor in dB")
    ap.add_argument("--wet", type=float, default=1.0, help="dry/wet mix 0..1 (1 = full processed)")
    args = ap.parse_args()

    print(f"reading: {args.input}")
    print(f"  start={args.start}s  dur={'full' if args.dur is None else f'{args.dur}s'}")
    t0 = time.time()
    x, sr = read_wav_mono_f32(args.input, args.start, args.dur)
    print(f"  loaded {len(x)/sr:.1f}s at {sr} Hz in {time.time()-t0:.2f}s")
    print(f"  input  RMS: {rms_db(x):+.1f} dBFS   peak: {20*np.log10(np.max(np.abs(x))+1e-12):+.1f} dBFS")

    cfg = Config(sr=sr,
                 feedback=not args.no_feedback,
                 denoise=not args.no_denoise,
                 dereverb=not args.no_dereverb)
    print(f"  stages: feedback={cfg.feedback}  denoise={cfg.denoise}  dereverb={cfg.dereverb}")

    pipe = VoicePipeline(cfg)
    # Override denoise + dereverb with tuned params.
    pipe.nr = NoiseReducer(sr=sr, win=cfg.win, hop=cfg.hop,
                           over_subtraction=args.nr_over,
                           gain_floor_db=args.nr_floor_db)
    pipe.dr = Dereverb(sr=sr, win=cfg.win, hop=cfg.hop,
                       strength=args.dr_strength,
                       gain_floor_db=args.dr_floor_db)
    print(f"  nr: over={args.nr_over} floor={args.nr_floor_db}dB   dr: strength={args.dr_strength} floor={args.dr_floor_db}dB   wet={args.wet}")
    t0 = time.time()
    y_proc = pipe.process(x)
    # Optional dry/wet blend (1 = full processed, 0 = dry).
    n = min(len(x), len(y_proc))
    y = args.wet * y_proc[:n] + (1.0 - args.wet) * x[:n]
    rt = time.time() - t0
    dur_s = len(x) / sr
    print(f"processed in {rt:.1f}s  ({dur_s/rt:.1f}x realtime)")
    x = x[:n]
    print(f"  output RMS: {rms_db(y):+.1f} dBFS   peak: {20*np.log10(np.max(np.abs(y))+1e-12):+.1f} dBFS")
    print(f"  RMS delta: {rms_db(y) - rms_db(x):+.2f} dB")
    # Active notches the feedback stage placed (if any).
    if cfg.feedback and getattr(pipe.fb, "history", None):
        freqs = sorted({round(f, 1) for f in pipe.fb.history})
        if freqs:
            print(f"  feedback notches placed at: {freqs[:20]}{'...' if len(freqs)>20 else ''} Hz")

    out = args.out or (os.path.splitext(args.input)[0] + ".cleer.wav")
    write_wav_pcm16(out, y, sr)
    print(f"wrote: {out}  ({os.path.getsize(out)/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
