"""Run the pipeline on the test signal and MEASURE what each stage removed.

This is the proof that we actually reverse-engineered the internals: it
reports, in dB, how much the howl, the background noise, and the reverb
tail were knocked down.
"""

from __future__ import annotations

import os
import sys
import numpy as np
from scipy.io import wavfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ""))
from defeedback import VoicePipeline, Config  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "..", "out")
HOWL_FREQ = 3147.0
HOWL_ONSET = 2.0


def load(name):
    sr, x = wavfile.read(os.path.join(OUT, name))
    x = x.astype(np.float64)
    if x.dtype == np.int16 or x.max() > 1.5:
        x /= 32768.0
    return sr, x


def goertzel_power(x, sr, f):
    """Energy at a single frequency (narrowband), via a sliding DFT bin."""
    n = len(x)
    k = np.exp(-2j * np.pi * f * np.arange(n) / sr)
    return np.abs(np.sum(x * k)) ** 2 / n


def db(ratio):
    return 10.0 * np.log10(ratio + 1e-20)


def short_time_rms(x, sr, win_ms=20):
    w = int(sr * win_ms / 1000)
    n = len(x) // w
    return np.array([np.sqrt(np.mean(x[i * w:(i + 1) * w] ** 2) + 1e-20)
                     for i in range(n)]), w


def main():
    sr, clean = load("clean.wav")
    _, dirty = load("dirty.wav")

    pipe = VoicePipeline(Config(sr=float(sr), feedback=True, denoise=True, dereverb=True))
    proc = pipe.process(dirty)
    m = min(len(proc), len(dirty), len(clean))
    proc, dirty, clean = proc[:m], dirty[:m], clean[:m]
    wavfile.write(os.path.join(OUT, "processed.wav"), sr, proc.astype(np.float32))

    print("=" * 60)
    print("DE-FEEDBACK REFERENCE — EFFECTIVENESS REPORT")
    print("=" * 60)

    # --- 1. Feedback / howl ------------------------------------------------
    on = int(HOWL_ONSET * sr) + sr // 2  # well after onset
    p_dirty = goertzel_power(dirty[on:], sr, HOWL_FREQ)
    p_proc = goertzel_power(proc[on:], sr, HOWL_FREQ)
    print(f"\n[1] FEEDBACK SUPPRESSION  (howl at {HOWL_FREQ:.0f} Hz)")
    print(f"    narrowband power: {db(p_dirty):7.1f} dB -> {db(p_proc):7.1f} dB")
    print(f"    >>> howl attenuated by {db(p_dirty) - db(p_proc):.1f} dB")
    print(f"    notches placed at: "
          f"{', '.join(f'{f:.0f}' for f in pipe.fb.detected_frequencies[:8]) or 'none'} Hz")

    # --- 2. Noise + reverb tail in the silent gaps -------------------------
    rms_clean, w = short_time_rms(clean, sr)
    rms_dirty, _ = short_time_rms(dirty, sr)
    rms_proc, _ = short_time_rms(proc, sr)
    thresh = np.percentile(rms_clean, 30)
    gaps = rms_clean < thresh  # frames where the talker is silent
    gd = np.mean(rms_dirty[gaps] ** 2)
    gp = np.mean(rms_proc[gaps] ** 2)
    print(f"\n[2] NOISE + REVERB-TAIL REDUCTION  (in {gaps.sum()} silent frames)")
    print(f"    gap energy: {db(gd):7.1f} dB -> {db(gp):7.1f} dB")
    print(f"    >>> residual in gaps reduced by {db(gd) - db(gp):.1f} dB")

    # --- 3. Dereverb-only on the reverberant tail --------------------------
    dr_pipe = VoicePipeline(Config(sr=float(sr), feedback=False, denoise=False, dereverb=True))
    dr = dr_pipe.process(dirty)[:m]
    rms_dr, _ = short_time_rms(dr, sr)
    # Compare energy decay just after voice offsets (the reverb tail region).
    offsets = np.where((~gaps[:-1]) & (gaps[1:]))[0] + 1
    tail = []
    for o in offsets:
        seg = slice(o, min(o + 4, len(rms_dirty)))  # ~80 ms after offset
        if seg.stop > seg.start:
            tail.append((np.mean(rms_dirty[seg] ** 2), np.mean(rms_dr[seg] ** 2)))
    if tail:
        td = np.mean([a for a, _ in tail]); tp = np.mean([b for _, b in tail])
        print(f"\n[3] DEREVERB  (post-offset tail over {len(tail)} word endings)")
        print(f"    tail energy: {db(td):7.1f} dB -> {db(tp):7.1f} dB")
        print(f"    >>> reverb tail shortened by {db(td) - db(tp):.1f} dB")

    print("\nwrote", os.path.abspath(os.path.join(OUT, "processed.wav")))
    print("=" * 60)


if __name__ == "__main__":
    main()
