"""Synthesise a torture-test signal: voice + howl + noise + reverb.

We don't have a real mic recording in this repo, so we build a controlled
one where we know the ground truth:

  * a "voice": syllable-like bursts of formant filtered noise (broadband,
    time-varying — what the pipeline must KEEP),
  * a feedback "howl": a steady, growing sine that switches on partway
    through (what the feedback suppressor must KILL),
  * steady broadband noise (what the denoiser must remove from the gaps),
  * a synthetic exponentially-decaying reverb tail (what dereverb shortens).

Writes clean.wav (voice only) and dirty.wav (everything) so evaluate.py can
measure how much of each contaminant the pipeline removed.
"""

from __future__ import annotations

import os
import numpy as np
from scipy.signal import butter, lfilter
from scipy.io import wavfile

SR = 48000
DUR = 6.0
HOWL_FREQ = 3147.0      # the feedback tone we'll hunt for afterwards
HOWL_ONSET = 2.0        # seconds


def voice(sr, dur):
    n = int(sr * dur)
    rng = np.random.default_rng(0)
    exc = rng.standard_normal(n)
    # Wandering "formants" => speech-like broadband, time-varying content.
    out = np.zeros(n)
    t = np.arange(n) / sr
    for f0, bw in [(500, 200), (1500, 300), (2500, 400)]:
        fc = f0 + 150 * np.sin(2 * np.pi * 0.7 * t)
        lo = np.clip((fc - bw) / (sr / 2), 1e-3, 0.99)
        hi = np.clip((fc + bw) / (sr / 2), 1e-3, 0.999)
        b, a = butter(2, [lo.mean(), hi.mean()], btype="band")
        out += lfilter(b, a, exc)
    # Syllable envelope: gate it on and off so there are real silent gaps.
    syl = (np.sin(2 * np.pi * 3.0 * t) > -0.2).astype(float)
    syl = lfilter(*butter(2, 12 / (sr / 2)), syl)
    out *= syl
    out /= np.max(np.abs(out)) + 1e-9
    return out * 0.5


def reverb_ir(sr, rt60=0.6):
    n = int(sr * rt60 * 1.2)
    t = np.arange(n) / sr
    rng = np.random.default_rng(1)
    ir = rng.standard_normal(n) * np.exp(-6.9 * t / rt60)
    ir[0] = 1.0  # direct path
    return ir / np.max(np.abs(ir))


def main():
    out_dir = os.path.join(os.path.dirname(__file__), "..", "out")
    os.makedirs(out_dir, exist_ok=True)
    n = int(SR * DUR)
    t = np.arange(n) / SR

    clean = voice(SR, DUR)

    # Reverberate the voice.
    ir = reverb_ir(SR)
    wet = np.convolve(clean, ir)[:n]
    wet /= np.max(np.abs(wet)) + 1e-9
    wet *= 0.5

    # Howl: switches on at HOWL_ONSET and ramps up (runaway loop gain).
    howl = np.sin(2 * np.pi * HOWL_FREQ * t)
    ramp = np.clip((t - HOWL_ONSET) / 1.0, 0, 1) * (t >= HOWL_ONSET)
    howl *= 0.6 * ramp

    # Steady broadband noise.
    noise = np.random.default_rng(2).standard_normal(n) * 0.02

    dirty = wet + howl + noise
    dirty /= np.max(np.abs(dirty)) + 1e-9
    dirty *= 0.9

    wavfile.write(os.path.join(out_dir, "clean.wav"), SR, _to_i16(clean))
    wavfile.write(os.path.join(out_dir, "dirty.wav"), SR, _to_i16(dirty))
    print("wrote", os.path.abspath(os.path.join(out_dir, "dirty.wav")))
    print(f"howl freq = {HOWL_FREQ} Hz, onset = {HOWL_ONSET}s")


def _to_i16(x):
    return np.clip(x, -1, 1).astype(np.float32)


if __name__ == "__main__":
    main()
