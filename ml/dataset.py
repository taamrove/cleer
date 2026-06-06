"""Training data for the neural mask predictor.

We need (features -> ideal gain) pairs. The classic recipe for speech
enhancement:

  1. take CLEAN speech,
  2. take NOISE,
  3. mix them at a random SNR -> NOISY,
  4. the target gain per band is the Ideal Ratio Mask (IRM):
         g = E_clean / (E_clean + E_noise)
     which we can compute exactly because we mixed it ourselves.

The model learns to predict that gain from features it can also compute at
run time (no peeking at the clean signal):

  feat = [ log band energy , band "SNR" vs the running noise-floor minimum ]

The same `make_examples` is reused for ON-DEVICE personalisation: pass the
user's captured room noise as the noise bank and the model fine-tunes to it.

Here "clean speech" is synthesised (harmonic voiced + noisy unvoiced syllables)
so the reference is self-contained. For production, swap `clean_bank` for a real
speech corpus — nothing else changes.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, lfilter

from bands import MelBands
from stft_np import stft_mag


def voiced(sr, dur, f0, rng):
    n = int(sr * dur)
    t = np.arange(n) / sr
    # Wandering pitch + a few harmonics => harmonic structure to learn.
    f = f0 * (1 + 0.03 * np.sin(2 * np.pi * 4 * t + rng.uniform(0, 6)))
    phase = 2 * np.pi * np.cumsum(f) / sr
    sig = sum((1.0 / h) * np.sin(h * phase) for h in range(1, 8))
    # Formant shaping.
    for fc in rng.uniform([400, 1200, 2400], [900, 1800, 3200]):
        lo = np.clip((fc - 200) / (sr / 2), 1e-3, 0.99)
        hi = np.clip((fc + 200) / (sr / 2), 1e-3, 0.999)
        b, a = butter(2, [lo, hi], btype="band")
        sig = sig + 0.5 * lfilter(b, a, sig)
    return sig / (np.max(np.abs(sig)) + 1e-9)


def unvoiced(sr, dur, rng):
    n = int(sr * dur)
    x = rng.standard_normal(n)
    fc = rng.uniform(2000, 6000)
    b, a = butter(2, fc / (sr / 2), btype="high")
    x = lfilter(b, a, x)
    return x / (np.max(np.abs(x)) + 1e-9)


def clean_utterance(sr, dur, rng):
    """A few syllables with silent gaps."""
    n = int(sr * dur)
    out = np.zeros(n)
    pos = 0
    while pos < n:
        seg = rng.uniform(0.12, 0.30)
        L = int(seg * sr)
        if rng.random() < 0.75:
            s = voiced(sr, seg, rng.uniform(90, 220), rng)
        else:
            s = unvoiced(sr, seg, rng)
        env = np.hanning(len(s))
        s = s * env * rng.uniform(0.4, 1.0)
        end = min(pos + L, n)
        out[pos:end] += s[: end - pos]
        pos = end + int(rng.uniform(0.03, 0.18) * sr)  # gap
    return out


def noise_white(sr, n, rng):
    return rng.standard_normal(n)


def noise_pink(sr, n, rng):
    w = rng.standard_normal(n)
    b, a = butter(1, 0.02)
    return lfilter(b, a, w)


def noise_hum(sr, n, rng):
    t = np.arange(n) / sr
    f = rng.choice([50, 60, 120])
    return np.sin(2 * np.pi * f * t) + 0.3 * np.sin(2 * np.pi * 2 * f * t)


DEFAULT_NOISES = [noise_white, noise_pink, noise_hum]


class FeatureExtractor:
    """Turns a magnitude frame into the model's input features, statefully
    tracking the per-band noise floor (running minimum). Mirrored in Swift."""

    def __init__(self, bands: MelBands, min_window=60):
        self.bands = bands
        self.min_window = min_window
        self.buf = []

    def reset(self):
        self.buf = []

    def features(self, power_frame: np.ndarray) -> np.ndarray:
        e = self.bands.band_energy(power_frame) + 1e-9       # (n_bands,)
        self.buf.append(e)
        if len(self.buf) > self.min_window:
            self.buf.pop(0)
        floor = np.min(np.stack(self.buf), axis=0) + 1e-9
        log_e = np.log(e)
        snr = np.log(e) - np.log(floor)                      # >= 0
        return np.concatenate([log_e, snr]).astype(np.float64)


def make_examples(sr=48000, win=1024, hop=256, n_bands=32,
                  n_utts=120, noises=None, rng=None):
    """Return (X, Y): features (N, 2*n_bands) and IRM targets (N, n_bands)."""
    rng = rng or np.random.default_rng(0)
    noises = noises if noises is not None else DEFAULT_NOISES
    bands = MelBands(win // 2 + 1, sr, n_bands)
    fe = FeatureExtractor(bands)

    X, Y = [], []
    for _ in range(n_utts):
        dur = rng.uniform(1.5, 2.5)
        clean = clean_utterance(sr, dur, rng)
        n = len(clean)
        nz_fn = noises[rng.integers(len(noises))]
        nz = nz_fn(sr, n, rng) if callable(nz_fn) else _draw_from_bank(nz_fn, n, rng)
        nz = nz / (np.std(nz) + 1e-9)

        snr_db = rng.uniform(-5, 20)
        gain = 10 ** (-snr_db / 20) * (np.std(clean) + 1e-9)
        noisy = clean + gain * nz

        Sc = stft_mag(clean, win, hop) ** 2
        Sn = stft_mag(gain * nz, win, hop) ** 2
        Sx = stft_mag(noisy, win, hop) ** 2

        # Ideal Ratio Mask per band (exact, since we know clean & noise).
        irm_band = bands.band_energy(Sc) / (bands.band_energy(Sc) + bands.band_energy(Sn) + 1e-9)

        fe.reset()
        for t in range(Sx.shape[0]):
            X.append(fe.features(Sx[t]))
            Y.append(np.clip(irm_band[t], 0.0, 1.0))
    return np.array(X), np.array(Y), bands


def _draw_from_bank(bank, n, rng):
    """Tile/crop a captured-noise clip (used for on-device personalisation)."""
    if len(bank) >= n:
        s = rng.integers(0, len(bank) - n + 1)
        return bank[s:s + n].astype(np.float64)
    reps = int(np.ceil(n / len(bank)))
    return np.tile(bank, reps)[:n].astype(np.float64)
