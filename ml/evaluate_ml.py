"""Compare the trained neural mask vs the classical denoiser on held-out audio.

Builds a fresh noisy mixture (clean voice + noise, unseen RNG), enhances it with
(a) the classical minimum-statistics/Wiener denoiser and (b) the neural mask,
then reports output SNR against the known clean reference. Higher = better.
"""

from __future__ import annotations

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "reference"))

from bands import MelBands                       # noqa: E402
from dataset import clean_utterance, FeatureExtractor, noise_pink  # noqa: E402
from model import MaskMLP                        # noqa: E402
from defeedback.stft import STFT                 # noqa: E402
from defeedback.noise_reducer import NoiseReducer  # noqa: E402

SR, WIN, HOP, N_BANDS = 48000, 1024, 256, 32


def snr_db(clean, est):
    m = min(len(clean), len(est))
    c, e = clean[:m], est[:m]
    return 10 * np.log10(np.sum(c ** 2) / (np.sum((e - c) ** 2) + 1e-12) + 1e-12)


def neural_enhance(noisy, model, bands, fe):
    stft = STFT(WIN, HOP)
    spec = stft.forward(noisy)
    fe.reset()
    for t in range(spec.shape[0]):
        power = np.abs(spec[t]) ** 2
        feat = fe.features(power)[None, :]
        band_gain = model.predict(feat)[0]
        bin_gain = bands.gains_to_bins(band_gain[None, :])[0]
        spec[t] *= bin_gain
    return stft.inverse(spec)


def classical_enhance(noisy):
    stft = STFT(WIN, HOP)
    nr = NoiseReducer(SR, WIN, HOP)
    spec = stft.forward(noisy)
    for t in range(spec.shape[0]):
        spec[t] *= nr.mask(np.abs(spec[t]))
    return stft.inverse(spec)


def main():
    rng = np.random.default_rng(777)  # unseen during training
    clean = clean_utterance(SR, 4.0, rng)
    nz = noise_pink(SR, len(clean), rng)
    nz /= np.std(nz) + 1e-9
    snr_in = 5.0
    gain = 10 ** (-snr_in / 20) * (np.std(clean) + 1e-9)
    noisy = clean + gain * nz

    bands = MelBands(WIN // 2 + 1, SR, N_BANDS)
    fe = FeatureExtractor(bands)
    model = MaskMLP.load(os.path.join(os.path.dirname(__file__), "out", "mask_mlp.npz"))

    neural = neural_enhance(noisy, model, bands, fe)
    classical = classical_enhance(noisy)

    print("=" * 56)
    print("NEURAL vs CLASSICAL DENOISE  (output SNR, higher = better)")
    print("=" * 56)
    print(f"  noisy input      : {snr_db(clean, noisy):6.2f} dB")
    print(f"  classical (Wiener): {snr_db(clean, classical):6.2f} dB")
    print(f"  neural (MLP mask) : {snr_db(clean, neural):6.2f} dB")
    g_c = snr_db(clean, classical) - snr_db(clean, noisy)
    g_n = snr_db(clean, neural) - snr_db(clean, noisy)
    print(f"\n  classical improvement: {g_c:+.2f} dB")
    print(f"  neural improvement   : {g_n:+.2f} dB")
    print("=" * 56)


if __name__ == "__main__":
    main()
