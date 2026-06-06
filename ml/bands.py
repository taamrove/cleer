"""Mel band filterbank shared by training and inference.

The neural mask predictor doesn't work on all 513 FFT bins — that would be a
big, slow, hard-to-train model. Instead we compress the spectrum to a small
number of perceptual (mel) bands (like RNNoise's 22 bands), predict one gain
per band, then interpolate those band gains back up to per-bin gains.

`band_matrix` : (n_bands x n_bins)  -- sum FFT power into band energies.
`interp_matrix`: (n_bins x n_bands) -- spread band gains back onto bins.

The Swift `MelBands.swift` builds the identical matrices so features and gain
application match exactly between the Python model and the app.
"""

from __future__ import annotations

import numpy as np


def hz_to_mel(f):
    return 2595.0 * np.log10(1.0 + f / 700.0)


def mel_to_hz(m):
    return 700.0 * (10.0 ** (m / 2595.0) - 1.0)


class MelBands:
    def __init__(self, n_bins: int, sr: float, n_bands: int = 32):
        self.n_bins = n_bins
        self.n_bands = n_bands
        self.sr = sr
        self.bin_freqs = np.linspace(0, sr / 2, n_bins)

        m_lo, m_hi = hz_to_mel(0.0), hz_to_mel(sr / 2)
        # n_bands triangular filters need n_bands+2 mel edge points.
        mel_edges = np.linspace(m_lo, m_hi, n_bands + 2)
        self.edges = mel_to_hz(mel_edges)
        self.centers = self.edges[1:-1]

        # Analysis: triangular weights summing power into bands.
        W = np.zeros((n_bands, n_bins))
        for b in range(n_bands):
            lo, ctr, hi = self.edges[b], self.edges[b + 1], self.edges[b + 2]
            for k in range(n_bins):
                f = self.bin_freqs[k]
                if lo <= f <= ctr and ctr > lo:
                    W[b, k] = (f - lo) / (ctr - lo)
                elif ctr < f <= hi and hi > ctr:
                    W[b, k] = (hi - f) / (hi - ctr)
        # Normalise each band so energy is an average, not a sum (stable scale).
        W /= (W.sum(axis=1, keepdims=True) + 1e-9)
        self.band_matrix = W

        # Synthesis: each bin's gain = linear interp across the two nearest
        # band centres (rows sum to 1).
        B = np.zeros((n_bins, n_bands))
        for k in range(n_bins):
            f = self.bin_freqs[k]
            if f <= self.centers[0]:
                B[k, 0] = 1.0
            elif f >= self.centers[-1]:
                B[k, -1] = 1.0
            else:
                j = np.searchsorted(self.centers, f) - 1
                j = min(max(j, 0), n_bands - 2)
                f0, f1 = self.centers[j], self.centers[j + 1]
                w = (f - f0) / (f1 - f0)
                B[k, j] = 1 - w
                B[k, j + 1] = w
        self.interp_matrix = B

    def band_energy(self, power: np.ndarray) -> np.ndarray:
        """power: (..., n_bins) -> (..., n_bands)."""
        return power @ self.band_matrix.T

    def gains_to_bins(self, band_gains: np.ndarray) -> np.ndarray:
        """band_gains: (..., n_bands) -> (..., n_bins)."""
        return band_gains @ self.interp_matrix.T
