"""Minimal frame-based STFT magnitude for training (offline, vectorised)."""

from __future__ import annotations

import numpy as np


def _hann(n):
    return 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n) / n)


def stft_mag(x: np.ndarray, win: int = 1024, hop: int = 256) -> np.ndarray:
    """Return magnitude spectrogram, shape (n_frames, win//2+1)."""
    w = _hann(win)
    if len(x) < win:
        x = np.concatenate([x, np.zeros(win - len(x))])
    n_frames = 1 + (len(x) - win) // hop
    out = np.empty((n_frames, win // 2 + 1))
    for i in range(n_frames):
        seg = x[i * hop:i * hop + win] * w
        out[i] = np.abs(np.fft.rfft(seg))
    return out
