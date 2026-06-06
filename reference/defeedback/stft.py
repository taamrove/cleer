"""Short-time Fourier transform helpers.

The whole "AI denoise / dereverb" family of algorithms operates on a
spectrogram: chop the signal into overlapping frames, window each frame,
FFT it, manipulate the per-bin magnitude (apply a gain mask), then
overlap-add back to the time domain. This module is the plumbing for that.

In the native macOS app this is replaced by Accelerate's vDSP FFT, but the
math here is the exact reference the Swift port mirrors.
"""

from __future__ import annotations

import numpy as np


def hann(n: int) -> np.ndarray:
    # Periodic Hann (matches numpy.fft / STFT-COLA conventions).
    return 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(n) / n)


class STFT:
    """Overlap-add STFT/ISTFT with a constant-overlap-add (COLA) window.

    With a Hann window and hop = win/4 the squared window sums to a
    constant, so analysis+synthesis reconstructs the signal exactly (up to
    the gain mask we apply in between).
    """

    def __init__(self, win: int = 1024, hop: int = 256):
        assert win % hop == 0, "win must be a multiple of hop for clean COLA"
        self.win = win
        self.hop = hop
        self.window = hann(win)
        # COLA normalisation for analysis*synthesis windowing.
        # sum of squared window over all overlapping positions.
        norm = np.zeros(win)
        for k in range(0, win, hop):
            norm += np.roll(self.window ** 2, k)
        # norm is periodic with period hop; take the constant value.
        self._cola = norm[0]

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Return complex spectrogram, shape (n_frames, win//2+1)."""
        win, hop = self.win, self.hop
        # Pad so the first and last samples get full window coverage.
        pad = win
        xp = np.concatenate([np.zeros(pad), x, np.zeros(pad)])
        n_frames = 1 + (len(xp) - win) // hop
        frames = np.empty((n_frames, win // 2 + 1), dtype=np.complex128)
        for i in range(n_frames):
            seg = xp[i * hop : i * hop + win] * self.window
            frames[i] = np.fft.rfft(seg)
        self._pad = pad
        self._len = len(x)
        return frames

    def inverse(self, spec: np.ndarray) -> np.ndarray:
        win, hop = self.win, self.hop
        n_frames = spec.shape[0]
        out_len = (n_frames - 1) * hop + win
        out = np.zeros(out_len)
        for i in range(n_frames):
            seg = np.fft.irfft(spec[i], n=win) * self.window
            out[i * hop : i * hop + win] += seg
        out /= self._cola
        # Remove the padding added in forward().
        return out[self._pad : self._pad + self._len]


def freqs(win: int, sr: float) -> np.ndarray:
    return np.fft.rfftfreq(win, d=1.0 / sr)
