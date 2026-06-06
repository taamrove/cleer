"""Single-channel noise reduction by spectral gain masking.

This is the "AI denoise" stage. A neural denoiser (RNNoise, DTLN, etc.)
ultimately outputs one thing: a *per-frequency gain mask* in [0,1] that is
multiplied onto the noisy spectrum — keep the voice bins, attenuate the
noise bins. The network's only job is to predict that mask well.

To keep this reference runnable without shipping trained weights, we
compute the same mask with the classical estimator the networks are trained
to imitate:

  * Noise power per bin is tracked with **minimum statistics** (the noise
    floor is the running minimum of the smoothed power — it shows through in
    the gaps between words).
  * From that we form the a-posteriori and (decision-directed) a-priori SNR
    and apply a **Wiener gain**  G = xi / (1 + xi).

The Swift app keeps this exact interface (`process_spectrum`) so a CoreML
gain-predictor can be dropped in to replace `_mask` with no other changes.
"""

from __future__ import annotations

import numpy as np


class NoiseReducer:
    def __init__(
        self,
        sr: float,
        win: int,
        hop: int,
        over_subtraction: float = 1.5,   # how aggressively to remove noise
        gain_floor_db: float = -18.0,    # never attenuate more than this (musical-noise guard)
        noise_smooth: float = 0.9,       # power smoothing for the floor tracker
        min_window_frames: int = 60,     # window over which we take the running minimum
        dd_alpha: float = 0.96,          # decision-directed a-priori SNR smoothing
    ):
        self.sr = sr
        self.win = win
        self.hop = hop
        self.over = over_subtraction
        self.gain_floor = 10.0 ** (gain_floor_db / 20.0)
        self.noise_smooth = noise_smooth
        self.min_window = min_window_frames
        self.dd_alpha = dd_alpha

        bins = win // 2 + 1
        self.p_smooth = np.zeros(bins)
        self.noise_psd = np.full(bins, 1e-6)
        self._min_buf = []  # recent smoothed-power frames for running minimum
        self.prev_gain = np.ones(bins)
        self.prev_power = np.zeros(bins)
        self._init = False

    def _update_noise(self, power: np.ndarray):
        a = self.noise_smooth
        if not self._init:
            self.p_smooth = power.copy()
            self.noise_psd = power.copy()
            self._init = True
        else:
            self.p_smooth = a * self.p_smooth + (1 - a) * power
        self._min_buf.append(self.p_smooth.copy())
        if len(self._min_buf) > self.min_window:
            self._min_buf.pop(0)
        # Minimum statistics: noise floor = running min, slightly biased up.
        running_min = np.min(np.stack(self._min_buf, axis=0), axis=0)
        self.noise_psd = 1.5 * running_min + 1e-12

    def mask(self, mag: np.ndarray) -> np.ndarray:
        """Return the gain mask for one magnitude frame."""
        power = mag ** 2
        self._update_noise(power)
        noise = self.over * self.noise_psd

        gamma = power / (noise + 1e-12)                  # a-posteriori SNR
        # Decision-directed a-priori SNR (Ephraim-Malah).
        xi = (self.dd_alpha * (self.prev_gain ** 2) * self.prev_power / (noise + 1e-12)
              + (1 - self.dd_alpha) * np.maximum(gamma - 1.0, 0.0))
        xi = np.maximum(xi, 1e-6)
        gain = xi / (1.0 + xi)                            # Wiener gain
        gain = np.maximum(gain, self.gain_floor)

        self.prev_gain = gain
        self.prev_power = power
        return gain

    def process_spectrum(self, spec: np.ndarray) -> np.ndarray:
        """Apply the mask frame-by-frame to a complex spectrogram."""
        out = np.empty_like(spec)
        for i in range(spec.shape[0]):
            g = self.mask(np.abs(spec[i]))
            out[i] = spec[i] * g
        return out
