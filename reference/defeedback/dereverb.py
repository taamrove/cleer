"""Late-reverberation suppression by spectral gain.

Reverb is the room smearing each sound across time: a direct-path arrival
followed by an exponentially decaying tail of reflections. The tail at the
current frame is well modelled as a *delayed, decayed copy of the recent
signal power* in each band (this is the statistical model behind Lebart's
spectral-subtraction dereverb and the WPE family).

So we:
  1.  Estimate late-reverb power per bin = decay * (power D frames ago,
      smoothed).
  2.  Subtract it with a spectral gain, floored to avoid artefacts.

Like the denoiser this outputs a per-bin gain in [0,1], so the two masks
simply multiply together before a single inverse STFT.
"""

from __future__ import annotations

import numpy as np


class Dereverb:
    def __init__(
        self,
        sr: float,
        win: int,
        hop: int,
        rt60: float = 0.6,            # assumed room reverberation time (s)
        delay_ms: float = 40.0,       # boundary between "early" (kept) and "late" (removed)
        strength: float = 1.0,        # 0 = off, 1 = full late-tail subtraction
        gain_floor_db: float = -15.0,
        smooth: float = 0.6,          # power smoothing of the tail estimate
    ):
        self.sr = sr
        self.win = win
        self.hop = hop
        self.strength = strength
        self.gain_floor = 10.0 ** (gain_floor_db / 20.0)
        self.smooth = smooth

        frame_s = hop / sr
        self.delay = max(1, int(round((delay_ms / 1000.0) / frame_s)))
        # Per-frame energy decay implied by RT60 (-60 dB over rt60 seconds),
        # advanced by `delay` frames to predict the tail at the current frame.
        decay_per_frame = 10.0 ** (-3.0 * frame_s / rt60)  # -60dB == factor 1e-3
        self.decay = decay_per_frame ** self.delay

        bins = win // 2 + 1
        self.power_hist: list[np.ndarray] = []
        self.tail = np.zeros(bins)

    def mask(self, mag: np.ndarray) -> np.ndarray:
        power = mag ** 2
        self.power_hist.append(power)
        if len(self.power_hist) > self.delay + 1:
            self.power_hist.pop(0)

        if len(self.power_hist) > self.delay:
            delayed = self.power_hist[0]
            est = self.decay * delayed
            self.tail = self.smooth * self.tail + (1 - self.smooth) * est
        late = self.strength * self.tail

        gain = np.maximum(power - late, 0.0) / (power + 1e-12)
        gain = np.maximum(gain, self.gain_floor)
        return gain

    def process_spectrum(self, spec: np.ndarray) -> np.ndarray:
        out = np.empty_like(spec)
        for i in range(spec.shape[0]):
            g = self.mask(np.abs(spec[i]))
            out[i] = spec[i] * g
        return out
