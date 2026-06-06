"""Full de-feedback voice pipeline = the three stages chained.

Signal flow (this is exactly what one "instance" in the macOS app runs on
one device/channel):

    in -> [feedback suppressor: time-domain adaptive notches]
       -> STFT
       -> [noise mask] x [dereverb mask]   (multiply, single inverse STFT)
       -> out

Feedback suppression is done in the time domain first (zero latency, and it
removes the howl *before* it pollutes the spectral statistics the other two
stages rely on). Noise and dereverb share one STFT/ISTFT pair and just
multiply their gain masks together, so there's only one transform round-trip.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .stft import STFT
from .feedback_suppressor import FeedbackSuppressor
from .noise_reducer import NoiseReducer
from .dereverb import Dereverb


@dataclass
class Config:
    sr: float = 48000.0
    win: int = 1024
    hop: int = 256
    feedback: bool = True
    denoise: bool = True
    dereverb: bool = True


class VoicePipeline:
    def __init__(self, cfg: Config = Config()):
        self.cfg = cfg
        self.stft = STFT(cfg.win, cfg.hop)
        self.fb = FeedbackSuppressor(cfg.sr, cfg.win, cfg.hop)
        self.nr = NoiseReducer(cfg.sr, cfg.win, cfg.hop)
        self.dr = Dereverb(cfg.sr, cfg.win, cfg.hop)

    def process(self, x: np.ndarray) -> np.ndarray:
        y = x.astype(np.float64)
        if self.cfg.feedback:
            y = self.fb.process(y)
        if self.cfg.denoise or self.cfg.dereverb:
            spec = self.stft.forward(y)
            for i in range(spec.shape[0]):
                mag = np.abs(spec[i])
                g = np.ones_like(mag)
                if self.cfg.denoise:
                    g *= self.nr.mask(mag)
                if self.cfg.dereverb:
                    g *= self.dr.mask(mag)
                spec[i] = spec[i] * g
            y = self.stft.inverse(spec)
        return y
