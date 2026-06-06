"""Adaptive feedback (howl) suppression.

This is the defining feature of a "de-feedback" tool. Microphone feedback
is a closed acoustic loop (speaker -> mic -> amp -> speaker) that builds up
energy at the room/mic resonance whose loop gain exceeds unity. In the
spectrum it shows up as a *narrow, tonal, persistent and rising* peak — a
"howl" or "ring" — which is exactly what distinguishes it from voice or
music (which are broadband and time-varying).

Algorithm (the same idea behind dbx AFS, Behringer FBQ, etc.):

  1.  Detector (frequency domain): every hop, FFT the latest window, find
      spectral peaks that stick far above the local spectral floor, sit in
      the feedback-prone band, and *persist* across consecutive frames.
  2.  Actuator (time domain): for each confirmed howl, place an adaptive
      parametric notch (a biquad) at the precise frequency. Depth grows
      fast while the howl is present (attack) and is released slowly once
      it is gone, then the slot is freed for re-use.

Detection is frequency-domain (cheap, accurate); filtering is time-domain
(zero added latency — no overlap-add delay). That split is why these
suppressors can run "with essentially zero latency".
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import lfilter, lfilter_zi

from .stft import hann


def peaking_biquad(sr: float, f0: float, q: float, gain_db: float):
    """RBJ cookbook peaking-EQ biquad. Negative gain_db => a notch."""
    A = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * np.pi * f0 / sr
    cosw = np.cos(w0)
    alpha = np.sin(w0) / (2.0 * q)
    b0 = 1.0 + alpha * A
    b1 = -2.0 * cosw
    b2 = 1.0 - alpha * A
    a0 = 1.0 + alpha / A
    a1 = -2.0 * cosw
    a2 = 1.0 - alpha / A
    b = np.array([b0, b1, b2]) / a0
    a = np.array([1.0, a1 / a0, a2 / a0])
    return b, a


@dataclass
class Notch:
    freq: float
    q: float
    depth_db: float = 0.0          # current depth (<= 0)
    target_db: float = 0.0         # where depth is heading
    zi: np.ndarray = field(default=None)  # biquad state for sample continuity
    b: np.ndarray = field(default=None)
    a: np.ndarray = field(default=None)
    idle: int = 0                  # frames spent fully released


class FeedbackSuppressor:
    def __init__(
        self,
        sr: float,
        win: int = 1024,
        hop: int = 256,
        max_notches: int = 16,
        f_lo: float = 150.0,
        f_hi: float = 9000.0,
        prominence_db: float = 12.0,   # how far above the floor counts as a peak
        hold_db: float = 6.0,          # keep a notch while peak stays this high
        persist_frames: int = 6,       # frames a peak must survive to be "feedback"
        max_depth_db: float = 36.0,    # deepest notch
        q: float = 18.0,               # notch sharpness
        attack_db: float = 12.0,       # depth added per frame while howling
        release_db: float = 1.5,       # depth removed per frame once gone
    ):
        self.sr = sr
        self.win = win
        self.hop = hop
        self.window = hann(win)
        self.max_notches = max_notches
        self.f_lo, self.f_hi = f_lo, f_hi
        self.prominence_db = prominence_db
        self.hold_db = hold_db
        self.persist_frames = persist_frames
        self.max_depth_db = max_depth_db
        self.q = q
        self.attack_db = attack_db
        self.release_db = release_db

        self.bins = win // 2 + 1
        self.fft_freqs = np.fft.rfftfreq(win, d=1.0 / sr)
        self.persist = np.zeros(self.bins)     # per-bin persistence counter
        self.notches: list[Notch] = []
        self._log: list[float] = []            # freqs we placed notches at

    # ---- detection -------------------------------------------------------
    def _detect(self, ring: np.ndarray) -> list[tuple[float, int]]:
        """Return [(freq_hz, bin)] of bins currently confirmed as feedback."""
        mag = np.abs(np.fft.rfft(ring * self.window)) + 1e-12
        power = mag ** 2
        # Spectral floor: wide median over frequency = the broadband content.
        floor = median_filter(power, size=41, mode="nearest") + 1e-12
        prom_db = 10.0 * np.log10(power / floor)

        in_band = (self.fft_freqs >= self.f_lo) & (self.fft_freqs <= self.f_hi)
        is_local_max = np.zeros(self.bins, dtype=bool)
        is_local_max[1:-1] = (power[1:-1] > power[:-2]) & (power[1:-1] > power[2:])
        candidate = in_band & is_local_max & (prom_db > self.prominence_db)

        # Persistence: tonal peaks survive; transients decay away.
        self.persist[candidate] += 1.0
        self.persist[~candidate] *= 0.5

        confirmed = []
        for k in np.nonzero(self.persist >= self.persist_frames)[0]:
            confirmed.append((self._interp_freq(power, k), int(k)))
        return confirmed

    def _interp_freq(self, power: np.ndarray, k: int) -> float:
        """Parabolic interpolation around bin k for sub-bin centre frequency."""
        if 0 < k < self.bins - 1:
            a, b, c = (np.log(power[k - 1] + 1e-12),
                       np.log(power[k] + 1e-12),
                       np.log(power[k + 1] + 1e-12))
            denom = (a - 2 * b + c)
            delta = 0.5 * (a - c) / denom if denom != 0 else 0.0
            delta = float(np.clip(delta, -0.5, 0.5))
        else:
            delta = 0.0
        return (k + delta) * self.sr / self.win

    # ---- notch pool management ------------------------------------------
    def _bin_of(self, freq: float) -> int:
        return int(round(freq * self.win / self.sr))

    def _update_notches(self, confirmed: list[tuple[float, int]], ring: np.ndarray):
        mag = np.abs(np.fft.rfft(ring * self.window)) + 1e-12
        power = mag ** 2
        floor = median_filter(power, size=41, mode="nearest") + 1e-12
        prom_db = 10.0 * np.log10(power / floor)

        confirmed_bins = {b for _, b in confirmed}
        active_bins = {self._bin_of(n.freq) for n in self.notches}

        # Allocate notches for newly-confirmed howls.
        for freq, b in confirmed:
            if any(abs(b - self._bin_of(n.freq)) <= 1 for n in self.notches):
                continue  # already covered
            if len(self.notches) >= self.max_notches:
                break
            n = Notch(freq=freq, q=self.q, target_db=-self.max_depth_db)
            self.notches.append(n)
            self._log.append(freq)

        # Drive each notch's depth and release dead ones.
        survivors = []
        for n in self.notches:
            b = self._bin_of(n.freq)
            still_hot = (0 <= b < self.bins) and (prom_db[b] > self.hold_db)
            n.target_db = -self.max_depth_db if still_hot else 0.0
            if n.depth_db > n.target_db:
                n.depth_db = max(n.target_db, n.depth_db - self.attack_db)
            else:
                n.depth_db = min(n.target_db, n.depth_db + self.release_db)
            n.idle = n.idle + 1 if n.depth_db >= -1e-6 else 0
            # Recompute coefficients for this frame's depth.
            n.b, n.a = peaking_biquad(self.sr, n.freq, n.q, n.depth_db)
            if n.zi is None:
                n.zi = lfilter_zi(n.b, n.a) * 0.0
            if n.idle < 20:  # keep a freed slot around briefly to avoid flapping
                survivors.append(n)
        self.notches = survivors

    # ---- main streaming entry point -------------------------------------
    def process(self, x: np.ndarray) -> np.ndarray:
        out = np.empty_like(x, dtype=np.float64)
        ring = np.zeros(self.win)
        n = len(x)
        idx = 0
        while idx < n:
            block = x[idx : idx + self.hop]
            # Slide the new block into the analysis ring buffer.
            ring = np.roll(ring, -len(block))
            ring[-len(block):] = block
            confirmed = self._detect(ring)
            self._update_notches(confirmed, ring)
            # Apply the notch cascade in the time domain, state-continuous.
            y = block.astype(np.float64)
            for nt in self.notches:
                y, nt.zi = lfilter(nt.b, nt.a, y, zi=nt.zi)
            out[idx : idx + len(block)] = y
            idx += len(block)
        return out

    @property
    def detected_frequencies(self) -> list[float]:
        return list(self._log)
