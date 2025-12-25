from __future__ import annotations

from dataclasses import dataclass
from typing import Deque, List, Optional, Tuple
from collections import deque

import numpy as np


def rms(x: np.ndarray) -> float:
    if x.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(x * x)))


def spectral_energy(x: np.ndarray) -> float:
    """
    Proxy for spectral energy: sum of squared magnitude of FFT bins.
    This is not mel energy, but correlates well with overall energy/decay.
    """
    if x.size == 0:
        return 0.0
    # Window to reduce spectral leakage.
    w = np.hanning(len(x)).astype(np.float32)
    X = np.fft.rfft(x * w)
    mag2 = (X.real * X.real + X.imag * X.imag).astype(np.float32)
    return float(np.mean(mag2))


def estimate_f0_autocorr(x: np.ndarray, sample_rate: int = 16000) -> float:
    """
    Very lightweight pitch estimate via autocorrelation peak search.
    Returns 0.0 if unvoiced/uncertain.
    """
    if x.size < int(0.02 * sample_rate):
        return 0.0
    x = x.astype(np.float32)
    x = x - np.mean(x)
    e = rms(x)
    if e < 0.01:
        return 0.0

    # Autocorrelation
    corr = np.correlate(x, x, mode="full")[len(x) - 1 :]
    if corr[0] <= 0:
        return 0.0
    corr = corr / (corr[0] + 1e-8)

    # Search plausible pitch range
    fmin, fmax = 60.0, 350.0
    lag_min = int(sample_rate / fmax)
    lag_max = int(sample_rate / fmin)
    lag_max = min(lag_max, len(corr) - 1)
    if lag_max <= lag_min:
        return 0.0

    seg = corr[lag_min:lag_max]
    peak_idx = int(np.argmax(seg))
    peak = float(seg[peak_idx])
    if peak < 0.25:
        return 0.0
    lag = lag_min + peak_idx
    return float(sample_rate / max(1, lag))


@dataclass
class ProsodyFeatures:
    energy_slope: float
    f0_slope: float
    last_f0: float
    last_energy: float


class ProsodyTracker:
    """
    Maintains a short history (<=2s) of energy and F0 estimates to help
    decide HOLD vs SHIFT.
    """

    def __init__(self, sample_rate: int = 16000, frame_ms: int = 20, window_s: float = 2.0):
        self.sample_rate = sample_rate
        self.frame_ms = frame_ms
        self.max_frames = int((window_s * 1000) / frame_ms)
        self._energies: Deque[float] = deque(maxlen=self.max_frames)
        self._f0s: Deque[float] = deque(maxlen=self.max_frames)

    def push_frame(self, frame_f32: np.ndarray) -> None:
        e = rms(frame_f32)
        # Estimate pitch on a slightly longer window by reusing recent samples if possible.
        # For v1, estimate from the frame itself (cheap); decision engine uses slopes cautiously.
        f0 = estimate_f0_autocorr(frame_f32, sample_rate=self.sample_rate)
        self._energies.append(float(e))
        self._f0s.append(float(f0))

    def compute(self) -> ProsodyFeatures:
        energies = np.array(list(self._energies), dtype=np.float32)
        f0s = np.array(list(self._f0s), dtype=np.float32)
        if energies.size == 0:
            return ProsodyFeatures(energy_slope=0.0, f0_slope=0.0, last_f0=0.0, last_energy=0.0)

        # Slopes: compare last 5 frames to first 5 frames in window (robust + fast).
        n = energies.size
        k = min(5, n)
        e_start = float(np.mean(energies[:k]))
        e_end = float(np.mean(energies[-k:]))
        energy_slope = e_end - e_start

        # For F0 slope, consider only voiced frames
        voiced = f0s[f0s > 0.0]
        if voiced.size < 2:
            f0_slope = 0.0
            last_f0 = float(f0s[-1]) if f0s.size else 0.0
        else:
            k2 = min(5, voiced.size)
            f0_slope = float(np.mean(voiced[-k2:]) - np.mean(voiced[:k2]))
            last_f0 = float(voiced[-1])

        return ProsodyFeatures(
            energy_slope=float(energy_slope),
            f0_slope=float(f0_slope),
            last_f0=float(last_f0),
            last_energy=float(energies[-1]),
        )


