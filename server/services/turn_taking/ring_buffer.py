from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Optional, Tuple

import numpy as np


@dataclass
class AudioFrame:
    seq: int
    pcm16: bytes  # little-endian PCM16 mono
    sample_rate: int = 16000

    def as_float32(self) -> np.ndarray:
        x = np.frombuffer(self.pcm16, dtype=np.int16).astype(np.float32)
        return (x / 32768.0).clip(-1.0, 1.0)


class AudioRingBuffer:
    """
    Stores the last N seconds of 16kHz mono PCM16 frames (typically 20ms = 320 samples).
    Designed for low-latency turn-taking decisions and segment extraction.
    """

    def __init__(self, max_seconds: float = 2.5, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self.max_samples = int(max_seconds * sample_rate)
        self._frames: Deque[AudioFrame] = deque()
        self._samples_in_buf = 0

    def push(self, frame: AudioFrame) -> None:
        self._frames.append(frame)
        self._samples_in_buf += len(frame.pcm16) // 2
        while self._samples_in_buf > self.max_samples and self._frames:
            old = self._frames.popleft()
            self._samples_in_buf -= len(old.pcm16) // 2

    def newest_seq(self) -> Optional[int]:
        return self._frames[-1].seq if self._frames else None

    def slice_by_seq(self, start_seq: int, end_seq_inclusive: int) -> List[AudioFrame]:
        if not self._frames:
            return []
        return [f for f in self._frames if start_seq <= f.seq <= end_seq_inclusive]

    def concat_float32(self, frames: List[AudioFrame]) -> np.ndarray:
        if not frames:
            return np.zeros(0, dtype=np.float32)
        arrs = [f.as_float32() for f in frames]
        return np.concatenate(arrs, axis=0) if arrs else np.zeros(0, dtype=np.float32)


