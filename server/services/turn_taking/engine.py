from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from .decision import TurnDecision, TurnState
from .prosody import ProsodyFeatures, ProsodyTracker
from .ring_buffer import AudioFrame, AudioRingBuffer
from .vad import SileroVAD


@dataclass
class TurnSegment:
    start_seq: int
    end_seq: int
    audio_f32: np.ndarray  # concatenated audio in [-1,1]


class TurnTakingEngine:
    """
    Cascaded streaming turn-taking engine:
    - Ring buffer over audio frames
    - Silero VAD gatekeeper
    - Prosody tracker (energy + pitch)
    - Decision engine (heuristic v1)
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        frame_ms: int = 20,
        ring_seconds: float = 2.5,
    ):
        self.sample_rate = sample_rate
        self.frame_ms = frame_ms
        self.vad = SileroVAD(sample_rate=sample_rate)
        self.prosody = ProsodyTracker(sample_rate=sample_rate, frame_ms=frame_ms, window_s=2.0)
        self.ring = AudioRingBuffer(max_seconds=ring_seconds, sample_rate=sample_rate)

        # State
        self._in_speech = False
        self._speech_start_seq: Optional[int] = None
        self._last_speech_seq: Optional[int] = None
        self._silence_ms: float = 0.0
        self._last_decision: TurnDecision = TurnDecision(
            state=TurnState.USER_THINKING, p_speech=0.0, silence_ms=0.0, reason="init"
        )

        # thresholds (tuned for <200ms end detection)
        self.p_speech_on = 0.55
        self.p_speech_off = 0.35
        self.finish_silence_ms = 180.0
        self.thinking_silence_ms = 120.0
        self.barge_in_speech_ms = 80.0

    def push(self, frame: AudioFrame) -> Tuple[TurnDecision, Optional[TurnSegment]]:
        """
        Push one audio frame and get:
        - current decision
        - optional TurnSegment if USER_FINISHED triggered
        """
        self.ring.push(frame)
        x = frame.as_float32()
        vad_res = self.vad.infer(x)
        p = float(vad_res.p_speech)

        # Prosody tracker always gets frames while speech is active (and for short tail)
        if p >= self.p_speech_off or self._in_speech:
            self.prosody.push_frame(x)

        seg: Optional[TurnSegment] = None

        if p >= self.p_speech_on:
            # speech ongoing
            if not self._in_speech:
                self._in_speech = True
                self._speech_start_seq = frame.seq
            self._last_speech_seq = frame.seq
            self._silence_ms = 0.0
            self._last_decision = TurnDecision(
                state=TurnState.USER_TALKING, p_speech=p, silence_ms=0.0, reason="vad_on"
            )
            return self._last_decision, None

        # Below speech-on threshold
        if self._in_speech and p <= self.p_speech_off:
            self._silence_ms += self.frame_ms
        else:
            # If we weren't in speech, keep silence_ms at 0 (idle)
            if not self._in_speech:
                self._silence_ms = 0.0

        feats = self.prosody.compute()

        if self._in_speech and self._silence_ms >= self.finish_silence_ms and self._should_shift(feats):
            # finalize segment: from speech_start_seq to last_speech_seq
            start_seq = int(self._speech_start_seq or frame.seq)
            end_seq = int(self._last_speech_seq or frame.seq)
            frames = self.ring.slice_by_seq(start_seq, end_seq)
            audio_f32 = self.ring.concat_float32(frames)
            seg = TurnSegment(start_seq=start_seq, end_seq=end_seq, audio_f32=audio_f32)

            self._last_decision = TurnDecision(
                state=TurnState.USER_FINISHED,
                p_speech=p,
                silence_ms=self._silence_ms,
                reason=f"shift silence_ms={self._silence_ms:.0f} f0_slope={feats.f0_slope:.2f} e_slope={feats.energy_slope:.4f}",
            )

            # reset for next utterance
            self._in_speech = False
            self._speech_start_seq = None
            self._last_speech_seq = None
            self._silence_ms = 0.0
            return self._last_decision, seg

        if self._in_speech and self._silence_ms >= self.thinking_silence_ms:
            self._last_decision = TurnDecision(
                state=TurnState.USER_THINKING,
                p_speech=p,
                silence_ms=self._silence_ms,
                reason=f"hold silence_ms={self._silence_ms:.0f}",
            )
        else:
            self._last_decision = TurnDecision(
                state=TurnState.USER_THINKING if self._in_speech else TurnState.USER_THINKING,
                p_speech=p,
                silence_ms=self._silence_ms,
                reason="idle_or_short_pause",
            )

        return self._last_decision, None

    def _should_shift(self, feats: ProsodyFeatures) -> bool:
        # Heuristic: energy decays and either pitch falls or becomes unvoiced.
        energy_decay = feats.energy_slope < -0.005 or feats.last_energy < 0.02
        pitch_fall_or_unvoiced = feats.last_f0 <= 0.0 or feats.f0_slope < 0.0
        return bool(energy_decay and pitch_fall_or_unvoiced)


