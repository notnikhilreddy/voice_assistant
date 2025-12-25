from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


def _rms_energy(x: np.ndarray) -> float:
    if x.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(x * x)))


@dataclass
class VADResult:
    p_speech: float


class SileroVAD:
    """
    Silero VAD wrapper.

    If an ONNX model is available at `models/silero_vad.onnx`, use onnxruntime.
    Otherwise fall back to a lightweight energy VAD (still functional, but less accurate).
    """

    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self._use_fallback = False
        self._ort_session = None
        self._ort_in_name = None
        self._ort_sr_name = None

        model_path = os.getenv(
            "SILERO_VAD_ONNX_PATH",
            str(Path(__file__).resolve().parents[3] / "models" / "silero_vad.onnx"),
        )
        try:
            if os.path.exists(model_path):
                try:
                    import onnxruntime as ort  # type: ignore
                except Exception as e:
                    raise RuntimeError(
                        "onnxruntime is required for Silero VAD ONNX. Install with `pip install onnxruntime`."
                    ) from e

                providers = ["CPUExecutionProvider"]
                self._ort_session = ort.InferenceSession(model_path, providers=providers)
                ins = self._ort_session.get_inputs()
                self._ort_in_name = ins[0].name if ins else "input"
                # Some exported silero onnx expects sr input; tolerate absence.
                if len(ins) > 1:
                    self._ort_sr_name = ins[1].name
                print(f"Silero VAD (onnxruntime) loaded from {model_path}")
            else:
                print(f"Silero VAD ONNX not found at {model_path}; using energy VAD fallback.")
                self._use_fallback = True
        except Exception as e:
            print(f"Failed to init Silero VAD (onnx). Using energy fallback. Error: {e}")
            self._use_fallback = True

    def infer(self, audio_f32: np.ndarray) -> VADResult:
        """
        audio_f32: float32 mono in [-1,1], typically 20ms (320 samples) at 16kHz.
        """
        if audio_f32.dtype != np.float32:
            audio_f32 = audio_f32.astype(np.float32)
        if audio_f32.ndim != 1:
            audio_f32 = audio_f32.reshape(-1)

        if self._use_fallback or self._ort_session is None:
            # Simple energy gate, mapped to pseudo-probability.
            e = _rms_energy(audio_f32)
            # Tuned loosely for typical mic amplitude post-AGC.
            p = min(1.0, max(0.0, (e - 0.01) / 0.05))
            return VADResult(p_speech=float(p))

        try:
            x = audio_f32.reshape(1, -1)
            feeds = {self._ort_in_name: x}
            if self._ort_sr_name:
                feeds[self._ort_sr_name] = np.array([self.sample_rate], dtype=np.int64)
            out = self._ort_session.run(None, feeds)
            # Common export returns [p_speech]
            p = float(np.array(out[0]).reshape(-1)[0])
            return VADResult(p_speech=max(0.0, min(1.0, p)))
        except Exception as e:
            # Safety fallback on any runtime error.
            e = _rms_energy(audio_f32)
            p = min(1.0, max(0.0, (e - 0.01) / 0.05))
            return VADResult(p_speech=float(p))


