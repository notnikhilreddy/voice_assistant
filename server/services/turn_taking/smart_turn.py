from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class SmartTurnResult:
    probability_complete: float
    prediction_complete: bool


class SmartTurnDetector:
    """
    Smart Turn v3 (pipecat-ai/smart-turn-v3) local inference.

    Implementation is derived from pipecat-ai/smart-turn `inference.py`:
    - ONNX model expects `input_features` shaped (B, 80, 800)
    - Use WhisperFeatureExtractor(chunk_length=8) on the *last* 8 seconds of 16kHz mono audio.
    - Output is a sigmoid probability of completion.
    """

    def __init__(self, repo_id: Optional[str] = None):
        self.repo_id = repo_id or os.getenv("SMART_TURN_REPO_ID", "pipecat-ai/smart-turn-v3")
        self.threshold = float(os.getenv("SMART_TURN_THRESHOLD", "0.5"))
        self.window_s = int(os.getenv("SMART_TURN_WINDOW_S", "8"))
        self._session = None
        self._feature_extractor = None
        # Cache a "hard failure" (e.g., missing repo file / no network) so we don't retry and spam logs
        # on every new websocket / engine instance.
        if not hasattr(SmartTurnDetector, "_GLOBAL_LOAD_FAILED"):
            SmartTurnDetector._GLOBAL_LOAD_FAILED = None  # type: ignore[attr-defined]

    def _ensure_loaded(self) -> None:
        failed = getattr(SmartTurnDetector, "_GLOBAL_LOAD_FAILED", None)  # type: ignore[attr-defined]
        if failed:
            raise RuntimeError(failed)
        if self._session is not None and self._feature_extractor is not None:
            return

        try:
            import onnxruntime as ort  # type: ignore
            from huggingface_hub import hf_hub_download  # type: ignore
            from transformers import WhisperFeatureExtractor  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "SmartTurn requires onnxruntime + huggingface_hub + transformers (WhisperFeatureExtractor)."
            ) from e

        # Try a few likely filenames to stay robust across model repo updates.
        candidates = [
            os.getenv("SMART_TURN_ONNX_FILENAME", "").strip(),
            "smart-turn-v3.1-cpu.onnx",
            "smart-turn-v3.1.onnx",
            "smart-turn-v3.onnx",
            "model_int8.onnx",
            "model_fp32.onnx",
        ]
        candidates = [c for c in candidates if c]
        last_err = None
        onnx_path = None
        for fn in candidates:
            try:
                onnx_path = hf_hub_download(self.repo_id, fn)
                break
            except Exception as e:
                last_err = e
                continue
        if onnx_path is None:
            msg = f"Failed to download SmartTurn ONNX from {self.repo_id}: {last_err}"
            setattr(SmartTurnDetector, "_GLOBAL_LOAD_FAILED", msg)  # type: ignore[attr-defined]
            raise RuntimeError(msg)

        so = ort.SessionOptions()
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        so.inter_op_num_threads = 1
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self._session = ort.InferenceSession(onnx_path, sess_options=so)
        self._feature_extractor = WhisperFeatureExtractor(chunk_length=self.window_s)

    def _truncate_to_last_window(self, audio_16k: np.ndarray) -> np.ndarray:
        audio_16k = np.asarray(audio_16k, dtype=np.float32).reshape(-1)
        max_samples = self.window_s * 16000
        if audio_16k.size > max_samples:
            return audio_16k[-max_samples:]
        if audio_16k.size < max_samples:
            pad = max_samples - audio_16k.size
            return np.pad(audio_16k, (pad, 0), mode="constant", constant_values=0.0)
        return audio_16k

    def predict(self, audio_16k: np.ndarray) -> SmartTurnResult:
        self._ensure_loaded()
        assert self._session is not None and self._feature_extractor is not None

        audio_16k = self._truncate_to_last_window(audio_16k)
        inputs = self._feature_extractor(
            audio_16k,
            sampling_rate=16000,
            return_tensors="np",
            padding="max_length",
            max_length=self.window_s * 16000,
            truncation=True,
            do_normalize=True,
        )
        input_features = inputs.input_features.squeeze(0).astype(np.float32)
        input_features = np.expand_dims(input_features, axis=0)  # (1, 80, 800)

        outputs = self._session.run(None, {"input_features": input_features})
        prob = float(outputs[0][0].item())
        return SmartTurnResult(probability_complete=prob, prediction_complete=(prob >= self.threshold))


def preload_smart_turn() -> None:
    """Pre-download and load SmartTurn ONNX model."""
    try:
        detector = SmartTurnDetector()
        detector._ensure_loaded()
        print("SmartTurn v3 ONNX loaded successfully.")
    except Exception as e:
        print(f"SmartTurn preload failed: {e}")
