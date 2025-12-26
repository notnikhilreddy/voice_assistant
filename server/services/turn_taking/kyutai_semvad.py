from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass
class KyutaiVADStep:
    p_end_of_turn: float
    text_delta: str = ""
    # Full decoded text so far (preferred for UI; avoids SentencePiece byte-piece artifacts).
    text: str = ""


def _linear_resample(x: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    if x.size == 0 or src_rate == dst_rate:
        return x.astype(np.float32, copy=False)
    x = x.astype(np.float32, copy=False).reshape(-1)
    ratio = float(dst_rate) / float(src_rate)
    n_out = int(round(x.shape[0] * ratio))
    if n_out <= 1:
        return np.zeros((0,), dtype=np.float32)
    t = np.linspace(0.0, x.shape[0] - 1, num=n_out, dtype=np.float32)
    i0 = np.floor(t).astype(np.int32)
    i1 = np.minimum(i0 + 1, x.shape[0] - 1)
    frac = t - i0.astype(np.float32)
    return (x[i0] * (1.0 - frac) + x[i1] * frac).astype(np.float32)


class KyutaiSemanticVAD:
    """
    Kyutai semantic end-of-turn detector (\"semantic VAD\") using moshi_mlx.

    Implementation derived directly from Kyutai's reference scripts:
    - delayed-streams-modeling/scripts/stt_from_mic_mlx.py
    - delayed-streams-modeling/scripts/stt_from_file_mlx.py
    which call `LmGen.step_with_extra_heads(...)` and use:
      `pr_vad = vad_heads[2][0,0,0].item()`

    Notes:
    - Kyutai expects 24kHz input blocks of 1920 samples (~80ms).
      We accept 16kHz float frames and internally buffer+resample to 24k.
    - The VAD heads are available in the *-candle* checkpoints (loaded via `load_pytorch_weights`).
    """

    def __init__(
        self,
        hf_repo: Optional[str] = None,
        input_rate: int = 16000,
        model_rate: int = 24000,
        block_samples_24k: int = 1920,
    ):
        self.input_rate = input_rate
        self.model_rate = model_rate
        self.block_samples_24k = block_samples_24k

        self.hf_repo = hf_repo or os.getenv("KYUTAI_TURN_MODEL_ID", "kyutai/stt-1b-en_fr-candle")

        self._initialized = False
        self._mx = None
        self._gen = None
        self._text_tok = None
        self._audio_tok = None
        self._other_codebooks = None

        self._buf_16k = np.zeros((0,), dtype=np.float32)
        self._text_ids: list[int] = []

    def _ensure_loaded(self) -> None:
        if self._initialized:
            return
        try:
            import mlx.core as mx  # type: ignore
            import mlx.nn as nn  # type: ignore
            import sentencepiece  # type: ignore
            import rustymimi  # type: ignore
            from huggingface_hub import hf_hub_download  # type: ignore
            from moshi_mlx import models, utils  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "Kyutai semantic VAD requires moshi_mlx (+ huggingface_hub, rustymimi, sentencepiece)."
            ) from e

        lm_config_path = hf_hub_download(self.hf_repo, "config.json")
        with open(lm_config_path, "r") as fobj:
            cfg_dict = json.load(fobj)

        mimi_weights = hf_hub_download(self.hf_repo, cfg_dict["mimi_name"])
        moshi_name = cfg_dict.get("moshi_name", "model.safetensors")
        moshi_weights = hf_hub_download(self.hf_repo, moshi_name)
        tok_path = hf_hub_download(self.hf_repo, cfg_dict["tokenizer_name"])

        lm_config = models.LmConfig.from_config_dict(cfg_dict)
        model = models.Lm(lm_config)
        model.set_dtype(mx.bfloat16)
        if moshi_weights.endswith(".q4.safetensors"):
            nn.quantize(model, bits=4, group_size=32)
        elif moshi_weights.endswith(".q8.safetensors"):
            nn.quantize(model, bits=8, group_size=64)

        print(f"Loading Kyutai semantic-VAD moshi weights from {moshi_weights}")
        if self.hf_repo.endswith("-candle"):
            model.load_pytorch_weights(moshi_weights, lm_config, strict=True)
        else:
            model.load_weights(moshi_weights, strict=True)

        text_tokenizer = sentencepiece.SentencePieceProcessor(tok_path)  # type: ignore

        generated_codebooks = lm_config.generated_codebooks
        other_codebooks = lm_config.other_codebooks
        mimi_codebooks = max(generated_codebooks, other_codebooks)
        audio_tokenizer = rustymimi.Tokenizer(mimi_weights, num_codebooks=mimi_codebooks)  # type: ignore

        model.warmup()
        gen = models.LmGen(
            model=model,
            max_steps=4096,
            text_sampler=utils.Sampler(top_k=25, temp=0),
            audio_sampler=utils.Sampler(top_k=250, temp=0.8),
            check=False,
        )

        self._mx = mx
        self._gen = gen
        self._text_tok = text_tokenizer
        self._audio_tok = audio_tokenizer
        self._other_codebooks = other_codebooks
        self._initialized = True

    def reset(self) -> None:
        # A full reset of LmGen state requires re-instantiating it; easiest is to drop and reload.
        self._initialized = False
        self._mx = None
        self._gen = None
        self._text_tok = None
        self._audio_tok = None
        self._other_codebooks = None
        self._buf_16k = np.zeros((0,), dtype=np.float32)
        self._text_ids = []

    def _decode_text(self) -> str:
        if self._text_tok is None or not self._text_ids:
            return ""
        # Prefer SentencePiece full decoding (best UX; avoids <0x..> artifacts).
        if hasattr(self._text_tok, "decode_ids"):
            try:
                return str(self._text_tok.decode_ids(self._text_ids))  # type: ignore
            except Exception:
                pass
        # Fallback: join pieces
        try:
            parts = []
            for tid in self._text_ids:
                piece = self._text_tok.id_to_piece(tid)  # type: ignore
                # Filter obvious byte tokens
                if "<0x" in piece:
                    continue
                parts.append(piece.replace("▁", " "))
            return "".join(parts)
        except Exception:
            return ""

    def push_16k(self, frame_f32_16k: np.ndarray) -> Optional[KyutaiVADStep]:
        """
        Feed raw audio at 16kHz. Returns a KyutaiVADStep whenever we have processed
        one 80ms (1920@24k) block through the model.
        """
        self._ensure_loaded()
        assert self._mx is not None and self._gen is not None and self._audio_tok is not None

        x = np.asarray(frame_f32_16k, dtype=np.float32).reshape(-1)
        if x.size == 0:
            return None

        # Buffer and process in ~80ms @16k (1280 samples). Resample per block to 24k.
        self._buf_16k = np.concatenate([self._buf_16k, x], axis=0)
        block_16k = int(round(self.block_samples_24k * (self.input_rate / self.model_rate)))  # 1920 * (16/24)=1280

        if self._buf_16k.size < block_16k:
            return None

        chunk_16k = self._buf_16k[:block_16k]
        self._buf_16k = self._buf_16k[block_16k:]

        chunk_24k = _linear_resample(chunk_16k, self.input_rate, self.model_rate)
        if chunk_24k.size != self.block_samples_24k:
            # Pad/truncate to expected size
            if chunk_24k.size < self.block_samples_24k:
                chunk_24k = np.pad(chunk_24k, (0, self.block_samples_24k - chunk_24k.size))
            else:
                chunk_24k = chunk_24k[: self.block_samples_24k]

        mx = self._mx
        # Shape expected by rustymimi encode_step in Kyutai script: block[None, :, 0] -> [1, 1920]
        block = chunk_24k.astype(np.float32)[None, :]
        other_audio_tokens = self._audio_tok.encode_step(block[None, 0:1])
        other_audio_tokens = mx.array(other_audio_tokens).transpose(0, 2, 1)[:, :, : self._other_codebooks]

        text_token, vad_heads = self._gen.step_with_extra_heads(other_audio_tokens[0])

        pr_vad = 0.0
        if vad_heads:
            pr_vad = float(vad_heads[2][0, 0, 0].item())

        # Optional text delta
        text_delta = ""
        full_text = ""
        try:
            tid = int(text_token[0].item())
            if tid not in (0, 3) and self._text_tok is not None:
                # Track ids and decode the full text for stable UI
                self._text_ids.append(tid)
                full_text = self._decode_text().strip()
                # Provide a delta (best-effort); callers can ignore and use full_text.
                if full_text:
                    text_delta = full_text
        except Exception:
            pass

        return KyutaiVADStep(p_end_of_turn=pr_vad, text_delta=text_delta, text=full_text)


