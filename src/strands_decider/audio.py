"""Audio input for Gemma 4 E2B / E4B checkpoints: decoding, limits and token counts.

A clip arrives base64-encoded (WAV, FLAC, OGG/Vorbis/Opus or MP3, through libsndfile), is
mixed to mono and resampled to the 16 kHz Gemma 4's audio encoder reads. The encoder emits
one soft token per 40 ms (two stride-2 convolutions over 10 ms mel frames), so a 30 s clip,
the longest Gemma 4 takes, is 750 tokens. The count is read from the feature extractor's
frame mask exactly as `Gemma4Processor.replace_audio_token` reads it, so the prompt holds
one placeholder per soft token the encoder fills.

Needs soundfile and scipy (`pip install "strands-decider[audio]"`).
"""

from __future__ import annotations

import base64
import binascii
import io
import math
import re
from collections.abc import Sequence
from typing import Any

import numpy as np
import torch

SAMPLE_RATE = 16_000
# What docs/audio.md promises; libsndfile's other formats (raw PCM, AIFF, ...) are refused.
AUDIO_FORMATS = ("WAV", "FLAC", "OGG", "MP3")
# Gemma 4 takes at most 30 s per clip (its processor's 750-token cap at 40 ms per token).
MAX_SECONDS = 30.0


def _soundfile() -> Any:
    try:
        import soundfile
    except ImportError as e:  # pragma: no cover - depends on the environment
        raise ImportError('audio input needs soundfile and scipy: pip install "strands-decider[audio]"') from e
    return soundfile


def decode_audio(data: str, max_seconds: float = MAX_SECONDS) -> np.ndarray:
    """A base64 clip (optionally a `data:audio/...;base64,` URI) as mono float32 at 16 kHz.

    Only AUDIO_FORMATS are opened, and the duration is read from the header before any
    sample is decoded, so a long clip is refused cheaply rather than silently cut.
    """
    payload = re.sub(r"^data:audio/[\w.+-]+;base64,", "", data.strip())
    try:
        raw = base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError) as e:
        raise ValueError(f"audio is not valid base64: {e}") from e
    return read_audio(raw, max_seconds)


def read_audio(raw: bytes, max_seconds: float = MAX_SECONDS) -> np.ndarray:
    """Encoded audio bytes as mono float32 at 16 kHz (see decode_audio)."""
    sf = _soundfile()
    try:
        info = sf.info(io.BytesIO(raw))
    except Exception as e:
        raise ValueError(f"audio could not be read as {'/'.join(AUDIO_FORMATS)}: {e}") from e
    if info.format not in AUDIO_FORMATS:
        raise ValueError(f"audio format {info.format} is not accepted; send {'/'.join(AUDIO_FORMATS)}")
    if info.samplerate <= 0 or info.frames <= 0:
        raise ValueError("audio is empty")
    seconds = info.frames / info.samplerate
    if seconds > max_seconds:
        raise ValueError(f"audio is {seconds:.1f} s; at most {max_seconds:g} s per clip are accepted")
    try:
        wave, rate = sf.read(io.BytesIO(raw), dtype="float32", always_2d=True)
    except Exception as e:
        raise ValueError(f"audio could not be decoded: {e}") from e
    return resample(wave.mean(axis=1), int(rate))


def resample(wave: np.ndarray, rate: int) -> np.ndarray:
    """Mono `wave` at `rate` Hz as float32 at 16 kHz (polyphase, scipy's default filter)."""
    wave = np.asarray(wave, dtype=np.float32)
    if rate == SAMPLE_RATE:
        return wave
    from scipy.signal import resample_poly

    g = math.gcd(SAMPLE_RATE, rate)
    out: np.ndarray = resample_poly(wave, SAMPLE_RATE // g, rate // g).astype(np.float32)
    return out


def load_feature_extractor(base_model: str, revision: str | None = None) -> Any:
    """The base's `Gemma4AudioFeatureExtractor` (log-mel, 128 bins, 10 ms hop)."""
    import transformers

    rev: dict[str, Any] = {"revision": revision} if revision else {}
    return transformers.Gemma4AudioFeatureExtractor.from_pretrained(base_model, **rev)


def audio_token_counts(frame_mask: torch.Tensor) -> list[int]:
    """Soft tokens per clip: the valid mel frames after two stride-2 (kernel 3, pad 1)
    subsamplings, as `Gemma4Processor.replace_audio_token` counts them."""
    counts = []
    for mask in frame_mask:
        m = mask
        for _ in range(2):
            t_out = (len(m) + 2 - 3) // 2 + 1
            m = m[::2][:t_out]
        counts.append(int(m.sum()))
    return counts


def process_audio(
    feature_extractor: Any, clips: Sequence[np.ndarray]
) -> tuple[list[int], dict[str, torch.Tensor]]:
    """Mel features for the audio encoder, and the placeholder tokens each clip takes."""
    out = feature_extractor(list(clips), sampling_rate=SAMPLE_RATE, return_tensors="pt")
    mask = out["input_features_mask"]
    return audio_token_counts(mask), {"input_features": out["input_features"], "input_features_mask": mask}
