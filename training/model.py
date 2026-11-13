"""The model the trainer fine-tunes, as the decision interface defines it (docs/spec.md sections 4 to 7): Gemma 4's
text decoder run under the prefix_bidir attention mask, then a linear map from the last prompt token's hidden state to
one score per answer code.

    h = decoder(prompt)[:, -1]       prompts are padded on the left, so -1 is every row's last prompt token
    z = readout(h)                   [rows, 255]
    z = c * tanh(z / c)              c = final_logit_softcapping (30 for Gemma 4), as the LM head; skipped if unset
    z[:, k:] = -1e9                  a question with k options uses the first k codes only

Where the decoder comes from: either the Hugging Face release `google/gemma-4-*-it` itself, of which only the
`model.language_model.*` tensors are read (the vision and audio towers stay on disk), or a checkpoint this recipe
merged, which keeps that layout. A merged checkpoint brings its trained readout as `readout.safetensors`. A base has
none, so the readout starts as the embedding rows of the 255 code tokens; Gemma 4's LM head is tied to those
embeddings, so before any training the readout gives the code tokens the logits the base's LM head would give them
from the same hidden state. (The hidden state itself is not the base's: prefix_bidir is not the causal mask.)
"""

from __future__ import annotations

import json
import math
import re
import string
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from strands_decider import core

from . import attention, prompt

READOUT_FILE = "readout.safetensors"
LANGUAGE_MODEL = "model.language_model."


@dataclass(frozen=True)
class EncodedBatch:
    inputs: dict[str, torch.Tensor]
    counts: tuple[int, ...]
    input_tokens: int
    # per row: the index where the question starts, counting the left padding
    prefix_end: torch.Tensor


class LanguageModelWrapper(torch.nn.Module):
    """Puts the decoder under the attribute `language_model`. The adapters address modules by names such as
    `language_model.layers.3.self_attn.q_proj`, and this keeps those names the same for a base and a merged
    checkpoint."""

    def __init__(self, language_model: torch.nn.Module) -> None:
        super().__init__()
        self.language_model = language_model

    @property
    def config(self) -> Any:
        return self.language_model.config

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor, use_cache: bool = False
    ) -> Any:
        return self.language_model(
            input_ids=input_ids, attention_mask=attention_mask, use_cache=use_cache
        )

    def gradient_checkpointing_enable(self, **kwargs: Any) -> None:
        self.language_model.gradient_checkpointing_enable(**kwargs)


# transformers' text-decoder class for each Gemma 4 text config type (the 12B's checkpoint is `gemma4_unified`, every
# other size's `gemma4`).
TEXT_MODELS = {
    "gemma4_text": ("transformers.models.gemma4.modeling_gemma4", "Gemma4TextModel"),
    "gemma4_unified_text": (
        "transformers.models.gemma4_unified.modeling_gemma4_unified",
        "Gemma4UnifiedTextModel",
    ),
}
# Base tensors a text decoder does not load: the encoders and their projections, and (kv_shared_unused) the k/v
# projections and k-norm of the layers the text config marks KV-shared, which reuse an earlier layer's keys and values.
ENCODERS = re.compile(
    r"^model\.(vision_tower|vision_embedder|embed_vision|audio_tower|embed_audio)\."
)
KV_SHARED_UNUSED = ("k_proj", "v_proj", "k_norm")


def text_config(checkpoint: str | Path) -> Any:
    """The checkpoint's Gemma 4 text config."""
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(str(checkpoint))
    text = getattr(config, "text_config", None)
    if text is None or getattr(text, "model_type", "") not in TEXT_MODELS:
        raise ValueError(
            f"{checkpoint}: not a Gemma 4 checkpoint (text config {getattr(text, 'model_type', None)})"
        )
    return text


def kv_shared_unused(config: Any) -> set[str]:
    """The base names of the k/v projections and k-norms the KV-shared layers never read: those of the last
    `num_kv_shared_layers` layers only (transformers builds no such modules for them). Empty when no layer is shared."""
    first = config.num_hidden_layers - (getattr(config, "num_kv_shared_layers", 0) or 0)
    return {
        f"{LANGUAGE_MODEL}layers.{i}.self_attn.{module}.weight"
        for i in range(max(first, 0), config.num_hidden_layers)
        for module in KV_SHARED_UNUSED
    }


def load_decoder(checkpoint: str | Path, dtype: torch.dtype) -> torch.nn.Module:
    """The checkpoint's text decoder (transformers' own class for its type), from its `model.language_model.*` tensors,
    with SDPA attention. Refuses a checkpoint with a missing decoder tensor, or with any tensor it would ignore other
    than the encoders and the k/v of the layers the config marks KV-shared."""
    import importlib

    config = text_config(checkpoint)
    module, name = TEXT_MODELS[config.model_type]
    cls = getattr(importlib.import_module(module), name)
    from transformers.utils import logging

    level = logging.get_verbosity()
    logging.set_verbosity_error()  # the encoders' tensors are reported below, by the check, not as a table
    try:
        decoder, info = cls.from_pretrained(
            str(checkpoint),
            config=config,
            key_mapping={r"^model\.language_model\.": ""},
            dtype=dtype,
            attn_implementation="sdpa",
            output_loading_info=True,
        )
    finally:
        logging.set_verbosity(level)
    unused = kv_shared_unused(config)
    unexpected = sorted(
        k
        for k in info.get("unexpected_keys", ())
        if not ENCODERS.search(k) and k not in unused and LANGUAGE_MODEL + k not in unused
    )
    missing, mismatched = (
        sorted(info.get("missing_keys", ())),
        sorted(map(str, info.get("mismatched_keys", ()))),
    )
    if missing or unexpected or mismatched:
        raise ValueError(
            f"{checkpoint}: the decoder did not load cleanly: missing {missing[:3]}, "
            f"unexpected {unexpected[:3]}, mismatched {mismatched[:3]}"
        )
    return decoder


def check_revision(revision: str) -> None:
    if len(revision) != 40 or any(character not in string.hexdigits for character in revision):
        raise ValueError(
            f"revision {revision!r}: give the full 40-hex-digit commit, not a branch or tag"
        )


class DeciderModel(torch.nn.Module):
    """Decoder, readout and prompt template together. The decoder's own weights never train; they are bf16 on a GPU
    and float32 on the CPU.

    `checkpoint` is a local directory holding either a Gemma 4 base, whose readout starts from the tied LM head, or a
    merged checkpoint of this recipe, whose `readout.safetensors` is loaded."""

    def __init__(
        self,
        checkpoint: str | Path,
        *,
        prompt_template: str,
        device: str | None = None,
        cpu_threads: int = 8,
    ) -> None:
        super().__init__()
        # Process-wide settings, fixed so every run computes alike: the CPU thread count, and cuDNN attention off,
        # leaving SDPA its flash, memory-efficient and math kernels.
        torch.set_num_threads(cpu_threads)
        torch.backends.cuda.enable_cudnn_sdp(False)
        if prompt_template not in prompt.TEMPLATES:
            raise ValueError(
                f"prompt template {prompt_template!r} is not one of {prompt.TEMPLATES}"
            )
        trained = checkpoint_record(checkpoint).get("prompt")
        if trained and trained != prompt_template:
            raise ValueError(f"{checkpoint} was trained with {trained!r}, not {prompt_template!r}")
        self.checkpoint = Path(checkpoint)
        self.prompt = prompt_template
        self.device_name = device or ("cuda" if torch.cuda.is_available() else "cpu")
        from tokenizers import Tokenizer

        self.tokenizer = Tokenizer.from_file(str(self.checkpoint / "tokenizer.json"))
        pad = self.tokenizer.token_to_id("<pad>")
        self.pad_id = 0 if pad is None else pad
        self.codes = list(core.CODES)
        self.token_ids = prompt.code_token_ids(self.tokenizer, self.prompt)
        if len(set(self.token_ids)) != core.MAX_CODES:
            raise ValueError(
                "the tokenizer does not map the 255 codes to 255 different single tokens"
            )
        dtype = torch.bfloat16 if self.device_name.startswith("cuda") else torch.float32
        decoder = load_decoder(self.checkpoint, dtype)
        # Registered ahead of the decoder on purpose: DDP buckets gradients in parameter order, and this order is the
        # one the recipe's reference runs used.
        self.readout = torch.nn.Linear(
            decoder.config.hidden_size, core.MAX_CODES, bias=False, dtype=dtype
        )
        with torch.no_grad():
            saved = self.checkpoint / READOUT_FILE
            if saved.exists():
                from safetensors.torch import load_file

                self.readout.load_state_dict(load_file(str(saved)))
            else:
                self.readout.weight.copy_(decoder.embed_tokens.weight[self.token_ids])
        self.backbone = LanguageModelWrapper(decoder)
        attention.install_mask(self.backbone.language_model, owner=self)
        self.logit_softcap = self.backbone.config.final_logit_softcapping
        self._prefix_end: torch.Tensor | None = None
        self._split: dict | None = None
        self.requires_grad_(False)
        self.to(self.device_name)
        self.train(False)

    def install_split_attention(self, min_length: int) -> None:
        """For --attn-split: a microbatch `min_length` tokens wide or wider computes the prefix_bidir attention as two
        attention calls without a mask (attention.install_split)."""
        attention.install_split(self, self.backbone.language_model, min_length)

    def render(self, row: dict) -> core.RenderedQuestion:
        return prompt.render_question(row, self.tokenizer, self.prompt)

    def encode(self, rows: Sequence[dict], max_length: int = 16384) -> EncodedBatch:
        """Renders `rows` through the core, pads them on the left into one batch and puts it on the model's device."""
        if not rows:
            raise ValueError("encode() got no rows")
        rendered = [self.render(row) for row in rows]
        ids, mask, ends = prompt.left_pad(rendered, self.pad_id, max_length)
        counts = tuple(len(q.codes) for q in rendered)
        inputs = {
            "input_ids": ids.to(self.device_name),
            "attention_mask": mask.to(self.device_name),
        }
        return EncodedBatch(inputs, counts, int(mask.sum()), ends)

    def forward(self, batch: EncodedBatch) -> torch.Tensor:
        """Soft-capped float32 logits, [rows, 255], with -1e9 on every code beyond a row's option count."""
        self._prefix_end = batch.prefix_end
        try:
            hidden: torch.Tensor = self.backbone(**batch.inputs, use_cache=False).last_hidden_state
        finally:
            self._prefix_end = None
        hidden = hidden[:, -1]
        logits: torch.Tensor = self.readout(hidden).float()
        if self.logit_softcap:
            logits = torch.tanh(logits / self.logit_softcap) * self.logit_softcap
        mask = (
            torch.arange(core.MAX_CODES, device=logits.device)[None]
            >= torch.tensor(batch.counts, device=logits.device)[:, None]
        )
        return logits.masked_fill(mask, -1e9)

    @torch.inference_mode()  # type: ignore[untyped-decorator]
    def option_probabilities(
        self,
        rows: Sequence[dict],
        batch_size: int = 8,
        temperature: float = 1.0,
        max_length: int = 16384,
    ) -> list[list[float]]:
        """Per row, the softmax over its own options, at `temperature`. For checks and tests; strands-decider's
        runtimes do the serving."""
        if not math.isfinite(temperature) or temperature <= 0 or batch_size < 1:
            raise ValueError("temperature must be finite and above 0, and batch_size at least 1")
        was_training = self.training
        self.eval()
        out: list[list[float]] = []
        try:
            for start in range(0, len(rows), batch_size):
                batch = self.encode(rows[start : start + batch_size], max_length=max_length)
                probabilities = (self(batch) / temperature).softmax(-1).cpu().tolist()
                out.extend(
                    values[:count]
                    for values, count in zip(probabilities, batch.counts, strict=True)
                )
        finally:
            self.train(was_training)
        return out


def checkpoint_record(checkpoint: str | Path) -> dict:
    """The `training.json` a merged checkpoint carries, or {} for a base."""
    path = Path(checkpoint) / "training.json"
    return json.loads(path.read_text()) if path.exists() else {}
