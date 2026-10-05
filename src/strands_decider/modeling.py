"""The model: a pretrained LLM torso with its LM head removed, plus a readout head.

    input -> [LLM torso] -> hidden states -> [head] -> one logit per option

Two heads exist. The pointer head, the v19 readout, scores option k from the hidden
state of option k's own last token, so the option count is unbounded; see PointerHead.
The slot head, the v1-v6 readout, maps the last-token state to K fixed slots. In both
cases the logits past N, the number of options this question declared, are masked
before the softmax, and one mechanism serves all three primitives; see schema.py.

Why last-token pooling on a causal model rather than a BERT-style [CLS]: the torso
is a decoder, so only the final position has attended to the whole prompt. Keeping
attention causal means the pretrained weights are used exactly as they were trained,
which matters a great deal when the adaptation is a rank-16 LoRA adapter.
"""

from __future__ import annotations

import contextlib
import json
import os
from dataclasses import asdict, dataclass, field
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from transformers import AutoConfig, AutoModel, AutoTokenizer

# Slots the head can address. This must match the largest option count the corpus
# actually contains: slots beyond that never receive a gradient and would ship at
# their random initialisation. `strands-decider data build --max-options 24` is the matching
# default. Questions with more options are rejected, not silently truncated.
DEFAULT_NUM_SLOTS = 24

# Large negative rather than -inf: keeps softmax finite under autocast/bf16, where
# -inf can produce NaNs if an entire row is masked by a malformed batch.
MASK_VALUE = -1e4


@dataclass
class StrandsDeciderConfig:
    base_model: str = "Qwen/Qwen3-1.7B-Base"
    # Hub revision of base_model the adapter was trained on. None loads the repo's default branch;
    # `load` falls back to the checkpoint's provenance.json.
    base_revision: str | None = None
    num_slots: int = DEFAULT_NUM_SLOTS
    # 0 = single linear projection. >0 inserts one GELU hidden layer of this width.
    head_hidden: int = 0
    head_dropout: float = 0.0
    # 3072 covers every JevBench long_policy item whole. Measured: 1024 -> 3072 moved
    # long_policy 0.263 -> 0.368 and changed nothing else; 3072 -> 4096 changed nothing
    # at all, so this is where the context curve goes flat.
    max_length: int = 3072
    torch_dtype: str = "bfloat16"
    # "random" (default) or "lm_head": seed slot k from the output-embedding row for
    # the token "k+1", the digit the LM would emit after `<answer>` if it were asked
    # for the option number. Only slots 0-8 have a single-token form in this
    # tokeniser, so higher slots keep the default init.
    head_init: str = "random"
    # >0 adds KL(frozen || student) to the loss, pulling the trained head toward the
    # untouched torso's own reading of the option-number tokens.
    kl_frozen_weight: float = 0.0
    # "slot" = Linear(hidden, num_slots), the v1-v6 readout. "pointer" scores each
    # option from its own hidden state; see PointerHead. Defaults to "slot" so every
    # existing checkpoint keeps loading unchanged.
    head_type: str = "slot"
    pointer_dim: int = 256
    # Populated after `strands-decider calibrate` runs; 1.0 is a no-op. `temperature` is the
    # fallback; `temperature_by_kind` overrides it per primitive where fitted.
    # One global scalar cannot serve all three: noul/choice/score sit at very
    # different accuracies, so a temperature good for one over-softens another.
    temperature: float = 1.0
    temperature_by_kind: dict[str, float] = field(default_factory=dict)
    # Per-kind temperatures for questions asked over images (vision.VisionEngine), fitted
    # on held-out image items; a kind missing here falls back to temperature_by_kind.
    image_temperature_by_kind: dict[str, float] = field(default_factory=dict)
    # Mirrors the training-time collator setting. Inference needs it to correct the
    # variance floor that smoothing imposes on score confidence (see schema.py).
    ordinal_smoothing: float = 0.0
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_targets: list[str] = field(
        default_factory=lambda: [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]
    )

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @classmethod
    def from_json(cls, path: str) -> StrandsDeciderConfig:
        with open(path, encoding="utf-8") as fh:
            raw = json.load(fh)
        # The candidate models' checkpoints name the field as their training configs did
        # before it was `base_revision`.
        if "base_model_revision" in raw:
            raw.setdefault("base_revision", raw.pop("base_model_revision"))
        return cls(**raw)


class SlotHead(nn.Module):
    """Projects a pooled hidden state to K slot logits.

    LayerNorm first: the torso's final hidden states are not normalised on most
    decoder architectures, and their scale drifts with sequence length, which makes
    an unnormalised linear head slow to converge.
    """

    def __init__(self, hidden_size: int, num_slots: int, hidden: int = 0, dropout: float = 0.0):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        if hidden > 0:
            self.proj: nn.Module = nn.Sequential(
                nn.Linear(hidden_size, hidden),
                nn.GELU(),
                nn.Linear(hidden, num_slots),
            )
        else:
            self.proj = nn.Linear(hidden_size, num_slots)

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        return self.proj(self.dropout(self.norm(pooled)))


class PointerHead(nn.Module):
    """Scores option k from option k's own hidden state, not from a fixed slot.

    A single attention score: query from the `<answer>` position, key from the last
    token of each option's line -- the token that has just read that option under
    causal attention. So an option's logit depends on what it *says*, and there is no
    per-slot parameter for a positional bias to live in.

    Three consequences. The option count is unbounded rather than capped at
    `num_slots`. Genericity is structural instead of something option shuffling has to
    beat into a positional head. And the parameter count does not grow with K.

    Measured on frozen v6 features, against the slot head refitted on the same rows:
    +0.063 overall at dp=256 and +0.058 at dp=16, where dp=16 is 70k parameters
    against the linear slot head's 53k. A 2.1M-parameter MLP slot head reaches only
    +0.032, so the gain is the addressing rather than capacity.

    The LayerNorm is not in the reference implementation but is in ours, because our
    torso's hidden states are not scaled for a bare dot product -- without it the loss
    leaves the range cross-entropy over K options can occupy at all.
    """

    def __init__(self, hidden_size: int, dim: int = 256, dropout: float = 0.0):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.q = nn.Linear(hidden_size, dim)
        self.k = nn.Linear(hidden_size, dim)
        self.scale = dim ** -0.5

    def forward(self, decide: torch.Tensor, options: torch.Tensor) -> torch.Tensor:
        """decide [B, d], options [B, K, d] -> logits [B, K]."""
        d = self.q(self.dropout(self.norm(decide))).unsqueeze(-1)   # [B, dim, 1]
        o = self.k(self.dropout(self.norm(options)))                # [B, K, dim]
        return (o @ d).squeeze(-1) * self.scale


def build_head(config: StrandsDeciderConfig, hidden_size: int) -> nn.Module:
    """The readout named by `config.head_type`, in fp32 either way.

    The head trains in fp32 even when the torso is bf16: it is tiny, and a
    low-precision classifier is a needless source of calibration error.
    """
    if config.head_type == "pointer":
        head: nn.Module = PointerHead(
            hidden_size, dim=config.pointer_dim, dropout=config.head_dropout
        )
    elif config.head_type == "slot":
        head = SlotHead(
            hidden_size,
            config.num_slots,
            hidden=config.head_hidden,
            dropout=config.head_dropout,
        )
    else:
        raise ValueError(f"unknown head_type {config.head_type!r}")
    return head.to(torch.float32)


def gather_options(hidden: torch.Tensor, opt_idx: torch.Tensor) -> torch.Tensor:
    """hidden [B, L, d] + opt_idx [B, K] -> [B, K, d].

    Padded entries are -1; they are clamped to 0 here and carry arbitrary values, which
    is safe because `masked_log_softmax` drops every slot past the row's option count.
    """
    idx = opt_idx.clamp_min(0).unsqueeze(-1).expand(-1, -1, hidden.size(-1))
    return hidden.gather(1, idx)


def pool_last_token(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Gather the hidden state at each sequence's final unmasked position.

    Works for right-padded batches. `attention_mask` may be longer than
    `hidden_states` when a shared prefix cache supplies earlier positions; in that
    case we index relative to the tail that was actually forwarded.
    """
    seq_len = hidden_states.size(1)
    tail_mask = attention_mask[:, -seq_len:]
    lengths = tail_mask.sum(dim=1)
    if (lengths == 0).any():
        raise ValueError("every sequence must have at least one unmasked token in the tail")
    idx = (lengths - 1).to(torch.long)
    gather_idx = idx.view(-1, 1, 1).expand(-1, 1, hidden_states.size(-1))
    return hidden_states.gather(1, gather_idx).squeeze(1)


def masked_log_softmax(logits: torch.Tensor, n_slots: torch.Tensor) -> torch.Tensor:
    """Log-softmax over each row's first `n_slots[i]` entries; masked slots -> -inf.

    Masking (rather than always using all K) is what lets one head serve a 2-option
    noul and a 10-level score without the unused slots stealing probability mass.
    """
    k = logits.size(-1)
    ar = torch.arange(k, device=logits.device).unsqueeze(0)
    valid = ar < n_slots.unsqueeze(1)
    masked = logits.masked_fill(~valid, MASK_VALUE)
    log_probs = F.log_softmax(masked.float(), dim=-1)
    return log_probs.masked_fill(~valid, float("-inf"))


def apply_temperature(logits: torch.Tensor, temperature: Any) -> torch.Tensor:
    """Divide logits by a scalar, or by a per-row tensor of temperatures.

    A per-row tensor is what lets one batch mix primitives while each gets its own
    fitted temperature -- necessary because a single request may carry a noul, a
    choice and a score together.
    """
    if temperature is None:
        return logits
    if isinstance(temperature, torch.Tensor):
        t = temperature.to(logits.device, logits.dtype).clamp_min(1e-6)
        return logits / t.view(-1, 1)
    if temperature == 1.0:
        return logits
    return logits / float(temperature)


class StrandsDeciderModel(nn.Module):
    """LLM torso + slot head. The LM output head is never loaded."""

    def __init__(self, config: StrandsDeciderConfig, torso: nn.Module, tokenizer: Any):
        super().__init__()
        self.config = config
        self.torso = torso
        self.tokenizer = tokenizer
        self.head = build_head(config, self.hidden_size(torso))

    @staticmethod
    def hidden_size(torso: nn.Module) -> int:
        cfg = getattr(torso, "config", None)
        for attr in ("hidden_size", "n_embd", "d_model"):
            if cfg is not None and getattr(cfg, attr, None):
                return int(getattr(cfg, attr))
        raise ValueError("could not determine hidden size from torso config")

    @classmethod
    def from_pretrained_base(
        cls,
        config: StrandsDeciderConfig,
        *,
        device_map: str | None = None,
        attn_implementation: str | None = None,
    ) -> StrandsDeciderModel:
        """Load the base model's torso (AutoModel, so no LM head) and attach LoRA."""
        tok = AutoTokenizer.from_pretrained(config.base_model, revision=config.base_revision)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token

        torso = cls._load_torso(config, device_map, attn_implementation)
        model = cls(config, torso, tok)
        if config.use_lora:
            model.attach_lora()
        return model

    @staticmethod
    def _load_torso(
        config: StrandsDeciderConfig,
        device_map: str | None,
        attn_implementation: str | None,
    ) -> nn.Module:
        kwargs: dict[str, Any] = {
            "dtype": getattr(torch, config.torch_dtype),
            "revision": config.base_revision,
        }
        if device_map:
            kwargs["device_map"] = device_map
        if attn_implementation:
            kwargs["attn_implementation"] = attn_implementation
        base_cfg = AutoConfig.from_pretrained(config.base_model, revision=config.base_revision)
        if base_cfg.model_type in {"qwen3_5", "qwen3_5_text"}:
            # Qwen3.5 checkpoints are multimodal; AutoModel would hand back the wrapper with
            # a vision tower. Load the text tower through its causal-LM class, which maps the
            # checkpoint's weight names, and keep only the decoder (the output head is tied to
            # the input embeddings, which the KL reference reads instead).
            import transformers

            lm = transformers.Qwen3_5ForCausalLM.from_pretrained(
                config.base_model, config=base_cfg.get_text_config(), **kwargs
            )
            torso = lm.model
        else:
            torso = AutoModel.from_pretrained(config.base_model, **kwargs)
        torso.config.use_cache = True
        return torso

    @staticmethod
    def is_hybrid(torso: nn.Module) -> bool:
        """True for torsos with recurrent (linear-attention) layers, e.g. Qwen3.5.

        Their per-layer state is not a key/value cache, so the shared-prefix cache in
        infer.py cannot broadcast it yet.
        """
        cfg = getattr(torso, "config", None)  # a PEFT wrapper forwards this to the base
        return "linear_attention" in (getattr(cfg, "layer_types", None) or [])

    def attach_lora(self) -> None:
        from peft import LoraConfig, get_peft_model

        lora_cfg = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=self.config.lora_targets,
            bias="none",
            # No PEFT task head -- our SlotHead is the task head and is saved separately.
            task_type="FEATURE_EXTRACTION",
        )
        self.torso = get_peft_model(self.torso, lora_cfg)

    def freeze_torso(self) -> None:
        for p in self.torso.parameters():
            p.requires_grad_(False)

    def trainable_parameters(self) -> tuple[int, int]:
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        return trainable, total

    def encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        past_key_values: Any = None,
    ) -> torch.Tensor:
        out = self.torso(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=past_key_values is not None,
            return_dict=True,
        )
        return out.last_hidden_state


    # ---- pretrained-readout hooks ------------------------------------------
    def slot_token_ids(self) -> dict[int, int]:
        """slot index -> token id of the option number the prompt shows for it.

        `_option_block` numbers options from 1, and the prompt ends at `<answer>`, so
        the natural continuation for slot k is the token "k+1". Only 1-9 are single
        tokens in the Qwen vocabulary; slots past that have no usable row and are
        omitted, which covers 78.6% of the training corpus and almost every real
        request (JevBench families are mostly 2-5 options).
        """
        out: dict[int, int] = {}
        for k in range(self.config.num_slots):
            ids = self.tokenizer.encode(str(k + 1), add_special_tokens=False)
            if len(ids) == 1:
                out[k] = ids[0]
        return out

    def _output_embedding(self) -> torch.Tensor:
        """The LM output matrix. Qwen3 ties embeddings, so the input table *is* it.

        `AutoModel` never loads `lm_head`, so without the tie there would be nothing
        to read; assert rather than silently seeding from an unrelated matrix.
        """
        base = self.torso
        base = getattr(base, "base_model", base)
        base = getattr(base, "model", base)
        emb = base.get_input_embeddings() if hasattr(base, "get_input_embeddings") else None
        if emb is None:
            raise RuntimeError("torso exposes no input embedding to use as an LM head")
        cfg = getattr(self.torso, "config", None)
        if cfg is not None and not getattr(cfg, "tie_word_embeddings", False):
            raise RuntimeError(
                "base model does not tie embeddings, so the input table is not the LM head"
            )
        return emb.weight

    def init_head_from_lm_head(self) -> int:
        """Seed the slot head with the LM's own option-number readout directions."""
        if self.config.head_hidden:
            raise ValueError("lm_head init applies to the linear head only")
        table = self._output_embedding()
        slots = self.slot_token_ids()
        proj = self.head.proj
        with torch.no_grad():
            rows = table[list(slots.values())].to(torch.float32)
            # Match the scale the randomly-initialised head was given, so the change
            # under test is the *direction* carried by the LM head, not a larger step.
            rows = rows * (proj.weight.std() / rows.std().clamp_min(1e-8))
            for position, slot in enumerate(slots):
                proj.weight[slot].copy_(rows[position].to(proj.weight.device))
            if proj.bias is not None:
                proj.bias[list(slots)] = 0.0
        return len(slots)

    def frozen_slot_log_probs(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor, n_slots: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """The untouched torso's own option-number distribution, and who it covers.

        Adapters are disabled rather than a second copy loaded, so this costs one
        extra forward and no extra weights. Rows needing a slot without a
        single-token number are excluded and reported in the mask.
        """
        slots = self.slot_token_ids()
        usable = max(slots) + 1 if slots else 0
        eligible = n_slots <= usable
        if not bool(eligible.any()):
            return torch.zeros(0, device=input_ids.device), eligible
        table = self._output_embedding()
        with torch.no_grad():
            disable = getattr(self.torso, "disable_adapter", None)
            ctx = disable() if callable(disable) else contextlib.nullcontext()
            with ctx:
                hidden = self.encode(input_ids, attention_mask)
            pooled = pool_last_token(hidden, attention_mask).to(torch.float32)
            rows = table[[slots[k] for k in sorted(slots)]].to(torch.float32)
            logits = pooled @ rows.t()
            pad = self.config.num_slots - logits.size(-1)
            if pad > 0:
                logits = F.pad(logits, (0, pad), value=MASK_VALUE)
            return masked_log_softmax(logits, n_slots), eligible

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        n_slots: torch.Tensor,
        labels: torch.Tensor | None = None,
        label_dist: torch.Tensor | None = None,
        weights: torch.Tensor | None = None,
        past_key_values: Any = None,
        temperature: Any | None = None,
        opt_idx: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        hidden = self.encode(input_ids, attention_mask, past_key_values=past_key_values)
        pooled = pool_last_token(hidden, attention_mask).to(torch.float32)
        if self.config.head_type == "pointer":
            if opt_idx is None:
                raise ValueError("pointer head needs opt_idx (option token positions)")
            # Indices are relative to what was actually forwarded, so with a shared
            # prefix cache the caller passes suffix-relative positions and this works
            # unchanged -- `hidden` is the suffix in that case.
            options = gather_options(hidden, opt_idx).to(torch.float32)
            logits = self.head(pooled, options)
        else:
            logits = self.head(pooled)

        logits = apply_temperature(logits, self.config.temperature
                                   if temperature is None else temperature)

        log_probs = masked_log_softmax(logits, n_slots)
        out: dict[str, torch.Tensor] = {"logits": logits, "log_probs": log_probs}

        per_example: torch.Tensor | None = None
        if label_dist is not None:
            # Soft targets (ordinal smoothing for score questions). Masked entries are
            # -inf in log_probs but carry zero target mass, so zero them before the dot
            # product to avoid 0 * -inf = NaN.
            safe = torch.where(torch.isinf(log_probs), torch.zeros_like(log_probs), log_probs)
            per_example = -(label_dist * safe).sum(dim=-1)
        elif labels is not None:
            per_example = F.nll_loss(log_probs, labels, reduction="none")

        if per_example is not None:
            if weights is None:
                out["loss"] = per_example.mean()
            else:
                # Weighted mean, not weighted sum: keeps the loss scale (and so the
                # effective learning rate) independent of how the weights are set.
                w = weights.to(per_example.dtype)
                out["loss"] = (per_example * w).sum() / w.sum().clamp_min(1e-8)
        return out

    def save_pretrained(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, CONFIG_NAME), "w", encoding="utf-8") as fh:
            fh.write(self.config.to_json())
        torch.save(self.head.state_dict(), os.path.join(path, "slot_head.pt"))
        if self.config.use_lora:
            self.torso.save_pretrained(os.path.join(path, "lora"))
        self.tokenizer.save_pretrained(path)

    @classmethod
    def load(
        cls,
        path: str,
        *,
        device_map: str | None = None,
        attn_implementation: str | None = None,
        trainable: bool = False,
    ) -> StrandsDeciderModel:
        """A saved checkpoint, its adapter frozen unless `trainable` (to continue training it)."""
        path = checkpoint_dir(path)
        config = StrandsDeciderConfig.from_json(config_path(path))
        config.base_revision = base_revision(path, config)
        # Check the checkpoint's own files before the torso loads its 2B weights. Without
        # this check, a checkpoint without its adapter loads and gives other probabilities.
        lora_dir = os.path.join(path, "lora")
        if config.use_lora and not os.path.isdir(lora_dir):
            raise FileNotFoundError(
                f"{lora_dir}: missing, but {os.path.basename(config_path(path))} sets use_lora"
            )
        head_state = load_head_state(path)
        tok = AutoTokenizer.from_pretrained(path)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token

        torso = cls._load_torso(config, device_map, attn_implementation)

        if config.use_lora:
            from peft import PeftModel

            torso = PeftModel.from_pretrained(torso, lora_dir, is_trainable=trainable)

        # __init__ would re-attach LoRA on top of the adapter we just loaded, so build
        # the module directly and restore the head weights in place.
        obj = cls.__new__(cls)
        nn.Module.__init__(obj)
        obj.config = config
        obj.torso = torso
        obj.tokenizer = tok
        obj.head = build_head(config, cls.hidden_size(torso))
        obj.head.load_state_dict(head_state)
        obj.head.to(torch.float32)
        return obj


CONFIG_NAME = "strands_decider_config.json"
# Checkpoints saved before the rename, including the published strands-decider-2B-hobson-v19.
LEGACY_CONFIG_NAME = "hobson_config.json"


def config_path(path: str) -> str:
    """The config file `load` reads in checkpoint directory `path`: CONFIG_NAME, or
    LEGACY_CONFIG_NAME when only that one exists. Writers that update a checkpoint's
    config (calibrate) write to this same file, so `load` sees the update."""
    new = os.path.join(path, CONFIG_NAME)
    if not os.path.exists(new):
        legacy = os.path.join(path, LEGACY_CONFIG_NAME)
        if os.path.exists(legacy):
            return legacy
    return new


def base_revision(path: str, config: StrandsDeciderConfig) -> str | None:
    """The base model revision for checkpoint `path`: the config's, else the one its
    provenance.json (written by every Hugging Face export) records for the same base."""
    prov = os.path.join(path, "provenance.json")
    if config.base_revision or not os.path.exists(prov):
        return config.base_revision
    with open(prov, encoding="utf-8") as fh:
        meta = json.load(fh)
    return meta.get("base_model_revision") if meta.get("base_model") == config.base_model else None


def checkpoint_dir(path: str, revision: str | None = None) -> str:
    """`path` if it is a local directory, else the Hub model repo `path` downloaded to the
    Hub cache (`strands-decider serve <hub-org>/strands-decider-2B-<release>`). If that download fails for any reason (no
    such repo, private or gated, offline, a proxy error, not a valid repo id), the error
    names the path and gives the reason."""
    if os.path.isdir(path):
        return path
    try:
        return str(snapshot_download(path, revision=revision))
    except Exception as e:
        raise FileNotFoundError(
            f"{path}: not a local directory, and could not be downloaded from the "
            f"Hugging Face Hub: {type(e).__name__}: {e}"
        ) from e


def load_head_state(path: str) -> dict[str, torch.Tensor]:
    """The head weights: `head.safetensors` (a Hugging Face export) or the `slot_head.pt`
    that `save_pretrained` writes. Both hold the same state dict. This function refuses a
    directory with both files, because nothing tells which one is correct."""
    st = os.path.join(path, "head.safetensors")
    pt = os.path.join(path, "slot_head.pt")
    if os.path.exists(st) and os.path.exists(pt):
        raise ValueError(f"{path}: holds both head.safetensors and slot_head.pt. Keep only one.")
    if os.path.exists(st):
        from safetensors.torch import load_file

        loaded: dict[str, torch.Tensor] = load_file(st)
        return loaded
    # weights_only=True is the default from torch 2.6. It is passed explicitly anyway.
    state: dict[str, torch.Tensor] = torch.load(pt, map_location="cpu", weights_only=True)
    return state
