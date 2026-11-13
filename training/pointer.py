"""v1's pointer head as the trainer's second head (`--head pointer`).

The model is this repository's StrandsDeciderModel (src/strands_decider/modeling.py): a decoder torso (Qwen3, Qwen3.5
or Gemma 4) with a PEFT LoRA adapter, read by the PointerHead, which scores each option from the hidden state of the
last token of the option's line against the prompt's last token. The prompt is v1's numbered render
(strands_decider.prompting.build_prompt) under the causal mask. This module puts that model behind the interface
training/train.py drives, so the step plan, the loss, the schedule and the dev evaluation are the token readout's:

    PointerModel.encode(rows)     renders and tokenizes rows as v1's SystemOneCollator does
    PointerModel(batch)           the head's float32 logits, [rows, 255], -1e9 past each row's options
    augment(rows, rng)            the collator's augmentation, drawn in its order
    target_matrix(rows)           each row's target, in the order its options were rendered
    PointerModel.save(directory)  v1's checkpoint layout, which StrandsDeciderModel.load and v1's engines read

Rows are the trainer's (training/README.md, "The data"). v1's score questions are fixed-order choices whose option
names are the level indices, marked `source.score`; `source.instruction_variants` holds a question's other phrasings.
training/convert.py writes both from v1's examples.
"""

from __future__ import annotations

import json
import os
import random
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from strands_decider.core import MAX_CODES

from . import targets
from .data.format import Example

PROMPT = "numbered"  # v1's render: options numbered from 1, then `<answer>`
AUGMENTATION = "augmentation"  # the key augment() adds to a row: its option order and phrasing
MASKED = -1e9  # the token readout's logit for the codes past a row's options (training/model.py)


def example_from_row(row: dict) -> Example:
    """The v1 example a row stands for, options in their canonical order (noul: false, then true)."""
    question = row["question"]
    kind = "score" if row["source"].get("score") else question["type"]
    criteria = question.get("criteria") or {}
    options: list[list[Any]]
    if kind == "noul":
        options = [["false", criteria.get("false")], ["true", criteria.get("true")]]
    elif kind in ("choice", "score"):
        options = [[name, description] for name, description in criteria.items()]
    else:
        raise ValueError(
            f"{row['id']}: question type {kind!r}; the pointer head takes noul, choice and score"
        )
    label = targets.answer_label(row)
    names = [name for name, _ in options]
    index = int(label is True) if kind == "noul" else names.index(label)
    return Example(
        kind=kind,
        state=row["state"],
        instructions=question.get("instructions") or "",
        options=options,
        label=index,
        task=row["suite"],
        instruction_variants=list(row["source"].get("instruction_variants") or []),
    )


def augment(rows: Sequence[dict], rng: random.Random, reverse_score_prob: float) -> list[dict]:
    """New copies of `rows`, each with the option order and phrasing v1's collator draws for it, from `rng`, in the
    collator's order: per row, the option order first (a score's levels reversed with probability
    `reverse_score_prob`, every other question's options shuffled), then a phrasing among the question's
    instructions and its variants when it has any. A fixed-order choice that is not a score (training/convert.py's
    score levels from decision rows) keeps its order; its shuffle is drawn and dropped, as targets.shuffle_options
    does."""
    out = []
    for row in rows:
        example = example_from_row(row)
        order: list[int] | None
        if example.kind == "score":
            order = (
                list(reversed(range(example.n_options)))
                if rng.random() < reverse_score_prob
                else None
            )
        else:
            order = list(range(example.n_options))
            rng.shuffle(order)
            if row["source"].get("fixed_order"):
                order = None
        pool = example.all_instructions()
        phrasing = rng.choice(pool) if len(pool) > 1 else None
        out.append({**row, AUGMENTATION: {"order": order, "instructions": phrasing}})
    return out


def _order(row: dict) -> list[int] | None:
    order: list[int] | None = (row.get(AUGMENTATION) or {}).get("order")
    return order


def target_matrix(rows: Sequence[dict], device: torch.device) -> torch.Tensor:
    """targets.target_matrix with each row's columns in the order augment() rendered its options: column k holds the
    target of the canonical option order[k]."""
    values = targets.target_matrix(rows, device)
    for index, row in enumerate(rows):
        order = _order(row)
        if order is not None:
            values[index, : len(order)] = values[index, torch.tensor(order, device=device)]
    return values


def load_tokenizer(base: str | Path, force_bos: bool) -> Any:
    """The base's tokenizer as v1's StrandsDeciderModel.from_pretrained_base sets it up: a pad token, and BOS when
    `force_bos`."""
    from transformers import AutoTokenizer

    from strands_decider.modeling import ensure_bos

    tokenizer = AutoTokenizer.from_pretrained(str(base))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if force_bos:
        ensure_bos(tokenizer)
    return tokenizer


def render(row: dict) -> tuple[str, Any, int]:
    """(the prompt, the rendered question, the option count) of a row under its augmentation, if it has one."""
    from strands_decider.prompting import build_prompt

    example = example_from_row(row)
    augmentation = row.get(AUGMENTATION) or {}
    prompt, rendered = build_prompt(
        example.state,
        example.to_question(augmentation.get("instructions")),
        option_order=augmentation.get("order"),
    )
    return prompt, rendered, example.n_options


def prompts(row: dict, every_phrasing: bool = False) -> list[str]:
    """A row's prompt in canonical option order; with `every_phrasing`, one per phrasing augment() may draw."""
    from strands_decider.prompting import build_prompt

    example = example_from_row(row)
    pool = example.all_instructions() if every_phrasing else [example.instructions]
    return [build_prompt(example.state, example.to_question(phrasing))[0] for phrasing in pool]


def prompt_tokens(rows: Sequence[dict], tokenizer: Any, every_phrasing: bool = False) -> list[int]:
    """Each row's prompt length in tokens, in canonical option order; with `every_phrasing`, the longest over its
    phrasings. A row the render refuses raises."""
    texts, owners = [], []
    for index, row in enumerate(rows):
        for text in prompts(row, every_phrasing):
            texts.append(text)
            owners.append(index)
    lengths = [0] * len(rows)
    for start in range(0, len(texts), 1024):
        ids = tokenizer(texts[start : start + 1024])["input_ids"]
        for owner, row_ids in zip(owners[start : start + 1024], ids, strict=True):
            lengths[owner] = max(lengths[owner], len(row_ids))
    return lengths


def measure(rows: list[dict], tokenizer: Any, max_length: int) -> tuple[list[dict], int]:
    """dev_eval.measure for the numbered render: (the rows it renders within `max_length` tokens, each with
    `source.input_tokens` set, and the count dropped)."""
    renderable = []
    for row in rows:
        try:
            prompts(row)
        except (ValueError, TypeError, KeyError):
            continue
        renderable.append(row)
    lengths = prompt_tokens(renderable, tokenizer)
    kept = [
        {**row, "source": {**row["source"], "input_tokens": n}}
        for row, n in zip(renderable, lengths, strict=True)
        if n <= max_length
    ]
    return kept, len(rows) - len(kept)


@dataclass(frozen=True)
class PointerBatch:
    inputs: dict[
        str, torch.Tensor
    ]  # input_ids, attention_mask, n_slots, opt_idx: what StrandsDeciderModel takes
    counts: tuple[int, ...]
    input_tokens: int


class PointerModel(torch.nn.Module):
    """StrandsDeciderModel on the base `base` (a local directory), with a new LoRA adapter and pointer head, behind the
    trainer's interface. The torso is bf16 on a GPU and float32 on the CPU; the head is float32. `base_model` and
    `revision` are what the saved config records, so v1's loader finds the base on the Hub."""

    def __init__(
        self,
        base: str | Path,
        *,
        base_model: str,
        revision: str | None,
        rank: int,
        alpha: float,
        dropout: float,
        lora_targets: Sequence[str],
        pointer_dim: int,
        head_dropout: float,
        force_bos: bool,
        max_length: int,
        ordinal_smoothing: float,
        device: str,
    ) -> None:
        super().__init__()
        from strands_decider.modeling import StrandsDeciderConfig, StrandsDeciderModel

        config = StrandsDeciderConfig(
            base_model=str(base),
            head_dropout=head_dropout,
            max_length=max_length,
            torch_dtype="bfloat16" if device.startswith("cuda") else "float32",
            pointer_dim=pointer_dim,
            ordinal_smoothing=ordinal_smoothing,
            use_lora=True,
            lora_r=rank,
            lora_alpha=int(alpha) if float(alpha).is_integer() else alpha,  # type: ignore[arg-type]
            lora_dropout=dropout,
            lora_targets=list(lora_targets),
            force_bos=force_bos,
        )
        self.decider = StrandsDeciderModel.from_pretrained_base(config)
        self.local_base, self.base_model = str(base), base_model
        # What the checkpoint records, in its config and its adapter's: the base as the Hub knows it, not the local
        # copy it loaded from.
        config.base_model, config.base_revision = base_model, revision
        adapter = self.decider.torso.peft_config["default"]
        adapter.base_model_name_or_path, adapter.revision = base_model, revision
        self.tokenizer = self.decider.tokenizer
        self.device_name = device
        self.to(device)
        self.train(False)

    def gradient_checkpointing_enable(self) -> None:
        """As v1's trainer turns it on: non-reentrant, on the decoder inside the PEFT wrapper."""
        torso = self.decider.torso
        inner = getattr(getattr(torso, "base_model", torso), "model", torso)
        inner.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    def trainable_parameters(self) -> list[torch.nn.Parameter]:
        """The head's parameters, then the adapter's, in module order."""
        head = [p for p in self.decider.head.parameters() if p.requires_grad]
        adapter = [p for p in self.decider.torso.parameters() if p.requires_grad]
        return head + adapter

    def parameter_groups(
        self, lr: float, head_lr: float, weight_decay: float
    ) -> list[dict[str, Any]]:
        """v1's optimizer groups: the head's matrices at `head_lr` with `weight_decay`, its norm gains and biases at
        `head_lr` without, and the adapter at `lr` without."""
        decay = [p for p in self.decider.head.parameters() if p.requires_grad and p.ndim > 1]
        no_decay = [p for p in self.decider.head.parameters() if p.requires_grad and p.ndim <= 1]
        adapter = [p for p in self.decider.torso.parameters() if p.requires_grad]
        groups = [
            {"params": decay, "lr": head_lr, "weight_decay": weight_decay},
            {"params": no_decay, "lr": head_lr, "weight_decay": 0.0},
            {"params": adapter, "lr": lr, "weight_decay": 0.0},
        ]
        return [group for group in groups if group["params"]]

    def encode(self, rows: Sequence[dict], max_length: int = 4096) -> PointerBatch:
        """The batch SystemOneCollator builds for these rows: right padding, each option's last token for the head.
        Prompts are never cut: a row over `max_length` tokens raises."""
        from strands_decider.prompting import option_token_index

        if not rows:
            raise ValueError("encode() got no rows")
        texts, spans, counts = [], [], []
        for row in rows:
            prompt, rendered, count = render(row)
            texts.append(prompt)
            spans.append((len(prompt) - len(rendered.text), rendered.option_spans))
            counts.append(count)
        encoded = self.tokenizer(
            texts,
            padding=True,
            return_tensors="pt",
            padding_side="right",
            return_offsets_mapping=True,
        )
        mask = encoded["attention_mask"]
        if int(mask.sum(dim=1).max()) > max_length:
            raise ValueError(
                f"a prompt of {int(mask.sum(dim=1).max())} tokens is over max_length {max_length}"
            )
        offsets = encoded["offset_mapping"].tolist()
        positions = [option_token_index(offsets[i], s, base) for i, (base, s) in enumerate(spans)]
        width = max(len(p) for p in positions)
        inputs = {
            "input_ids": encoded["input_ids"],
            "attention_mask": mask,
            "n_slots": torch.tensor(counts, dtype=torch.long),
            "opt_idx": torch.tensor(
                [p + [-1] * (width - len(p)) for p in positions], dtype=torch.long
            ),
        }
        return PointerBatch(
            {k: v.to(self.device_name) for k, v in inputs.items()}, tuple(counts), int(mask.sum())
        )

    def forward(self, batch: PointerBatch) -> torch.Tensor:
        """The pointer head's logits at temperature 1, float32, padded to [rows, 255] with -1e9 past each row's
        options, as the token readout's are."""
        logits = self.decider(**batch.inputs, temperature=1.0)["logits"].float()
        rows, width = logits.shape
        padded = logits.new_full((rows, MAX_CODES), MASKED)
        padded[:, :width] = logits
        past = (
            torch.arange(MAX_CODES, device=logits.device)[None]
            >= torch.tensor(batch.counts, device=logits.device)[:, None]
        )
        return padded.masked_fill(past, MASKED)

    @torch.inference_mode()  # type: ignore[untyped-decorator]
    def option_probabilities(
        self, rows: Sequence[dict], batch_size: int = 8, max_length: int = 4096
    ) -> list[list[float]]:
        """Per row, the softmax over its own options in the order they were rendered. For checks and tests."""
        was_training = self.training
        self.eval()
        out: list[list[float]] = []
        try:
            for start in range(0, len(rows), batch_size):
                batch = self.encode(rows[start : start + batch_size], max_length=max_length)
                probabilities = self(batch).softmax(-1).cpu().tolist()
                out.extend(p[:n] for p, n in zip(probabilities, batch.counts, strict=True))
        finally:
            self.train(was_training)
        return out

    def save(self, directory: Path, record: dict[str, Any]) -> None:
        """v1's checkpoint (strands_decider_config.json, head.safetensors, lora/, the tokenizer), plus the training
        record as training.json, which v1's loaders do not read."""
        self.decider.save_pretrained(str(directory))
        # PEFT's model card names the base the adapter was loaded on, here the local copy: name the Hub's instead.
        card = directory / "lora" / "README.md"
        if card.exists():
            card.write_text(card.read_text().replace(self.local_base, self.base_model))
        (directory / "training.json").write_text(json.dumps(record, indent=2) + "\n")

    def load_trained(self, directory: Path) -> None:
        """The head and adapter weights of a checkpoint `save` wrote, into this model (to resume a run)."""
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file

        from strands_decider.modeling import load_head_state

        self.decider.head.load_state_dict(load_head_state(str(directory)))
        adapter = load_file(os.path.join(directory, "lora", "adapter_model.safetensors"))
        result = set_peft_model_state_dict(self.decider.torso, adapter)
        unexpected = list(getattr(result, "unexpected_keys", []) or [])
        if unexpected:
            raise ValueError(
                f"{directory}: adapter tensors this model has no place for: {unexpected[:3]}"
            )
        missing = [k for k in getattr(result, "missing_keys", []) or [] if "lora_" in k]
        if missing:
            raise ValueError(
                f"{directory}: the adapter lacks {len(missing)} tensors, such as {missing[:3]}"
            )
