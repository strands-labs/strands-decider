"""Tiny models and corpora for the CPU tests of the pointer head and of v1's tools: a byte-level BPE tokenizer
trained in the test, a randomly initialised Qwen3 torso, and v1 examples of every question kind, so nothing is
downloaded."""

from __future__ import annotations

import random
from pathlib import Path

import torch

from training.data.format import Example

WORDS = (
    "the contract states that payment is due within thirty days of delivery and any "
    "dispute must be raised in writing before the invoice date which party breached "
    "clause obligations notice termination refund shipping order customer seller "
    "question answer evidence claim supported refuted unknown true false option"
).split()


# ---------------------------------------------------------------- fixtures


def _text(rng: random.Random, n_words: int) -> str:
    return " ".join(rng.choice(WORDS) for _ in range(n_words)).capitalize() + "."


def _examples(n: int, seed: int) -> list:
    rng = random.Random(seed)
    out = []
    for k in range(n):
        kind = ("noul", "choice", "score")[k % 3]
        if kind == "noul":
            options = [["false", _text(rng, 4)], ["true", _text(rng, 4)]]
        elif kind == "choice":
            m = rng.choice([2, 3, 5, 11, 12])  # >9 options: no frozen-KL reference
            options = [[f"opt{j}", _text(rng, 3)] for j in range(m)]
        else:
            m = rng.choice([3, 5])
            options = [[str(j), _text(rng, 3)] for j in range(m)]
        out.append(
            Example(
                kind=kind,
                # lengths from a few words to a few hundred, so length grouping matters
                state=_text(rng, rng.choice([5, 20, 60, 150])),
                instructions=_text(rng, 6),
                options=options,
                label=rng.randrange(len(options)),
                task=f"task{k % 4}",
                weight=rng.choice([1.0, 1.0, 0.5, 2.0]),
                instruction_variants=[_text(rng, 6)] if k % 2 else [],
            )
        )
    return out


def _tokenizer_file(path: str, examples: list, vocab_size: int = 400) -> None:
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors, trainers

    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    tok.post_processor = processors.ByteLevel(trim_offsets=True)
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<pad>", "<eos>"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    from strands_decider.prompting import build_prompt

    texts = [build_prompt(ex.state, ex.to_question())[0] for ex in examples]
    tok.train_from_iterator(texts, trainer)
    tok.save(path)


def _load_tokenizer(path: str):
    from transformers import PreTrainedTokenizerFast

    return PreTrainedTokenizerFast(tokenizer_file=path, pad_token="<pad>", eos_token="<eos>")


def _qwen3_config(vocab_size: int):
    from transformers import Qwen3Config

    return Qwen3Config(
        vocab_size=vocab_size,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=4096,
        tie_word_embeddings=True,
    )


def _tiny_torso(out: str, tokenizer_path: str) -> str:
    """Save a tiny random Qwen3 torso, the base model of `_tiny_checkpoint`, to `out`."""
    from transformers import Qwen3Model

    with torch.random.fork_rng():
        torch.manual_seed(0)
        Qwen3Model(_qwen3_config(len(_load_tokenizer(tokenizer_path)))).save_pretrained(out)
    return out


def _load_saved_torso(torso_dir: str):
    """Stands in for StrandsDeciderModel._load_torso: the torso `_tiny_torso` saved."""
    from transformers import Qwen3Model

    return lambda config, *a: Qwen3Model.from_pretrained(torso_dir)


def _tiny_checkpoint(out: str, tokenizer_path: str, torso_dir: str, seed: int = 0, **over) -> str:
    """A checkpoint as training saves one, over the torso in `torso_dir`: a LoRA adapter and
    a pointer head with random weights (lora_B and the head's LayerNorm too, so that each
    changes the outputs) and fitted temperatures."""
    from strands_decider.modeling import StrandsDeciderConfig, StrandsDeciderModel

    cfg = StrandsDeciderConfig(
        **{
            **dict(
                base_model="tiny",
                pointer_dim=16,
                lora_r=2,
                lora_alpha=4,
                lora_dropout=0.0,
                torch_dtype="float32",
                max_length=2048,
                temperature=1.7,
                temperature_by_kind={"choice": 0.8},
            ),
            **over,
        }
    )
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        torso = _load_saved_torso(torso_dir)(cfg)
        model = StrandsDeciderModel(cfg, torso, _load_tokenizer(tokenizer_path))
        model.attach_lora()
        with torch.no_grad():
            for n, p in model.named_parameters():
                if "lora_" in n or n.startswith("head.norm"):
                    p.normal_(std=0.5)
    model.save_pretrained(out)
    return out


def tiny_base(directory: Path, tokenizer_path: str) -> Path:
    """A base checkpoint directory as the trainer loads one with --base: the tiny Qwen3 torso of `_tiny_torso` and its
    tokenizer files (tokenizer.json among them)."""
    _tiny_torso(str(directory), tokenizer_path)
    _load_tokenizer(tokenizer_path).save_pretrained(str(directory))
    return directory


def _tiny_model_factory(tokenizer_path: str):
    """Stands in for StrandsDeciderModel.from_pretrained_base: same construction, tiny torso."""
    from transformers import Qwen3Model

    from strands_decider.modeling import StrandsDeciderModel

    def build(config, *, device_map=None, attn_implementation=None):
        tok = _load_tokenizer(tokenizer_path)
        torso = Qwen3Model(_qwen3_config(len(tok)))  # float32, random after train()'s seed
        config.lora_dropout = 0.0  # dropout masks cannot match across world sizes
        model = StrandsDeciderModel(config, torso, tok)
        if config.use_lora:
            model.attach_lora()
        return model

    return build


def convert(workdir: Path, base: Path, examples: list, *flags: str) -> Path:
    """`examples` written to workdir/examples.jsonl and converted for the pointer head on `base`; their weights are
    set to 1, which the converter requires."""
    from training import convert as converter
    from training.data.format import write_jsonl

    for ex in examples:
        ex.weight = 1.0
    write_jsonl(str(workdir / "examples.jsonl"), examples)
    out = workdir / "rows"
    converter.main(
        [
            "--examples",
            str(workdir / "examples.jsonl"),
            "--base",
            str(base),
            "--out",
            str(out),
            "--prompt",
            "numbered",
            "--max-length",
            "2048",
            *flags,
        ]
    )
    return out


POINTER_FLAGS = (
    "--head", "pointer", "--prompt", "numbered", "--revision", "0" * 40, "--rank", "2", "--alpha", "4",
    "--pointer-dim", "16", "--rows-per-step", "8", "--batch-size", "4", "--token-budget", "8192",
    "--max-length", "2048", "--cpu-threads", "1",
)  # fmt: skip


def train_pointer(rows: Path, base: Path, output: Path, *flags: str) -> Path:
    """One run of the trainer's pointer head, in this process and on the CPU (any GPU hidden), on rows `convert`
    wrote; returns the run directory."""
    from unittest import mock

    from training import train

    argv = [
        "--base", str(base), "--base-model", str(base), "--train", str(rows / "train.jsonl"),
        "--dev", str(rows / "dev.jsonl"), "--output", str(output), *POINTER_FLAGS, *flags,
    ]  # fmt: skip
    with mock.patch("torch.cuda.is_available", return_value=False):
        train.main(argv)
    return output
