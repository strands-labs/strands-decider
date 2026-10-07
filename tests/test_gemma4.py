"""Gemma 4 as a torso, on tiny random-weight Gemma 4 text models (no download):

- HostEmbedding keeps a table on the CPU through model moves, and looks up the same rows;
- the frozen option-number readout applies Gemma's final logit soft-cap, so it matches
  the model's own output head;
- the shared-prefix cache forks sliding-window layers (a prefix longer than the window)
  and KV-shared layers, and agrees with encoding each full prompt;
- ensure_bos makes a tokenizer that adds no BOS (gemma-4-E2B-it's) add one.
"""

from __future__ import annotations

import pytest
import torch

transformers = pytest.importorskip("transformers")
if not hasattr(transformers, "Gemma4TextConfig"):
    pytest.skip("this transformers has no Gemma 4", allow_module_level=True)

from strands_decider.infer import _expand_cache  # noqa: E402
from strands_decider.modeling import (  # noqa: E402
    HostEmbedding,
    StrandsDeciderConfig,
    StrandsDeciderModel,
)

DEV = "cuda" if torch.cuda.is_available() else "cpu"
LAYERS = ["sliding_attention", "sliding_attention", "full_attention", "sliding_attention"]


def _lm(num_kv_shared_layers=1, softcap=30.0):
    torch.manual_seed(0)
    cfg = transformers.Gemma4TextConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=len(LAYERS),
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        global_head_dim=16,
        hidden_size_per_layer_input=8,
        vocab_size_per_layer_input=128,
        layer_types=LAYERS,
        sliding_window=4,
        num_kv_shared_layers=num_kv_shared_layers,
        final_logit_softcapping=softcap,
        tie_word_embeddings=True,
        pad_token_id=0,
    )
    return transformers.Gemma4ForCausalLM(cfg).eval()


class _Digits:
    """Just enough tokenizer for slot_token_ids: "1".."9" -> ids 11..19."""

    def encode(self, text, add_special_tokens=False):
        return [10 + int(text)] if text.isdigit() and len(text) == 1 else [1, 2]


def test_host_embedding_stays_on_cpu_and_matches():
    table = _lm().model.embed_tokens_per_layer
    ids = torch.tensor([[3, 7, 9, 2]])
    want = table(ids)
    host = HostEmbedding(table)
    model = torch.nn.Sequential(host)
    model.to(DEV)
    assert host.inner.weight.device.type == "cpu"
    got = host(ids.to(DEV))
    assert got.device.type == DEV and torch.equal(got.cpu(), want)
    model.to(torch.float64)  # a dtype cast is applied, still on the CPU
    assert host.inner.weight.dtype == torch.float64 and host.inner.weight.device.type == "cpu"
    assert StrandsDeciderConfig().host_embeddings is False


def test_frozen_readout_applies_softcap():
    lm = _lm(softcap=2.0)  # a small cap makes the difference visible on random weights
    config = StrandsDeciderConfig(base_model="tiny", num_slots=4, head_type="pointer")
    model = StrandsDeciderModel(config, lm.model, _Digits())
    ids = torch.tensor([[5, 6, 7, 8, 9, 20, 21]])
    mask = torch.ones_like(ids)
    lp, eligible = model.frozen_slot_log_probs(ids, mask, torch.tensor([3]))
    assert bool(eligible[0])
    with torch.no_grad():
        logits = lm(input_ids=ids, attention_mask=mask).logits[0, -1]  # soft-capped by Gemma
    want = torch.log_softmax(logits[[11, 12, 13]].float(), -1)
    assert torch.allclose(lp[0, :3], want, atol=1e-4)


@pytest.mark.parametrize("shared", [0, 1], ids=["no-kv-sharing", "kv-sharing"])
@torch.inference_mode()
def test_shared_prefix_matches_full_prompts(shared):
    torso = _lm(num_kv_shared_layers=shared).model.to(DEV)
    g = torch.Generator().manual_seed(1)
    prefix = torch.randint(1, 128, (11,), generator=g).tolist()  # longer than the window (4)
    suffixes = [torch.randint(1, 128, (n,), generator=g).tolist() for n in (3, 5, 2)]
    pids = torch.tensor([prefix], device=DEV)
    pc = torso(
        input_ids=pids, attention_mask=torch.ones_like(pids), use_cache=True, return_dict=True
    ).past_key_values
    cache = _expand_cache(pc, len(suffixes))
    width = max(map(len, suffixes))
    ids = torch.tensor([s + [0] * (width - len(s)) for s in suffixes], device=DEV)
    mask = torch.tensor([[1] * len(s) + [0] * (width - len(s)) for s in suffixes], device=DEV)
    prefix_mask = torch.ones(len(suffixes), len(prefix), dtype=mask.dtype, device=DEV)
    full_mask = torch.cat([prefix_mask, mask], 1)
    shared_out = torso(
        input_ids=ids, attention_mask=full_mask, past_key_values=cache, return_dict=True
    ).last_hidden_state
    for i, s in enumerate(suffixes):
        whole = torch.tensor([prefix + s], device=DEV)
        ref = torso(
            input_ids=whole, attention_mask=torch.ones_like(whole), return_dict=True
        ).last_hidden_state
        assert torch.allclose(shared_out[i, : len(s)], ref[0, len(prefix) :], atol=1e-4), i


def _word_tokenizer(add_bos: bool):
    """A tiny fast tokenizer that, like gemma-4-E2B-it's, may add no BOS of its own."""
    from tokenizers import Tokenizer, models, pre_tokenizers, processors

    vocab = {"<bos>": 0, "<pad>": 1, "[UNK]": 2, "x": 3, "y": 4}
    core = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    core.pre_tokenizer = pre_tokenizers.Whitespace()
    if add_bos:
        core.post_processor = processors.TemplateProcessing(
            single="<bos> $A", pair="<bos> $A $B", special_tokens=[("<bos>", 0)]
        )
    return transformers.PreTrainedTokenizerFast(
        tokenizer_object=core, bos_token="<bos>", pad_token="<pad>", unk_token="[UNK]"
    )


def test_ensure_bos_adds_bos_only_with_special_tokens(tmp_path):
    from strands_decider.modeling import ensure_bos

    tok = _word_tokenizer(add_bos=False)
    assert tok("x y")["input_ids"] == [3, 4]
    ensure_bos(tok)
    assert tok("x y")["input_ids"] == [0, 3, 4]
    # Shared-prefix suffixes are encoded without special tokens and must stay BOS-free.
    assert tok("x y", add_special_tokens=False)["input_ids"] == [3, 4]
    ensure_bos(tok)  # idempotent
    assert tok("x y")["input_ids"] == [0, 3, 4]
    tok.save_pretrained(tmp_path)  # a checkpoint's tokenizer keeps it
    assert transformers.AutoTokenizer.from_pretrained(tmp_path)("x")["input_ids"] == [0, 3]


def test_ensure_bos_leaves_a_bos_adding_tokenizer_alone():
    from strands_decider.modeling import ensure_bos

    tok = _word_tokenizer(add_bos=True)
    before = tok.backend_tokenizer.post_processor.__getstate__()
    ensure_bos(tok)
    assert tok.backend_tokenizer.post_processor.__getstate__() == before
    assert tok("x")["input_ids"] == [0, 3]
    assert StrandsDeciderConfig().force_bos is False  # off unless a config asks for it
