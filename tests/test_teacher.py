"""The teacher must see exactly the prompt its benchmark score was measured on."""
from __future__ import annotations

import pytest

from strands_decider.data.format import Example
from strands_decider.data.teacher import LETTERS, render, to_row

NOUL = Example(kind="noul", state="WIN a FREE cruise", instructions="Is this spam?",
               options=[["false", "genuine"], ["true", "promotional"]], label=1, task="spam")
CHOICE = Example(kind="choice", state="Arsenal won 3-1.", instructions="Which section?",
                 options=[["World", "international"], ["Sports", ""]], label=1, task="ag_news")
SCORE = Example(kind="score", state="Great food.", instructions="How positive?",
                options=[["0", "bad"], ["1", "ok"], ["2", "good"]], label=2, task="yelp")


def test_rows_match_the_jevbench_semif_adapter_mapping():
    row, order = to_row(NOUL)
    # true first, as the adapter lists it; order maps teacher position -> canonical index
    assert [o["id"] for o in row["options"]] == ["true", "false"]
    assert [o["description"] for o in row["options"]] == ["true: promotional", "false: genuine"]
    assert order == [1, 0]
    row, order = to_row(CHOICE)
    # an empty description falls back to the option name, as the adapter does
    assert [o["description"] for o in row["options"]] == ["World: international", "Sports: Sports"]
    assert order == [0, 1]
    row, _ = to_row(SCORE)
    assert [o["description"] for o in row["options"]] == ["0: bad", "1: ok", "2: good"]


def test_too_many_options_get_no_teacher():
    many = Example(kind="choice", state="s", instructions="q",
                   options=[[f"o{i}", ""] for i in range(len(LETTERS) + 1)], label=0)
    assert to_row(many) is None


@pytest.mark.parametrize("ex", [NOUL, CHOICE, SCORE])
def test_prompt_is_byte_identical_to_semif(ex):
    semif = pytest.importorskip("semif_phase1.direct")
    transformers = pytest.importorskip("transformers")
    try:  # same chat template family as the teacher; skip if not cached offline
        tok = transformers.AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B", local_files_only=True)
    except OSError:
        pytest.skip("Qwen3 tokenizer not cached")
    row, _ = to_row(ex)
    ids, _, _ = semif.encode_prompt(tok, row, 4096)
    assert tok.encode(render(tok, row), add_special_tokens=False) == ids


class _StubTok:
    """Just enough tokenizer for label(): letters are ids 100.., other text is char codes."""

    pad_token_id = 0

    def apply_chat_template(self, messages, **_):
        return "".join(m["content"] for m in messages)[:48]

    def encode(self, text, add_special_tokens=False):
        if len(text) == 1 and text in LETTERS:
            return [100 + LETTERS.index(text)]
        return [1 + ord(c) % 90 for c in text]

    def decode(self, ids):
        return LETTERS[ids[0] - 100]


def test_label_matches_the_models_own_softcapped_head():
    """A soft-capped LM (Gemma) must be read through its cap, as its own head does."""
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    if not hasattr(transformers, "Gemma4TextConfig"):
        pytest.skip("this transformers has no Gemma 4")
    from strands_decider.data.teacher import label

    torch.manual_seed(0)
    cfg = transformers.Gemma4TextConfig(
        vocab_size=128, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
        num_attention_heads=2, num_key_value_heads=1, head_dim=16, global_head_dim=16,
        hidden_size_per_layer_input=8, vocab_size_per_layer_input=128,
        layer_types=["sliding_attention", "full_attention"], sliding_window=4,
        num_kv_shared_layers=0, final_logit_softcapping=2.0, tie_word_embeddings=True,
        pad_token_id=0)
    model = transformers.Gemma4ForCausalLM(cfg).eval()
    tok = _StubTok()
    probs = label(model, tok, [CHOICE], log_every=0)[0]
    row, _ = to_row(CHOICE)
    ids = torch.tensor([tok.encode(render(tok, row))])
    with torch.inference_mode():
        own = model(input_ids=ids).logits[0, -1, [100, 101]].float().softmax(-1).tolist()
    assert probs == pytest.approx(own, abs=1e-4)
