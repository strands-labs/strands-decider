"""Training render "strands-chat": the prompt is core.render's, the shuffle is its option_orders,
fixed_order rows keep their order, and a row's soft target reaches the loss as label_dist."""

from __future__ import annotations

import json

import pytest
import torch
import torch.nn.functional as F
from core_support import gemma4_tokenizer, toy_tokenizer
from transformers import PreTrainedTokenizerFast

from strands_decider import core
from strands_decider.modeling import StrandsDeciderConfig
from training.data.collate import CollatorConfig, SystemOneCollator
from training.data.format import Example, load_examples
from training.data.multistep import fits

ROWS = [
    Example(
        "choice",
        "a user asks for a refund",
        "Which team?",
        [["billing", "money"], ["tech", None], ["sales", "deals"], ["legal", "law"]],
        2,
    ),
    Example(
        "noul",
        "the server is down",
        "It is urgent.",
        [["false", "not urgent"], ["true", "urgent"]],
        1,
        target=[0.3, 0.7],
    ),
    Example(
        "choice", "rate it", "", [["bad", None], ["ok", None], ["good", None]], 0, fixed_order=True
    ),
    Example(
        "choice",
        "pick one",
        "Which?",
        [["x", "first"], ["y", "second"], ["z", "third"]],
        1,
        target=[0.1, 0.6, 0.3],
    ),
]


@pytest.fixture(params=["toy", "gemma4"])
def tok(request):
    raw = toy_tokenizer() if request.param == "toy" else gemma4_tokenizer()
    return PreTrainedTokenizerFast(tokenizer_object=raw, pad_token="<eos>")


def _collate(tok, train, rows=ROWS):
    coll = SystemOneCollator(tok, CollatorConfig(render="strands-chat", seed=3), train=train)
    orders = []
    draw = coll._option_order
    coll._option_order = lambda ex: orders.append(draw(ex)) or orders[-1]  # type: ignore[method-assign]
    return coll(rows), orders


def _core(tok, ex, order=None):
    orders = None if order is None else {"q": order}
    return core.render(ex.to_request(), tok.backend_tokenizer, core.CHAT, option_orders=orders)


@pytest.mark.parametrize("train", [False, True])
def test_the_batch_is_core_render_and_the_shuffle_is_its_option_orders(tok, train):
    batch, orders = _collate(tok, train)
    for i, (ex, order) in enumerate(zip(ROWS, orders, strict=True)):
        q = _core(tok, ex, order).questions["q"]
        n = int(batch["attention_mask"][i].sum())
        assert batch["input_ids"][i, :n].tolist() == list(q.ids)
        assert int(batch["prefix_end"][i]) == q.state_len
        if order is not None:  # the target follows its option to the new slot
            assert q.options[int(batch["labels"][i])] == ex.options[ex.label][0]
    if train:
        assert orders[0] is not None and orders[3] is not None  # choices shuffle (seed 3)
        assert orders[1] is None and orders[2] is None  # noul never; fixed_order never
        assert sorted(orders[0]) == [0, 1, 2, 3]


def test_soft_targets_reach_label_dist_in_slot_order(tok):
    batch, orders = _collate(tok, train=True)
    d = batch["label_dist"]
    assert d[1, :2].tolist() == pytest.approx([0.3, 0.7])
    assert d[3, :3].tolist() == pytest.approx([[0.1, 0.6, 0.3][k] for k in orders[3]])
    for i in (0, 2):  # rows without a target: one-hot on the label
        assert d[i].tolist() == F.one_hot(batch["labels"][i], d.size(1)).float().tolist()


def test_one_hot_rows_train_as_before(tok):
    rows = [ROWS[0], ROWS[2]]
    batch, _ = _collate(tok, train=True, rows=rows)
    assert "label_dist" not in batch  # plain NLL, as before
    # and in a batch with soft rows, a one-hot row's loss is the NLL to the bit
    logp = torch.log_softmax(torch.randn(2, 4), -1)
    onehot = F.one_hot(batch["labels"], 4).float()
    assert torch.equal(
        -(onehot * logp).sum(-1), F.nll_loss(logp, batch["labels"], reduction="none")
    )


def test_fits_measures_the_strands_chat_prompt(tok):
    n = [len(_core(tok, ex).questions["q"].ids) for ex in ROWS]
    assert fits(ROWS, tok, max(n), core.CHAT) == ROWS
    assert fits(ROWS, tok, max(n) - 1, core.CHAT) == [
        e for e, k in zip(ROWS, n, strict=True) if k < max(n)
    ]
    one = Example("choice", "s", "q", [["only", None], ["two", None]], 0)
    one.options = [["only", None]]  # core.render refuses a choice of one option
    assert fits([one], tok, 10_000, core.CHAT) == []


def test_token_readout_takes_strands_chat():
    StrandsDeciderConfig(head_type="token_readout", render="strands-chat")
    with pytest.raises(ValueError):
        StrandsDeciderConfig(head_type="pointer", render="strands-chat")


def test_the_v2_converter_keeps_targets_and_fixed_order_and_drops_refusals(tmp_path):
    from training.data import strands_v2

    def row(q, target, label, **source):
        return {"suite": "s", "state": "st", "question": q, "target": target, "label": label, "source": source}  # fmt: skip

    rows = [
        row({"type": "noul", "criteria": {"false": None, "true": "t"}}, 0.25, False),
        row({"type": "choice", "criteria": {"a": None, "b": "x"}}, [0, 1], "b", fixed_order=True),
        row({"type": "choice", "criteria": {"a": None}}, [1], "a"),  # one option: refused
    ]
    src, out = tmp_path / "v2.jsonl", tmp_path / "ex.jsonl"
    src.write_text("".join(json.dumps(r) + "\n" for r in rows) + '{"cut', encoding="utf-8")
    counts = strands_v2.main([str(src), str(out)])
    assert counts == {"kept": 2, "not_renderable": 1, "not_json": 1}
    noul, choice = load_examples([str(out)])
    assert (noul.label, noul.target, noul.instructions) == (0, [0.75, 0.25], "")
    assert (choice.label, choice.target, choice.fixed_order) == (1, None, True)
