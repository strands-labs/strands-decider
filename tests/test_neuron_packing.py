"""The Neuron backend's packed pass gives each question the prefix_bidir mask of its own prompt
(core.attention_mask), at its own positions, and reads it at its last token. Host-only: no
NeuronCore, no torch."""

from __future__ import annotations

import numpy as np
import pytest

from strands_decider.backends.neuron import pack, pack_joint, passes
from strands_decider.backends.neuron_model import STATE
from strands_decider.core import attention_mask


def _joint_allowed(qid: np.ndarray) -> np.ndarray:
    """neuron_model.decoder_layer's joint mask, in numpy."""
    return (qid[None, :] == STATE) | ((qid[:, None] == qid[None, :]) & (qid[None, :] >= 0))


def _split_allowed(b: dict) -> np.ndarray:
    """neuron_model.decoder_layer's split masks as one dense [T, T] matrix."""
    sb, pb = int(b["state_bucket"]), int(b["question_bucket"])
    t = sb + pb
    m = np.zeros((t, t), dtype=bool)
    m[:sb, :sb] = b["state_valid"][None, :]
    qid = b["qid"]
    m[sb:, :sb] = b["state_valid"][None, :]
    m[sb:, sb:] = (qid[:, None] == qid[None, :]) & (qid[None, :] >= 0)
    return m


@pytest.mark.parametrize("joint", [True, False])
def test_each_question_sees_its_own_prompt(joint: bool) -> None:
    rng = np.random.default_rng(0)
    state = list(rng.integers(10, 1000, 37))
    questions = [list(rng.integers(10, 1000, n)) for n in (5, 1, 12)]
    b = pack_joint(state, questions, pad_id=0) if joint else pack(state, questions, pad_id=0)
    allowed = _joint_allowed(b["qid"]) if joint else _split_allowed(b)
    s = len(state)
    state_idx = np.arange(s)
    for i, q in enumerate(questions):
        if joint:
            q_idx = s + sum(len(x) for x in questions[:i]) + np.arange(len(q))
        else:
            q_idx = int(b["state_bucket"]) + sum(len(x) for x in questions[:i]) + np.arange(len(q))
        idx = np.concatenate([state_idx, q_idx])
        # The rows and columns of this question's prompt, inside the packed pass ...
        sub = allowed[np.ix_(idx, idx)]
        np.testing.assert_array_equal(sub, attention_mask(s, s + len(q)))
        # ... see nothing else (no other question, no padding).
        others = np.setdiff1d(np.arange(allowed.shape[0]), idx)
        assert not allowed[np.ix_(q_idx, others)].any()
        np.testing.assert_array_equal(b["ids"][idx], state + q)
        np.testing.assert_array_equal(b["positions"][idx], np.arange(s + len(q)))
        assert b["rows"][i] == q_idx[-1]
    # The state never sees a question.
    assert not allowed[
        np.ix_(state_idx, np.setdiff1d(np.arange(allowed.shape[0]), state_idx))
    ].any()


def test_passes_split_on_the_bucket() -> None:
    qs = [[1] * 300, [1] * 300, [1] * 300]
    assert passes(qs, 640, 128) == [[0, 1], [2]]
    assert passes(qs, 4096, 2) == [[0, 1], [2]]
    with pytest.raises(ValueError):
        passes([[1] * 5000], 4096, 128)


@pytest.mark.parametrize("joint", [True, False])
def test_shared_layer_rows_see_their_own_prompt(joint: bool) -> None:
    """neuron_model.rows_of (into the KV-shared layers): each read row's mask over the packed
    keys equals its prompt's last row of core.attention_mask, and sees nothing else."""
    torch = pytest.importorskip("torch")
    from strands_decider.backends.neuron_model import rows_of

    rng = np.random.default_rng(1)
    state = list(rng.integers(10, 1000, 37))
    questions = [list(rng.integers(10, 1000, n)) for n in (5, 1, 12)]
    b = pack_joint(state, questions, pad_id=0) if joint else pack(state, questions, pad_id=0)
    sb, t = int(b["state_bucket"]), len(b["ids"])
    tab = torch.zeros(t, 4)
    _, bias, *_ = rows_of(
        torch.zeros(t, 2),
        torch.from_numpy(b["positions"]).long(),
        torch.from_numpy(b["qid"]),
        torch.from_numpy(b["state_valid"]),
        torch.from_numpy(b["rows"]).long(),
        tab,
        tab,
        tab,
        tab,
        state_len=sb,
    )
    allowed = (bias == 0).numpy()
    s, off = len(state), len(state) if joint else sb
    for i, q in enumerate(questions):
        q_idx = off + sum(len(x) for x in questions[:i]) + np.arange(len(q))
        idx = np.concatenate([np.arange(s), q_idx])
        np.testing.assert_array_equal(allowed[i, idx], attention_mask(s, s + len(q))[-1])
        assert not allowed[i, np.setdiff1d(np.arange(t), idx)].any()
