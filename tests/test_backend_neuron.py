"""The Neuron backend on Trainium, on tiny random-weight checkpoints (tests/decision_support.py)
compared with the transformers backend on the CPU; the last test needs
STRANDS_DECIDER_TEST_CHECKPOINT, a model checkpoint. Every test skips without a Neuron device
(tests/neuron_support.py); on a Trn2 instance run them with `pytest -q -m neuron`.

Each backend runs on the NeuronCores from the first one in STRANDS_NEURON_CORES (default 0),
one backend at a time: a TP=2 backend takes that core and the next.

Random weights amplify rounding: in bf16 a tiny model's hidden states move by about 0.1 of
their scale on the CPU alone. So a bf16 result is held to twice the CPU's own bf16 error on the
same prompt, not to a fixed bound, and float32 is held to a fixed one.
"""

from __future__ import annotations

import json
import os
import warnings
from pathlib import Path

import numpy as np
import pytest
from neuron_support import REAL, needs_neuron, needs_real_neuron

from decision_support import GEMMA4_VISION, UNIFIED, image_b64, model_checkpoint, request  # isort: skip

from strands_decider.backends.neuron import NeuronBackend, pack_joint
from strands_decider.backends.transformers import load_backend
from strands_decider.core import (
    NO_VISION,
    DecisionConfig,
    Request,
    load_readout,
    load_tokenizer,
    question_probabilities,
    render,
)
from strands_decider.decision import load_decision_engine
from strands_decider.refusals import RequestRefused

pytestmark = needs_neuron

_ROPE = {
    "full_attention": {
        "partial_rotary_factor": 0.25,
        "rope_theta": 1000000.0,
        "rope_type": "proportional",
    },
    "sliding_attention": {"rope_theta": 10000.0, "rope_type": "default"},
}
# One entry per model: the checkpoint kind and the decoder settings that give a tiny model that
# model's design. The vocabulary splits over the ranks (the real one is 262,144).
LAYOUTS = {
    # 31B: full layers whose values are their keys (attention_k_eq_v) with proportional RoPE on a
    # quarter of their 512-wide heads; sliding layers with 256-wide heads; fewer key/value heads
    # on the full layers (here 1, replicated over two ranks) than on the sliding ones.
    "31B": {
        "kind": GEMMA4_VISION,
        "text": {
            "vocab_size": 262144,
            "hidden_size": 256,
            "intermediate_size": 512,
            "num_hidden_layers": 2,
            "layer_types": ["sliding_attention", "full_attention"],
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "num_global_key_value_heads": 1,
            "head_dim": 256,
            "global_head_dim": 512,
            "attention_k_eq_v": True,
            "rope_parameters": _ROPE,
        },
    },
    # 12B: Gemma4Unified's text model, dense; 16 query heads over 8 key/value heads on the sliding
    # layers (here 4 over 2) and one replicated key/value head on the full ones, keys as values.
    "12B": {
        "kind": UNIFIED,
        "text": {
            "vocab_size": 262144,
            "hidden_size": 256,
            "intermediate_size": 512,
            "num_hidden_layers": 2,
            "layer_types": ["sliding_attention", "full_attention"],
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "num_global_key_value_heads": 1,
            "head_dim": 256,
            "global_head_dim": 512,
            "attention_k_eq_v": True,
            "rope_parameters": _ROPE,
        },
    },
}
FLOAT32 = 5e-3  # float32 on the device against float32 on the CPU (4.3e-4 on the CPU alone)
SAME = 1e-3  # one device graph, the same token positions: only masked keys differ
FIRST_CORE = int(os.environ.get("STRANDS_NEURON_CORES", "0").split("-")[0])
IMAGES = [image_b64(64, 48)]


def _rel(ref, got) -> float:
    """The largest difference, relative to the reference's scale."""
    ref, got = np.asarray(ref, np.float64), np.asarray(got, np.float64)
    return float(np.abs(ref - got).max()) / max(1.0, float(np.abs(ref).max()))


def _bf16_bound(cpu, state, questions) -> float:
    """Twice the CPU's own bf16 error on this prompt, and at least 0.05."""
    truth = cpu["float32"].forward(state, questions)
    return max(0.05, 2 * _rel(truth, cpu["bfloat16"].forward(state, questions)))


class _Devices:
    """The one backend (or engine) alive on the NeuronCores, by key: asking for another closes it."""

    def __init__(self):
        self.key, self.obj, self.close = None, None, None

    def get(self, key, make):
        if key != self.key:
            self.release()
            self.obj, self.close = make()
            self.key = key
        return self.obj

    def release(self):
        if self.close is not None:
            self.close()
        self.key, self.obj, self.close = None, None, None


@pytest.fixture(scope="module")
def devices():
    d = _Devices()
    yield d
    d.release()


@pytest.fixture(scope="module", params=list(LAYOUTS))
def layout(request):
    return request.param


@pytest.fixture(scope="module")
def ckpt(layout, tmp_path_factory) -> Path:
    spec = LAYOUTS[layout]
    path = model_checkpoint(
        tmp_path_factory.mktemp(layout) / "ckpt", spec["kind"], text=spec["text"]
    )
    # The backend reads a sharded checkpoint's index; save_pretrained wrote one file.
    from safetensors import safe_open

    with safe_open(str(path / "model.safetensors"), framework="pt") as f:
        names = list(f.keys())
    index = {"weight_map": dict.fromkeys(names, "model.safetensors")}
    (path / "model.safetensors.index.json").write_text(json.dumps(index))
    return path


@pytest.fixture(scope="module")
def cpu(ckpt):
    """The transformers backend on the CPU, a full pass per question, by dtype."""
    return {
        dt: load_backend(str(ckpt), dtype=dt, reuse_state=False) for dt in ("float32", "bfloat16")
    }


@pytest.fixture(scope="module")
def neuron(ckpt, devices):
    """neuron(tp=1, dtype="bfloat16", **opts): a NeuronBackend that returns hidden states."""

    def get(tp: int = 1, dtype: str = "bfloat16", **opts):
        cores = str(FIRST_CORE) if tp == 1 else f"{FIRST_CORE}-{FIRST_CORE + tp - 1}"
        env = {"NEURON_RT_VISIBLE_CORES": cores}
        if dtype == "float32":
            env["NEURON_CC_FLAGS"] = "--auto-cast=none"  # no float32 matmul in bf16
        settings = {"env": env, "dtype": dtype, "applies_readout": False, **opts}

        def make():
            b = NeuronBackend(str(ckpt), tp=tp, max_length=2048, opts=settings)
            return b, b.close

        return devices.get((str(ckpt), tp, dtype, tuple(sorted(opts.items()))), make)

    return get


def _random_ids(path, *lengths: int, seed: int = 1) -> list[list[int]]:
    vocab, g = load_tokenizer(path).get_vocab_size(), np.random.default_rng(seed)
    return [[int(t) for t in g.integers(10, vocab - 10, n)] for n in lengths]


def _prompt(path, req: dict | None = None):
    """The state's ids and each question's, as the core renders `req`."""
    r = render(Request.model_validate(req or request()), load_tokenizer(path))
    return list(r.state_ids), [list(q.question_ids) for q in r.questions.values()]


# a ------------------------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_forward_agrees_with_the_cpu(ckpt, cpu, neuron, dtype):
    """Every question's final-norm row against the transformers backend's float32 full pass:
    the rendered request's three questions and a 65-token one, packed into one pass."""
    state, questions = _prompt(ckpt)
    questions = [*questions, *_random_ids(ckpt, 65)]
    truth = cpu["float32"].forward(state, questions)
    got = neuron(dtype=dtype).forward(state, questions)
    bound = FLOAT32 if dtype == "float32" else _bf16_bound(cpu, state, questions)
    assert _rel(truth, got) < bound, (_rel(truth, got), bound)


def test_a_one_file_checkpoint_needs_no_index(ckpt, neuron, devices, tmp_path):
    """The 12B ships one model.safetensors and no index: the same rows as the indexed copy."""
    one = tmp_path / "one-file"
    one.mkdir()
    for f in ckpt.iterdir():
        if f.name != "model.safetensors.index.json":
            (one / f.name).symlink_to(f)
    state, questions = _prompt(ckpt)
    indexed = neuron().forward(state, questions)
    devices.release()
    b = NeuronBackend(
        str(one),
        tp=1,
        max_length=2048,
        opts={"env": {"NEURON_RT_VISIBLE_CORES": str(FIRST_CORE)}, "applies_readout": False},
    )
    try:
        assert _rel(indexed, b.forward(state, questions)) < SAME
    finally:
        b.close()


# b ------------------------------------------------------------------------------------------


def test_each_question_is_answered_as_if_alone(ckpt, neuron):
    """A question's row alone, and first among the others in one pass: the same 128-token
    bucket and positions, so only the other questions' keys (masked) differ."""
    backend = neuron()
    state, *questions = _random_ids(ckpt, 40, 5, 9, 13, 30)
    for i, q in enumerate(questions):
        together = [q, *questions[:i], *questions[i + 1 :]]
        assert len(pack_joint(state, [q], pad_id=0)["ids"]) == len(
            pack_joint(state, together, pad_id=0)["ids"]
        )
        alone = backend.forward(state, [q])[0]
        packed = backend.forward(state, together)[0]
        assert _rel(alone, packed) < SAME, (i, _rel(alone, packed))


# c ------------------------------------------------------------------------------------------


def test_the_split_pass_agrees_with_the_joint_one(ckpt, cpu, neuron):
    """A 900-token state past the joint-packing limit (lowered to 256): its state bucket runs in
    the NKI Library's attention_cte. Both passes against the CPU's float32."""
    state, questions = _prompt(ckpt)
    state = state + _random_ids(ckpt, 900 - len(state), seed=2)[0]
    truth = cpu["float32"].forward(state, questions)
    bound = _bf16_bound(cpu, state, questions)
    joint = neuron().forward(state, questions)
    split = neuron(joint_max=256).forward(state, questions)
    assert _rel(truth, joint) < bound, ("joint", _rel(truth, joint), bound)
    assert _rel(truth, split) < bound, ("split", _rel(truth, split), bound)
    assert _rel(joint, split) < bound, ("joint vs split", _rel(joint, split), bound)


# d ------------------------------------------------------------------------------------------


def test_two_ranks_agree_with_one(ckpt, cpu, neuron):
    """Tensor parallelism over two NeuronCores, sequence parallel, against one core."""
    state, questions = _prompt(ckpt)
    bound = _bf16_bound(cpu, state, questions)
    one = neuron(tp=1).forward(state, questions)
    two = neuron(tp=2).forward(state, questions)
    assert _rel(one, two) < bound, (_rel(one, two), bound)


# e ------------------------------------------------------------------------------------------


def test_the_backends_readout_is_the_cores(ckpt, neuron):
    """The backend's readout of its hidden rows answers as the core's readout of the same rows,
    and a pass that applies it returns those probabilities."""
    backend = neuron()
    req = Request.model_validate(request())
    rendered = render(req, load_tokenizer(ckpt))
    state, questions = (
        list(rendered.state_ids),
        [list(q.question_ids) for q in rendered.questions.values()],
    )
    config, weight = DecisionConfig.load(str(ckpt)), load_readout(str(ckpt))
    hidden = backend.forward(state, questions)
    probs = backend.readout(hidden)
    core = question_probabilities(hidden, config, rendered, weight=weight)
    ours = question_probabilities(probs, config, rendered)
    for name in core:
        np.testing.assert_allclose(ours[name], core[name], atol=1e-6, err_msg=name)
    backend.applies_readout = True
    try:
        applied = backend.forward(state, questions)
    finally:
        backend.applies_readout = False
    np.testing.assert_allclose(applied, probs, atol=1e-6)


# f, g ---------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def engines(ckpt, devices):
    """(Neuron engine, CPU float32 engine, CPU bf16 engine) through load_decision_engine."""

    def make():
        with pytest.MonkeyPatch.context() as mp:
            mp.setenv("STRANDS_NEURON_TP", "1")
            mp.setenv("STRANDS_NEURON_CORES", str(FIRST_CORE))
            mp.setenv("STRANDS_NEURON_WARM", "none")
            engine = load_decision_engine(str(ckpt), device="neuron")
        return engine, engine.backend.close

    def get():
        return devices.get(("engine", str(ckpt)), make)

    cpu32 = load_decision_engine(str(ckpt), device="cpu", dtype="float32")
    cpu16 = load_decision_engine(str(ckpt), device="cpu", dtype="bfloat16")
    return get, cpu32, cpu16


def test_images_are_refused_with_the_backends_reason(ckpt, neuron, engines):
    backend = neuron()
    assert not backend.takes_images
    assert backend.image_problem == NO_VISION
    rendered = render(Request.model_validate(request(IMAGES)), load_tokenizer(ckpt))
    assert rendered.images
    with pytest.raises(ValueError, match=NO_VISION):
        backend.forward(list(rendered.state_ids), [[5]], rendered.images)
    engine = engines[0]()
    with pytest.raises(RequestRefused, match=NO_VISION):
        engine.evaluate(Request.model_validate(request(IMAGES)))


def _answers(engine) -> dict:
    return engine.evaluate(Request.model_validate(request())).model_dump()["answers"]


def _distribution(answer: dict) -> dict[str, float]:
    if answer["type"] == "noul":
        return {"true": answer["noul"], "false": 1.0 - answer["noul"]}
    return answer["probabilities"]


def test_a_decision_engine_answers_as_on_the_cpu(engines):
    """`load_decision_engine(..., device="neuron")`: every answer a distribution, within twice
    the CPU's own bf16 error of the CPU's float32, and the same top choice wherever the CPU's
    float32 answer leads by more than that."""
    get, cpu32, cpu16 = engines
    ours, ref, ref16 = _answers(get()), _answers(cpu32), _answers(cpu16)
    assert set(ours) == set(ref)
    for name in ref:
        p, r, r16 = _distribution(ours[name]), _distribution(ref[name]), _distribution(ref16[name])
        assert set(p) == set(r)
        # Each answer probability is rounded to 4 decimals (core.readout), so k of them sum to
        # 1 within k * 5e-5.
        assert abs(sum(p.values()) - 1.0) <= 5e-5 * len(p) + 1e-9, (name, p)
        assert all(0.0 <= v <= 1.0 for v in p.values()), name
        keys = list(r)
        floor = max(abs(r[k] - r16[k]) for k in keys)
        bound = max(0.05, 2 * floor)
        assert max(abs(p[k] - r[k]) for k in keys) < bound, (name, p, r, bound)
        ranked = sorted(keys, key=r.get, reverse=True)
        if r[ranked[0]] - r[ranked[1]] > bound:
            assert max(keys, key=p.get) == ranked[0], (name, p, r)


# h ------------------------------------------------------------------------------------------


class TestRealCheckpoint:
    pytestmark = needs_real_neuron

    def test_its_first_golden_lines_conform(self, devices):
        """The first 20 text lines of the checkpoint's golden_vectors.jsonl against the conformance
        bounds (tests/core_conformance.py); a smoke test, not the 1,000-question check. Runs on
        STRANDS_NEURON_TP cores (16 by default) from STRANDS_NEURON_CORES."""
        from core_conformance import MEAN_ABS_DP, TOP1, agreement, read_golden_vectors

        from strands_decider.backends.neuron import load_backend as load_neuron
        from strands_decider.modeling import checkpoint_dir

        devices.release()
        path = checkpoint_dir(REAL)
        golden = Path(path) / "golden_vectors.jsonl"
        reference = read_golden_vectors(golden)
        lines = []
        with open(golden, encoding="utf-8") as f:
            for number, raw in enumerate(f):
                if raw.strip() and not (line := json.loads(raw))["images"]:
                    lines.append((number, line))
                if len(lines) == 20:
                    break
        tokenizer, config = load_tokenizer(path), DecisionConfig.load(path)
        with pytest.MonkeyPatch.context() as mp, warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mp.setenv("STRANDS_NEURON_WARM", "none")
            backend = load_neuron(path)
        try:
            ours, ref = {}, {}
            for number, line in lines:
                rendered = render(Request.model_validate(line["request"]), tokenizer)
                names = list(line["questions"])
                rows = backend.forward(
                    line["state_ids"], [line["questions"][q]["question_ids"] for q in names]
                )
                probs = question_probabilities(rows, config, rendered)
                for name in names:
                    ours[(number, name)] = probs[name].tolist()
                    ref[(number, name)] = reference[(number, name)]
        finally:
            backend.close()
        a = agreement(ours, ref)
        assert a.top1 >= TOP1 and a.mean_abs_dp <= MEAN_ABS_DP, a
