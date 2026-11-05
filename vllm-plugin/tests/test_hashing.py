"""Content-keyed prefix-cache extra keys: the digests, the wrapper, and which cached blocks a request can reuse."""

from __future__ import annotations

import hashlib
import sys
import types
from array import array
from types import SimpleNamespace

import pytest

from strands_decider_v2_vllm import hashing

STATE_DOMAIN = b"strands-decider-v2/state\0"
QUESTION_DOMAIN = b"strands-decider-v2/question\0"


def make_request(tokens, prefix_end=None, extra_args=None, base_keys=None):
    if extra_args is None and prefix_end is not None:
        extra_args = {"prefix_end": prefix_end}
    return SimpleNamespace(
        prompt_token_ids=list(tokens),
        sampling_params=SimpleNamespace(extra_args=extra_args),
        base_keys=base_keys,
    )


def sha(domain, tokens):
    return hashlib.sha256(domain + array("Q", tokens).tobytes()).digest()


# --- boundary and digests ------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "extra_args, expected",
    [
        (None, None),
        ({}, None),
        ({"other": 1}, None),
        ({"prefix_end": 4}, 4),
        ({"prefix_end": 4.0}, 4),
        ({"prefix_end": 4.5}, None),
        ({"prefix_end": True}, None),
        ({"prefix_end": "4"}, None),
    ],
)
def test_boundary(extra_args, expected):
    assert hashing._boundary(make_request(range(10), extra_args=extra_args)) == expected


def test_boundary_without_sampling_params():
    assert hashing._boundary(SimpleNamespace(prompt_token_ids=[1, 2], sampling_params=None)) is None


def test_digests_cover_state_and_question():
    tokens = [7, 3, 9, 1, 4, 4, 2, 8]
    r = make_request(tokens, prefix_end=5)
    assert hashing._digests(r) == (
        5,
        sha(STATE_DOMAIN, tokens[:5]),
        sha(QUESTION_DOMAIN, tokens[5:]),
    )
    assert hashing._digests(r) is getattr(r, hashing._ATTR), "computed once per request"


@pytest.mark.parametrize("s", [0, 8, 9, -1, None])
def test_no_digests_without_a_valid_boundary(s):
    r = make_request(range(8), prefix_end=s, extra_args=None if s is not None else {})
    assert hashing._digests(r) is None
    assert getattr(r, hashing._ATTR) == (), "the negative result is cached too"
    assert hashing._digests(r) is None


def test_no_digests_without_prompt_tokens():
    r = SimpleNamespace(
        prompt_token_ids=None, sampling_params=SimpleNamespace(extra_args={"prefix_end": 2})
    )
    assert hashing._digests(r) is None


def test_state_and_question_domains_differ():
    # The same tokens as a state and as a question give different digests.
    tokens = [5, 6, 7, 5, 6, 7]
    _, state, question = hashing._digests(make_request(tokens, prefix_end=3))
    assert state != question


# --- the wrapper ---------------------------------------------------------------------------------------------------


@pytest.fixture
def kcu(monkeypatch):
    """A stand-in for vllm.v1.core.kv_cache_utils whose original function returns the request's base keys."""

    def generate_block_hash_extra_keys(request, start_token_idx, end_token_idx, start_mm_idx):
        return request.base_keys, start_mm_idx + 1

    modules = {
        name: types.ModuleType(name)
        for name in ("vllm", "vllm.v1", "vllm.v1.core", "vllm.v1.core.kv_cache_utils")
    }
    modules["vllm"].v1 = modules["vllm.v1"]
    modules["vllm.v1"].core = modules["vllm.v1.core"]
    modules["vllm.v1.core"].kv_cache_utils = modules["vllm.v1.core.kv_cache_utils"]
    modules[
        "vllm.v1.core.kv_cache_utils"
    ].generate_block_hash_extra_keys = generate_block_hash_extra_keys
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    hashing.install_block_hash_keys()
    return modules["vllm.v1.core.kv_cache_utils"]


def test_install_is_idempotent(kcu):
    wrapped = kcu.generate_block_hash_extra_keys
    hashing.install_block_hash_keys()
    assert kcu.generate_block_hash_extra_keys is wrapped


def test_requests_without_boundary_keep_vllm_keys(kcu):
    for base in (None, ("lora",)):
        r = make_request(range(16), base_keys=base)
        for start in (0, 4, 8):
            assert kcu.generate_block_hash_extra_keys(r, start, start + 4, 2) == (base, 3)


def test_keys_per_block(kcu):
    tokens = list(range(100, 116))
    r = make_request(tokens, prefix_end=6)
    state, question = sha(STATE_DOMAIN, tokens[:6]), sha(QUESTION_DOMAIN, tokens[6:])
    f = kcu.generate_block_hash_extra_keys
    assert f(r, 0, 4, 0) == ((("sdv2_state", state),), 1)  # block 0: the whole state
    assert f(r, 4, 8, 0) == ((("sdv2_question", question),), 1)  # ends past S: the whole question
    assert f(r, 8, 12, 5) == ((("sdv2_question", question),), 6)


def test_block_0_past_the_boundary_gets_both_keys_after_vllm_keys(kcu):
    tokens = list(range(8))
    r = make_request(tokens, prefix_end=2, base_keys=("lora", "salt"))
    keys, _ = kcu.generate_block_hash_extra_keys(r, 0, 4, 0)
    assert keys == (
        "lora",
        "salt",
        ("sdv2_state", sha(STATE_DOMAIN, tokens[:2])),
        ("sdv2_question", sha(QUESTION_DOMAIN, tokens[2:])),
    )


def test_state_block_after_block_0_keeps_vllm_keys(kcu):
    r = make_request(range(16), prefix_end=12)
    assert kcu.generate_block_hash_extra_keys(r, 4, 8, 0) == (None, 1)


def block_hashes(kcu, request, block_size=4):
    """vLLM's scheme in miniature: a chain over full blocks of (parent hash, block tokens, extra keys)."""
    tokens, parent, out = request.prompt_token_ids, b"", []
    for start in range(0, len(tokens) - block_size + 1, block_size):
        keys, _ = kcu.generate_block_hash_extra_keys(request, start, start + block_size, 0)
        parent = hashlib.sha256(
            repr((parent, tuple(tokens[start : start + block_size]), keys)).encode()
        ).digest()
        out.append(parent)
    return out


def shared_blocks(a, b):
    n = 0
    while n < min(len(a), len(b)) and a[n] == b[n]:
        n += 1
    return n


def test_cache_reuse(kcu):
    state = list(range(10))  # S = 10; blocks of 4: [0, 4) and [4, 8) hold only state tokens
    question_a, question_b = [50, 51, 52, 53, 54, 55], [60, 61, 62, 63, 64, 65]
    a = block_hashes(kcu, make_request(state + question_a, prefix_end=10))
    assert (
        shared_blocks(a, block_hashes(kcu, make_request(state + question_a, prefix_end=10))) == 4
    ), "exact repeat"
    assert (
        shared_blocks(a, block_hashes(kcu, make_request(state + question_b, prefix_end=10))) == 2
    ), "a further question reuses the state's full blocks and nothing that reaches the question"
    changed = state[:9] + [
        99
    ]  # differs only in the last state token, which every state token attends to
    assert (
        shared_blocks(a, block_hashes(kcu, make_request(changed + question_a, prefix_end=10))) == 0
    )
    assert (
        shared_blocks(a, block_hashes(kcu, make_request(state + question_a, prefix_end=8))) == 0
    ), "other S"
    assert shared_blocks(a, block_hashes(kcu, make_request(state + question_a))) == 0, (
        "causal request"
    )


# --- the vLLM internals the wrapper relies on (skipped without vLLM) ------------------------------------------------


def _names(code):
    names = set(code.co_names)
    for const in code.co_consts:
        if isinstance(const, types.CodeType):
            names |= _names(const)
    return names


def test_vllm_looks_the_function_up_at_call_time():
    real = pytest.importorskip("vllm.v1.core.kv_cache_utils")
    import inspect

    assert list(inspect.signature(real.generate_block_hash_extra_keys).parameters) == [
        "request",
        "start_token_idx",
        "end_token_idx",
        "start_mm_idx",
    ]
    # The request block hasher calls it, or the lookahead variant that calls it, as a module global.
    assert _names(real.get_request_block_hasher.__code__) & {
        "generate_block_hash_extra_keys",
        "generate_lookahead_block_hash_extra_keys",
    }
    if hasattr(real, "generate_lookahead_block_hash_extra_keys"):
        assert "generate_block_hash_extra_keys" in _names(
            real.generate_lookahead_block_hash_extra_keys.__code__
        )
