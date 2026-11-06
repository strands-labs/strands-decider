"""Content-keyed prefix-cache extra keys: the digests, the wrapper, and which cached blocks a request can reuse."""

from __future__ import annotations

import hashlib
import itertools
import sys
import types
from array import array
from types import SimpleNamespace

import pytest

from strands_decider_v2_vllm import hashing

STATE_DOMAIN = b"strands-decider-v2/state\0"
QUESTION_DOMAIN = b"strands-decider-v2/question\0"
_ids = itertools.count()


@pytest.fixture(autouse=True)
def _no_pending():
    yield
    hashing._PENDING.clear()


def make_request(tokens, prefix_end=None, base_keys=None):
    """A request as the hasher sees it; with ``prefix_end`` it is a v2 request the engine core has validated."""
    rid = f"req-{next(_ids)}"
    if prefix_end is not None:
        hashing.expect(rid, list(tokens), prefix_end)
    return SimpleNamespace(request_id=rid, prompt_token_ids=list(tokens), base_keys=base_keys)


def sha(domain, tokens):
    return hashlib.sha256(domain + array("Q", tokens).tobytes()).digest()


# --- digests --------------------------------------------------------------------------------------------------------


def test_digests_cover_state_and_question():
    tokens = [7, 3, 9, 1, 4, 4, 2, 8]
    assert hashing.digests(tokens, 5) == (
        5,
        sha(STATE_DOMAIN, tokens[:5]),
        sha(QUESTION_DOMAIN, tokens[5:]),
    )


def test_a_validated_request_gets_its_digests_once():
    tokens = [7, 3, 9, 1, 4, 4, 2, 8]
    r = make_request(tokens, prefix_end=5)
    assert hashing._digests(r) == hashing.digests(tokens, 5)
    assert hashing._digests(r) is getattr(r, hashing._ATTR), "computed once per request"
    hashing.settle(r)
    assert not hashing._PENDING and hashing._digests(r) == hashing.digests(tokens, 5)


def test_settle_attaches_digests_no_block_has_asked_for_yet():
    r = make_request(range(8), prefix_end=3)
    hashing.settle(r)  # a prompt shorter than one block: vLLM hashed nothing at construction
    assert getattr(r, hashing._ATTR) == hashing.digests(list(range(8)), 3)


def test_a_request_no_v2_engine_validated_has_no_digests():
    r = make_request(range(8))
    assert hashing._digests(r) is None
    assert getattr(r, hashing._ATTR) == (), "the negative result is cached too"
    hashing.settle(r)
    assert hashing._digests(r) is None


def test_state_and_question_domains_differ():
    # The same tokens as a state and as a question give different digests.
    _, state, question = hashing.digests([5, 6, 7, 5, 6, 7], 3)
    assert state != question


# --- the wrapper (a stand-in for vLLM's module) ---------------------------------------------------------------------


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


def test_requests_without_digests_keep_vllm_keys(kcu):
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


# --- the vLLM internals the wrapper relies on, and vLLM's real hasher (skipped without vLLM) ------------------------


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


@pytest.fixture
def vllm_hasher():
    kcu = pytest.importorskip("vllm.v1.core.kv_cache_utils")
    from vllm.sampling_params import SamplingParams
    from vllm.utils.hashing import sha256
    from vllm.v1.request import Request

    hashing.install_block_hash_keys()
    kcu.init_none_hash(sha256)
    hasher = kcu.get_request_block_hasher(16, sha256)

    def build(tokens, s=None, cache_salt=None):
        """Request.from_engine_core_request's path for a request the v2 engine core validated (s) or not."""
        rid = f"real-{next(_ids)}"
        sp = SamplingParams(max_tokens=1, extra_args=None if s is None else {"prefix_end": s})
        if s is not None:
            hashing.expect(rid, list(tokens), s)
        r = Request(
            request_id=rid,
            prompt_token_ids=list(tokens),
            sampling_params=sp,
            pooling_params=None,
            cache_salt=cache_salt,
            block_hasher=hasher,
        )
        hashing.settle(r)
        return r

    return kcu, sha256, build


def test_vllms_hasher_with_the_wrapper(vllm_hasher):
    kcu, sha256, build = vllm_hasher
    state, qa, qb = list(range(1000, 1040)), list(range(2000, 2030)), list(range(3000, 3030))
    # 16-token blocks: [0, 16) and [16, 32) end inside the state (S = 40); [32, 48) and later reach the question.
    a = build(state + qa, s=40).block_hashes
    assert len(a) == 4
    assert build(state + qa, s=40).block_hashes == a, "exact repeat: every block"
    assert shared_blocks(a, build(state + qb, s=40).block_hashes) == 2, (
        "further question: the state's blocks"
    )
    assert shared_blocks(a, build(state + qa, s=39).block_hashes) == 0, "other S"
    plain = build(state + qa)
    assert shared_blocks(a, plain.block_hashes) == 0, (
        "a causal request shares nothing with a v2 request"
    )
    # A request no v2 engine validated keeps vLLM's own hashes, exactly.
    parent, own = None, []
    for i in range(4):
        parent = kcu.hash_block_tokens(sha256, parent, (state + qa)[16 * i : 16 * i + 16], None)
        own.append(parent)
    assert plain.block_hashes == own
    # A cache salt still separates v2 requests with the same tokens and S.
    assert shared_blocks(a, build(state + qa, s=40, cache_salt="x").block_hashes) == 0
    assert not hashing._PENDING


def test_vllms_hasher_extends_a_v2_requests_hashes_with_the_digests(vllm_hasher):
    kcu, sha256, build = vllm_hasher
    prompt = list(range(10))
    r = build(prompt, s=4)  # shorter than one block: nothing hashed at construction
    assert r.block_hashes == [] and getattr(r, hashing._ATTR)[0] == 4
    r.append_output_token_ids(list(range(500, 506)))  # 16 tokens now: block 0 is hashed later
    r.update_block_hashes()
    _, state, question = hashing.digests(prompt, 4)
    expected = kcu.hash_block_tokens(
        sha256,
        None,
        prompt + list(range(500, 506)),
        (("sdv2_state", state), ("sdv2_question", question)),
    )
    assert r.block_hashes == [expected], "block 0 carries the prompt's digests after the fact too"
