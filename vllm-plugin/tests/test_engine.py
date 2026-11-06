"""The engine-core wrapper: a v2 engine refuses invalid requests with vLLM's per-request validation error before they are
scheduled, with prefix caching on or off; any other engine is untouched."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from strands_decider_v2_vllm import ARCH, hashing
from strands_decider_v2_vllm.engine import serves_v2, wrap_preprocess_add_request


class ValidationError(Exception):
    def __init__(self, message, *, parameter=None, value=None):
        super().__init__(message)
        self.parameter, self.value = parameter, value


class FakeRequest:
    """What vLLM's Request.from_engine_core_request does here: keep the id and hash the blocks at construction."""

    def __init__(self, request, hasher):
        self.request_id = request.request_id
        self.prompt_token_ids = request.prompt_token_ids
        self.digests_seen = hasher(self) if hasher else None


class FakeEngineCore:
    def __init__(self, architectures, caching=True):
        self.vllm_config = SimpleNamespace(
            model_config=SimpleNamespace(architectures=architectures)
        )
        self.request_block_hasher = hashing._digests if caching else None
        self.calls = 0

    def preprocess_add_request(self, request):
        self.calls += 1
        return FakeRequest(request, self.request_block_hasher), 0


def engine_request(prefix_end=9, request_id="r1", **sp_overrides):
    sp = dict(
        extra_args=None if prefix_end is None else {"prefix_end": prefix_end},
        max_tokens=1,
        temperature=1.0,
        top_k=-1,
        top_p=1.0,
        min_p=0.0,
        repetition_penalty=1.0,
        presence_penalty=0.0,
        frequency_penalty=0.0,
        logit_bias=None,
        bad_words=None,
        structured_outputs=None,
        allowed_token_ids=[7, 8],
        logprobs=2,
    )
    sp.update(sp_overrides)
    return SimpleNamespace(
        request_id=request_id,
        prompt_token_ids=list(range(20)),
        prompt_embeds=None,
        prompt_is_token_ids=None,
        sampling_params=SimpleNamespace(**sp),
        resumable=False,
    )


@pytest.fixture
def wrapped():
    preprocess = wrap_preprocess_add_request(FakeEngineCore.preprocess_add_request, ValidationError)
    yield preprocess
    hashing._PENDING.clear()


@pytest.mark.parametrize("caching", [True, False])
@pytest.mark.parametrize(
    "request_overrides, parameter",
    [
        ({"prefix_end": "9"}, "vllm_xargs.prefix_end"),
        ({"prefix_end": 0}, "vllm_xargs.prefix_end"),
        ({"prefix_end": 20}, "vllm_xargs.prefix_end"),
        ({"prefix_end": None}, "vllm_xargs.prefix_end"),
        ({"max_tokens": 300}, "max_tokens"),
    ],
)
def test_a_v2_engine_refuses_before_vllm_builds_the_request(
    wrapped, caching, request_overrides, parameter
):
    core = FakeEngineCore([ARCH], caching=caching)
    with pytest.raises(ValidationError) as err:
        wrapped(core, engine_request(**request_overrides))
    assert err.value.parameter == parameter and core.calls == 0
    assert not hashing._PENDING


def test_a_valid_request_gets_its_digests_while_vllm_hashes_it(wrapped):
    core = FakeEngineCore([ARCH])
    req, wave = wrapped(core, engine_request(prefix_end=9))
    assert wave == 0 and core.calls == 1
    assert req.digests_seen == hashing.digests(list(range(20)), 9), (
        "the hasher saw them at construction"
    )
    assert getattr(req, hashing._ATTR) == req.digests_seen
    assert not hashing._PENDING, "nothing left pending"


def test_without_prefix_caching_nothing_is_registered(wrapped):
    core = FakeEngineCore([ARCH], caching=False)
    req, _ = wrapped(core, engine_request(prefix_end=9))
    assert core.calls == 1 and req.digests_seen is None and not hashing._PENDING


def test_a_failing_vllm_preprocess_leaves_nothing_pending(wrapped):
    class Failing(FakeEngineCore):
        def preprocess_add_request(self, request):
            raise RuntimeError("grammar")

    preprocess = wrap_preprocess_add_request(Failing.preprocess_add_request, ValidationError)
    with pytest.raises(RuntimeError):
        preprocess(Failing([ARCH]), engine_request())
    assert not hashing._PENDING


@pytest.mark.parametrize(
    "architectures", [["Gemma4ForCausalLM"], ["StrandsDeciderV2CausalControlForCausalLM"]]
)
def test_other_engines_are_untouched(wrapped, architectures):
    core = FakeEngineCore(architectures)
    # Even a request the v2 checks would refuse passes straight to vLLM, and gets no digests.
    req, _ = wrapped(core, engine_request(prefix_end="9", max_tokens=300))
    assert core.calls == 1 and req.digests_seen is None and getattr(req, hashing._ATTR) == ()


def test_serves_v2_is_computed_once():
    core = FakeEngineCore([ARCH])
    assert serves_v2(core)
    core.vllm_config = None  # not read again
    assert serves_v2(core)


# --- against vLLM (skipped without it) ------------------------------------------------------------------------------


def test_install_wraps_vllms_engine_core_once():
    core = pytest.importorskip("vllm.v1.engine.core")
    from strands_decider_v2_vllm.engine import install_request_validation

    install_request_validation()
    wrapped_fn = core.EngineCore.preprocess_add_request
    install_request_validation()
    assert core.EngineCore.preprocess_add_request is wrapped_fn
    assert wrapped_fn._strands_decider_v2
    # The engine core's input thread calls it on every ADD and turns VLLMValidationError into a per-request error.
    import inspect

    assert "VLLMValidationError" in inspect.getsource(core.EngineCoreProc.process_input_sockets)
    assert "self.preprocess_add_request(" in inspect.getsource(
        core.EngineCoreProc.process_input_sockets
    )


def test_vllm_maps_the_refusal_to_a_validation_error_output():
    exceptions = pytest.importorskip("vllm.exceptions")
    outputs = pytest.importorskip("vllm.outputs")
    from strands_decider_v2_vllm.validation import RequestRefused

    refused = RequestRefused("x", parameter="max_tokens", value=300)
    err = exceptions.VLLMValidationError(
        str(refused), parameter=refused.parameter, value=refused.value
    )
    # EngineCore._handle_request_validation_error sends RequestError(code="validation_error"), which the frontend
    # raises as VLLMValidationError (HTTP 400).
    back = outputs.RequestError(code="validation_error", message=str(err)).to_exception()
    assert isinstance(back, exceptions.VLLMValidationError)
    assert "strands-decider-v2: x" in str(back) and "parameter=max_tokens" in str(back)
