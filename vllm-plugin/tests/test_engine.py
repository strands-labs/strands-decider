"""The engine-core wrapper: an engine serving the plugin's model refuses invalid requests with vLLM's per-request
validation error before they are scheduled, with prefix caching on or off; any other engine is untouched."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from strands_decider_vllm import ARCH, ARCH_MULTIMODAL, ARCH_UNIFIED, SERVER_CONFIG_KEY, hashing
from strands_decider_vllm import engine as engine_mod
from strands_decider_vllm.engine import (
    Policy,
    policy,
    serves_decider,
    started_by_server,
    wrap_preprocess_add_request,
    wrap_scheduler_init,
)


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
    def __init__(self, architectures, caching=True, policy=None):
        self.vllm_config = SimpleNamespace(
            model_config=SimpleNamespace(architectures=architectures)
        )
        self.request_block_hasher = hashing._digests if caching else None
        self.calls = 0
        if policy is not None:
            self.__dict__[engine_mod._POLICY] = policy

    def preprocess_add_request(self, request):
        self.calls += 1
        return FakeRequest(request, self.request_block_hasher), 0


def engine_request(prefix_end=9, request_id="r1", mm_features=None, **sp_overrides):
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
        mm_features=mm_features,
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
def test_a_decider_engine_refuses_before_vllm_builds_the_request(
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
    "architectures", [["Gemma4ForCausalLM"], ["StrandsDeciderCausalControlForCausalLM"]]
)
def test_other_engines_are_untouched(wrapped, architectures):
    core = FakeEngineCore(architectures)
    # Even a request the plugin's checks would refuse passes straight to vLLM, and gets no digests.
    req, _ = wrapped(core, engine_request(prefix_end="9", max_tokens=300))
    assert core.calls == 1 and req.digests_seen is None and getattr(req, hashing._ATTR) == ()


def test_serves_decider_is_computed_once():
    core = FakeEngineCore([ARCH])
    assert serves_decider(core)
    core.vllm_config = None  # not read again
    assert serves_decider(core)


# --- against vLLM (skipped without it) ------------------------------------------------------------------------------


def test_install_wraps_vllms_engine_core_once():
    core = pytest.importorskip("vllm.v1.engine.core")
    sched = pytest.importorskip("vllm.v1.core.sched.scheduler")
    from strands_decider_vllm.engine import install_request_validation

    install_request_validation()
    wrapped_fn, init = core.EngineCore.preprocess_add_request, sched.Scheduler.__init__
    install_request_validation()
    assert core.EngineCore.preprocess_add_request is wrapped_fn
    assert sched.Scheduler.__init__ is init and init._strands_decider
    assert wrapped_fn._strands_decider
    # The engine core's input thread calls it on every ADD and turns VLLMValidationError into a per-request error.
    import inspect

    assert "VLLMValidationError" in inspect.getsource(core.EngineCoreProc.process_input_sockets)
    assert "self.preprocess_add_request(" in inspect.getsource(
        core.EngineCoreProc.process_input_sockets
    )


def test_vllm_maps_the_refusal_to_a_validation_error_output():
    exceptions = pytest.importorskip("vllm.exceptions")
    outputs = pytest.importorskip("vllm.outputs")
    from strands_decider_vllm.validation import RequestRefused

    refused = RequestRefused("x", parameter="max_tokens", value=300)
    err = exceptions.VLLMValidationError(
        str(refused), parameter=refused.parameter, value=refused.value
    )
    # EngineCore._handle_request_validation_error sends RequestError(code="validation_error"), which the frontend
    # raises as VLLMValidationError (HTTP 400).
    back = outputs.RequestError(code="validation_error", message=str(err)).to_exception()
    assert isinstance(back, exceptions.VLLMValidationError)
    assert "strands-decider: x" in str(back) and "parameter=max_tokens" in str(back)


# --- images ---------------------------------------------------------------------------------------------------------


def image(identifier, offset=1, length=6):
    return SimpleNamespace(
        modality="image",
        identifier=identifier,
        mm_position=SimpleNamespace(offset=offset, length=length),
    )


SERVER = Policy(takes_images=True, accepts_images=True)


@pytest.mark.parametrize("arch", [ARCH_MULTIMODAL, ARCH_UNIFIED])
def test_an_image_request_gets_digests_over_its_images(wrapped, arch):
    core = FakeEngineCore([arch], policy=SERVER)
    req, _ = wrapped(core, engine_request(prefix_end=9, mm_features=[image("red")]))
    assert req.digests_seen == hashing.digests(list(range(20)), 9, (("red", 1, 6),))
    other, _ = wrapped(
        core, engine_request(prefix_end=9, mm_features=[image("blue")], request_id="r2")
    )
    assert other.digests_seen[1] != req.digests_seen[1], (
        "same tokens, another image: another state digest"
    )


@pytest.mark.parametrize("caching", [True, False])
@pytest.mark.parametrize(
    "policy, features, match",
    [
        (Policy(False, True), [image("a")], 'lacks "image"'),
        (Policy(True, False), [image("a")], "strands-decider-vllm serve"),
        (SERVER, [image("a", offset=5, length=6)], "inside the state"),  # ends at 11 > S = 9
        (SERVER, [image(str(i), offset=1 + i, length=1) for i in range(5)], "at most 4"),
        (SERVER, [SimpleNamespace(**{**vars(image("a")), "modality": "audio"})], "not audio"),
    ],
)
def test_bad_image_requests_are_refused_before_vllm_builds_them(
    wrapped, caching, policy, features, match
):
    core = FakeEngineCore([ARCH_MULTIMODAL], caching=caching, policy=policy)
    with pytest.raises(ValidationError, match=match) as err:
        wrapped(core, engine_request(prefix_end=9, mm_features=features))
    assert core.calls == 0 and not hashing._PENDING
    assert err.value.parameter in ("images", "multi_modal_data")


def test_the_boundary_is_checked_before_the_images(wrapped):
    core = FakeEngineCore([ARCH_MULTIMODAL], policy=Policy(False, False))
    with pytest.raises(ValidationError) as err:
        wrapped(core, engine_request(prefix_end="9", mm_features=[image("a")]))
    assert err.value.parameter == "vllm_xargs.prefix_end"


def test_a_request_without_mm_features_fails_loudly(wrapped):
    # Were vLLM to rename the attribute, the image checks and digests must not be skipped silently.
    req = engine_request(prefix_end=9)
    del req.mm_features
    with pytest.raises(AttributeError):
        wrapped(FakeEngineCore([ARCH_MULTIMODAL], policy=SERVER), req)


def test_an_unreadable_policy_refuses_the_image_request_and_the_engine_keeps_serving(
    wrapped, monkeypatch
):
    def unreadable(vllm_config):
        raise OSError("decision_config.json: connection reset")

    monkeypatch.setattr(engine_mod, "compute_policy", unreadable)
    core = FakeEngineCore([ARCH_MULTIMODAL])
    with pytest.raises(ValidationError) as err:
        wrapped(core, engine_request(prefix_end=9, mm_features=[image("a")]))
    assert str(err.value) == "strands-decider: " + engine_mod.UNREADABLE_POLICY
    assert "connection reset" not in str(err.value), "the error's own text stays in the log"
    assert err.value.parameter == "multi_modal_data" and core.calls == 0
    assert wrapped(core, engine_request(prefix_end=9, request_id="r2"))[0], "text is served"


class FakeScheduler:
    def __init__(self, vllm_config):
        self.vllm_config = vllm_config


def test_the_policy_is_computed_when_the_engine_builds_its_scheduler(monkeypatch):
    seen = []
    monkeypatch.setattr(
        engine_mod, "compute_policy", lambda vc: seen.append(vc) or Policy(True, True)
    )
    init = wrap_scheduler_init(FakeScheduler.__init__)
    decider = SimpleNamespace(model_config=SimpleNamespace(architectures=[ARCH_MULTIMODAL]))
    other = SimpleNamespace(model_config=SimpleNamespace(architectures=["Gemma4ForCausalLM"]))
    s = FakeScheduler.__new__(FakeScheduler)
    init(s, decider)
    assert seen == [decider] and s.__dict__[engine_mod._POLICY] == Policy(True, True)
    o = FakeScheduler.__new__(FakeScheduler)
    init(o, other)
    assert seen == [decider] and engine_mod._POLICY not in o.__dict__
    # The engine core reads the scheduler's policy instead of computing it again.
    core = FakeEngineCore([ARCH_MULTIMODAL])
    core.scheduler = s
    monkeypatch.setattr(engine_mod, "compute_policy", lambda vc: pytest.fail("computed again"))
    assert policy(core) == Policy(True, True)


def test_a_policy_that_cannot_be_computed_at_startup_stops_the_engine(monkeypatch):
    def unreadable(vllm_config):
        raise ValueError("strands-decider: decision_config.json: inputs is required")

    monkeypatch.setattr(engine_mod, "compute_policy", unreadable)
    init = wrap_scheduler_init(FakeScheduler.__init__)
    vc = SimpleNamespace(model_config=SimpleNamespace(architectures=[ARCH_MULTIMODAL]))
    with pytest.raises(ValueError, match="inputs is required"):
        init(FakeScheduler.__new__(FakeScheduler), vc)


def test_a_text_request_never_reads_the_policy(wrapped, monkeypatch):
    monkeypatch.setattr(
        engine_mod, "policy", lambda core: pytest.fail("policy read for a text request")
    )
    req, _ = wrapped(FakeEngineCore([ARCH_MULTIMODAL]), engine_request(prefix_end=9))
    assert req.digests_seen == hashing.digests(list(range(20)), 9)


@pytest.mark.parametrize(
    "additional, started",
    [
        ({}, False),
        ({SERVER_CONFIG_KEY: "0.3.0"}, True),
        ({SERVER_CONFIG_KEY: ""}, False),
        (None, False),
    ],
)
def test_the_server_marks_its_engines(additional, started):
    assert started_by_server(SimpleNamespace(additional_config=additional)) is started
