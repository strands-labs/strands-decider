"""Test only, never for serving: on PYTHONPATH, every Python process of the server accepts v2 requests with
``max_tokens`` above 1 (all other checks stay). The GPU check ``client.py resume`` uses it to make vLLM preempt v2
requests in the middle of a generation and resume them with their output tokens, which a real server never does,
and to check that the workers plan such steps without stopping the engine."""

import strands_decider_v2_vllm.validation as _validation

_check_sampling = _validation.check_sampling


class _OneToken:
    def __init__(self, sp):
        self._sp = sp

    def __getattr__(self, name):
        return 1 if name == "max_tokens" else getattr(self._sp, name)


def _check_sampling_any_length(sp):
    _check_sampling(_OneToken(sp))


_validation.check_sampling = _check_sampling_any_length
