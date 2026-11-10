"""The errors a decision runtime raises for a request, without dependencies, so that any
backend can raise them and `serve` can map them: a refusal is the caller's (HTTP 422), a busy
engine is temporary (HTTP 503)."""

from __future__ import annotations


class RequestRefused(ValueError):
    """A request the decision interface refuses: the caller's error (HTTP 422). Any other
    failure while answering an accepted request is the runtime's (HTTP 500)."""


class EngineBusy(RuntimeError):
    """No slot on the engine came free within the wait allowed (HTTP 503)."""
