"""``strands-decider-vllm serve``: the System One API, images included, on vLLM's engine in process
(README, Serving).

The server has no authentication and binds 127.0.0.1 by default. Expose it only behind the
deployment's own authentication: SageMaker's IAM-signed endpoint or an authenticating proxy.
"""

from __future__ import annotations

import argparse
import time
import uuid
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from starlette.requests import ClientDisconnect
from vllm.exceptions import VLLMValidationError

from . import __version__, logger
from .bodies import BodyReader
from .decider import Decider, Refused
from .settings import Settings, settings

_LOOPBACK_HOSTS = ("127.0.0.1", "::1", "localhost")


def build_app(s: Settings, start_engine: Callable[[], Any]) -> FastAPI:
    """The HTTP app. ``start_engine`` returns vLLM's AsyncLLM, started on the app's event loop."""
    bodies = BodyReader(s.max_body, s.max_buffered, s.body_read_timeout)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        decider = app.state.decider = Decider(start_engine(), s)
        try:
            yield
        finally:
            decider.close()
            decider.engine.shutdown()

    app = FastAPI(title="strands-decider-vllm", version=__version__, lifespan=lifespan)

    async def health(request: Request) -> dict[str, Any]:
        try:
            await request.app.state.decider.engine.check_health()
        except Exception as e:
            logger.error("strands-decider: engine unhealthy: %s", e)
            raise HTTPException(503, "engine unhealthy") from e
        return {
            "status": "ok",
            "model": s.served_name,
            "inputs": list(s.config.inputs),
            "max_length": s.max_length,
            "token_budget": s.token_budget,
            "max_token_budget": s.max_token_budget,
            "max_body_bytes": s.max_body,
            "version": __version__,
        }

    async def systemone(request: Request) -> JSONResponse:
        started = time.perf_counter()
        decider: Decider = request.app.state.decider
        try:
            async with bodies.read(request) as body:
                prepared = await decider.prepare(body)
            del body  # its bytes go with its share of the budget, before the engine runs
            response = await decider.answer(prepared)
        except HTTPException:
            raise
        except ClientDisconnect:
            return JSONResponse({"detail": "the client disconnected"}, status_code=400)
        except (Refused, VLLMValidationError) as e:  # the engine core's refusals, too
            raise HTTPException(422, str(e)) from e
        except Exception:
            error_id = uuid.uuid4().hex[:16]
            logger.exception("strands-decider: request failed (error id %s)", error_id)
            return JSONResponse(
                {"detail": "internal server error", "error_id": error_id}, status_code=500
            )
        response["latency_ms"] = round((time.perf_counter() - started) * 1000, 2)
        return JSONResponse(response)

    app.add_api_route("/health", health, methods=["GET"])
    app.add_api_route("/ping", health, methods=["GET"])  # SageMaker's
    app.add_api_route("/v1/systemone", systemone, methods=["POST"])
    app.add_api_route("/invocations", systemone, methods=["POST"])  # SageMaker's
    return app


def serve(args: argparse.Namespace) -> None:
    import uvicorn
    from vllm.v1.engine.async_llm import AsyncLLM

    if args.host not in _LOOPBACK_HOSTS:
        logger.warning(
            "strands-decider-vllm serve: binding %s. The server has no authentication: anyone who "
            "can reach this address can send it requests. Put the deployment's own authentication "
            "in front of it (SageMaker's IAM-signed endpoint, an authenticating proxy), or bind "
            "127.0.0.1 and use an SSH port forward.",
            args.host,
        )
    s = settings(args)
    uvicorn.run(
        build_app(s, lambda: AsyncLLM.from_engine_args(s.engine_args)),
        host=args.host,
        port=args.port,
        log_level="info",
        limit_concurrency=args.limit_concurrency,
    )
