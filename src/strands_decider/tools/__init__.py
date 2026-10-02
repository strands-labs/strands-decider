"""Strands Agents tools around the decider: ``Agent(tools=ALL_TOOLS)``.

Ten ``@tool`` functions a Strands agent can call, each returning JSON to branch on and one
line to read. ``decider_ask`` is the raw System One call; the rest fix the question type or
compose questions and keep their thresholds as arguments. ``HttpDecider`` reaches a
``strands-decider serve`` process; ``LocalDecider`` runs the engine in this process.

    from strands import Agent
    from strands_decider.tools import ALL_TOOLS

    agent = Agent(tools=ALL_TOOLS)   # STRANDS_DECIDER_URL or STRANDS_DECIDER_CHECKPOINT set
    agent("Is this ticket urgent, and which team should take it? ...")

Needs the ``strands`` extra: ``pip install "strands-decider[strands]"``.
"""

from __future__ import annotations

from ._client import (
    ENV_CHECKPOINT,
    ENV_URL,
    DeciderEndpoint,
    DeciderUnavailable,
    DeciderUsage,
    HttpDecider,
    LocalDecider,
    from_env,
)
from ._common import DECIDER_STATE_KEY, configure
from .batch import BATCH_TOOLS, decider_classify, decider_rank, decider_sift
from .patterns import PATTERN_TOOLS, decider_fan_out, decider_route
from .primitives import PRIMITIVE_TOOLS, decider_ask, decider_info, decider_usage
from .single import SINGLE_TOOLS, decider_check, decider_choose, decider_rate

ALL_TOOLS = [*PRIMITIVE_TOOLS, *SINGLE_TOOLS, *BATCH_TOOLS, *PATTERN_TOOLS]

__all__ = [
    "ALL_TOOLS",
    "BATCH_TOOLS",
    "DECIDER_STATE_KEY",
    "ENV_CHECKPOINT",
    "ENV_URL",
    "PATTERN_TOOLS",
    "PRIMITIVE_TOOLS",
    "SINGLE_TOOLS",
    "DeciderEndpoint",
    "DeciderUnavailable",
    "DeciderUsage",
    "HttpDecider",
    "LocalDecider",
    "configure",
    "decider_ask",
    "decider_check",
    "decider_choose",
    "decider_classify",
    "decider_fan_out",
    "decider_info",
    "decider_rank",
    "decider_rate",
    "decider_route",
    "decider_sift",
    "decider_usage",
    "from_env",
]
