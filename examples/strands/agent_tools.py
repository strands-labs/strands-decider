"""Hand the decider to a Strands agent as tools, so the agent can *ask* for a decision.

The intervention example puts the decider in a seam the agent never sees. This one does the
opposite: the eleven ``@tool`` functions under ``strands_decider.tools`` go in the agent's
tool list, and the chat model calls them the way it calls any tool -- to triage a ticket, to
filter a list, to rank candidates -- getting back a calibrated number instead of its own
opinion. A chat model proposes; the decider judges.

    strands-decider serve StrandsAgents/strands-decider-2B-hobson-v19 --port 8099
    python examples/strands/agent_tools.py

The first half needs no LLM at all: ``agent.tool.<name>(...)`` calls a tool directly, which is
how to use the decider from plain Python with the SDK's validation and result shape. The
second half needs AWS credentials for Bedrock to drive the agent; it is skipped when a call
fails. STRANDS_DECIDER_URL overrides the endpoint (default http://127.0.0.1:8099); set
STRANDS_DECIDER_CHECKPOINT instead to load the model in this process without a server.
"""

from __future__ import annotations

import json
import sys

from strands import Agent

from strands_decider.tools import ALL_TOOLS, DeciderUnavailable, HttpDecider, configure, from_env

TICKETS = [
    "Help! My payouts have been failing for 3 days!",
    "Could you tell me your opening hours on Saturday?",
    "The app crashes every time I open the settings page.",
    "I want to upgrade to the annual plan, is there a discount?",
]


def show(title: str, result: dict) -> None:
    print(f"\n{title}")
    print(f"  {result['content'][-1]['text']}")
    if result["status"] != "success":
        return
    print("  " + json.dumps(result["content"][0]["json"], ensure_ascii=False)[:400])


def main() -> int:
    try:
        decider = from_env()
    except DeciderUnavailable:
        decider = HttpDecider()  # the default port; the first call says how to start one
    configure(decider)

    agent = Agent(tools=ALL_TOOLS, callback_handler=None)

    # ---- direct calls: the SDK validates the input and shapes the result, no LLM involved
    info = agent.tool.decider_info()
    if info["status"] != "success":
        print(info["content"][0]["text"])
        return 1
    show("decider_info", info)

    show(
        "decider_route: one ticket, two questions, one pass",
        agent.tool.decider_route(
            state=TICKETS[0],
            handlers={
                "billing": "payments, payouts, invoices and refunds",
                "technical": "bugs, crashes and errors in the product",
                "sales": "plans, pricing, upgrades and discounts",
                "general": "opening hours, addresses and other questions",
            },
        ),
    )
    show(
        "decider_classify: every ticket, grouped",
        agent.tool.decider_classify(
            items=TICKETS,
            question="Which team should handle this ticket?",
            options={"billing": "money", "technical": "bugs", "sales": "plans", "general": "other"},
        ),
    )
    show(
        "decider_sift: which tickets need a reply today",
        agent.tool.decider_sift(
            items=TICKETS,
            question="Does this ticket need a reply today?",
            criteria={"true": "the writer is blocked or losing money", "false": "it can wait"},
        ),
    )
    show(
        "decider_rate: how frustrated",
        agent.tool.decider_rate(
            state=TICKETS[0],
            question="How frustrated is the writer?",
            levels=["calm", "frustrated", "furious"],
        ),
    )
    show("decider_usage", agent.tool.decider_usage())

    # ---- through the chat model: the agent decides when to ask the decider
    print("\nasking the agent (needs Bedrock credentials; skipped if the call fails)")
    try:
        reply = agent(
            "Here are four support tickets. Use the decider tools, not your own judgement, to say "
            "which team takes each one and which ones need a reply today:\n- "
            + "\n- ".join(TICKETS)
        )
    except Exception as exc:  # the example should still have shown the direct calls
        print(f"  skipped: {type(exc).__name__}: {exc}")
        return 0
    print(f"\n--- the agent's reply ---\n{reply}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
