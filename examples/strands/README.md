# Strands decider inside a Strands agent

Two worked examples of strands decider inside a [Strands](https://github.com/strands-agents/sdk-python)
agent. Both run locally on the default Bedrock model plus the locally served decider, from a clone
of the repository, and expect a server on port 8099 unless `STRANDS_DECIDER_URL` is set.

## As tools: `agent_tools.py`

```bash
pip install -e ".[strands]"
strands-decider serve StrandsAgents/strands-decider-2B-hobson-v19 --port 8099
python examples/strands/agent_tools.py
```

The eleven `@tool` functions under `strands_decider.tools` go in the agent's tool list, so the chat
model can ask the decider for a decision (`decider_route`, `decider_classify`, `decider_sift`,
`decider_rate`, ...) and branch on a calibrated number instead of its own opinion. The first half
of the script calls the tools directly through `agent.tool.<name>(...)`, which needs no LLM and is
how to use the decider from plain Python with the SDK's validation and result shape; the second
half hands four tickets to the agent and lets it decide which tools to call. The tools section of
the top-level [README](../../README.md#as-tools-the-agent-calls) lists them.

## As a gate: `tool_call_intervention.py`

**Prerequisites:** Strands drives the agent with an LLM call to Amazon Bedrock, so you need
AWS credentials with Bedrock access in your environment before you run the example. See the
[Strands Bedrock model provider docs](https://strandsagents.com/latest/documentation/docs/user-guide/concepts/model-providers/amazon-bedrock/)
for the supported credential options.

```bash
git clone https://github.com/strands-labs/strands-decider && cd strands-decider
pip install -e . strands-agents
strands-decider serve StrandsAgents/strands-decider-2B-hobson-v19 --port 8099

python examples/strands/tool_call_intervention.py
```

The scenario is deliberately small. The agent has one `get_weather` tool and a system prompt
that makes it eager, and the user asks "*What's the weather?*" without saying where. The agent
guesses a city and calls the tool anyway. Before that call runs, strands decider reads the
conversation and the proposed call and answers two narrow yes/no questions about it:

1. Are the argument values grounded in facts the user actually provided?
2. Is it premature to call this tool now, before clarifying with the user?

A few lines of plain Python turn those probabilities into a decision, and the agent goes back
to ask which city you meant instead of confidently reporting the weather somewhere nobody
mentioned.

The seam is Strands' intervention system. You write an `InterventionHandler` with a
`before_tool_call` method and pass it to the agent, and it runs before any tool executes. What
you return is a typed action rather than text: the example uses `Proceed` to let the call run
and `Guide` to hand the model back its turn with feedback, which is what makes this different
from a hard allow/deny gate — `Guide` corrects the course instead of refusing. The handler is
an ordinary Python class and Strands has no opinion about what goes inside it, so the same
shape holds whether you call a decision model, a policy engine, or another agent.

Why classify instead of letting the agent check itself? The gate is a local forward pass in a
couple of hundred milliseconds, the conversation never leaves the machine, and each verdict is
a number your `if` statements branch on rather than prose the agent can negotiate with.

> This example is an illustration rather than a recommendation: the questions, the thresholds
> and the policy were all picked by hand. The point is that a decision this cheap can sit in a
> path where an LLM call never could.
