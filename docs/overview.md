# Overview

System One models understand natural language and make a decision based on that
information. They do not generate text — they return a typed answer from a known
output space along with calibrated confidence.

Strands Decider is a System One model. It supports three question types:
**Choice**, **Noul** (yes/no), and **Score**.

## Question types

### Choice

Given a list of options, pick one.

> **Example**
>
> _State:_ "What color is this fruit: Banana"
>
> _Choices:_ Red, Yellow, or Blue?
>
> _Strands Decider:_ **Yellow**

Try it yourself:

```bash
pip install strands-decider

strands-decider ask StrandsAgents/strands-decider-2B-qwen3.5-v1-2610 \
  --state "What color is this fruit: Banana" \
  --choice "color=red,yellow,blue"
```

### Noul (yes / no)

Ask a yes-or-no question. Decider returns the probability that the answer is
true.

```bash
strands-decider ask StrandsAgents/strands-decider-2B-qwen3.5-v1-2610 \
  --state "Help! My payouts have been failing for 3 days." \
  --noul "urgent=is this request urgent?"
```

### Score

Rate something on an ordered scale. Decider returns the expected score, the
distribution across levels, and confidence.

```bash
strands-decider ask StrandsAgents/strands-decider-2B-qwen3.5-v1-2610 \
  --state "The checkout button is missing on mobile." \
  --score "severity=low,medium,high,critical"
```

## Image support

Strands Decider can also answer questions about images when loaded with
`--vision`. See [Images](images.md) for supported models, request format, and
measurements.

```bash
pip install "strands-decider[vision]"
strands-decider serve StrandsAgents/strands-decider-2B-qwen3.5-v1-2610 --vision --port 8000
```

## Good use cases for System One models

- **Replacing decision making from LLMs with Decider models.** LLMs are great at
  making decisions based on some information, but they can be costly and slow.
  Switching to a System One model reduces token cost and response latency. You
  no longer pay for idle compute while waiting for an LLM — Decider often
  responds in under one second, and because it produces a single output token,
  you do not pay for output tokens.
- **Model routing.** Use a System One model to route an incoming question to
  the correctly scoped model — Haiku for easy problems, Opus for hard ones. See
  [Decision models in the Strands SDK](https://strandsagents.com/docs/user-guide/sdk/decision-models/).
- **Interventions.** Use a System One model to decide whether a human should
  intervene in an agent's action. If an agent might perform an action that
  cannot be deterministically identified as harmful, Decider can flag it for
  human review. _Interventions docs coming soon._
- **Filtering.** Use a System One model inside a graph to decide whether a
  request should be processed by an LLM (expensive) or skipped to save cost.
  _Blog post link coming soon._

## Interested in learning or trying more?

- [Strands Decider on GitHub](https://github.com/strands-agents/strands-decider) — design and inference docs
- [Train a model yourself](https://github.com/strands-labs/strands-decider/tree/main/training)
