# Getting started with Strands Decider in Strands SDK

This walkthrough takes you from install to using Strands Decider as a model
router inside a Strands agent.

## 1. Install the CLI

```bash
pip install strands-decider
```

## 2. Ask a simple choice question

```bash
strands-decider ask StrandsAgents/strands-decider-2B-qwen3.5-v1-2610 \
  --state "Help! My payouts have been failing for 3 days." \
  --choice "team=billing,technical,sales"
```

Decider returns the selected option, a probability for every option, and a
confidence score.

## 3. Serve the model on a local HTTP server

```bash
strands-decider serve StrandsAgents/strands-decider-2B-qwen3.5-v1-2610 --port 8000
```

## 4. Curl the server for a response

```bash
curl -s localhost:8000/v1/systemone -H 'content-type: application/json' -d '{
  "state": "Help! My payouts have been failing for 3 days.",
  "questions": {
    "team": {
      "type": "choice",
      "instructions": "Which team should handle this?",
      "criteria": {
        "billing": "billing and payments",
        "technical": "bugs and outages",
        "sales": "pre-sales questions"
      }
    }
  }
}'
```

## 5. Create an agent that uses Decider for model routing

See [Decision models in the Strands SDK](https://strandsagents.com/docs/user-guide/sdk/decision-models/)
for the model-routing agent walkthrough.
