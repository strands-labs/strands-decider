"""The installed strands_decider core renders the training rows exactly as the committed fixture records
(fixtures/render.json): a core that renders differently would change what the recipe trains on."""

import hashlib
import json

import pytest
from trainer_support import FIXTURES

from strands_decider import core
from training.prompt import bound_options, core_question

FIXTURE = json.loads((FIXTURES / "render.json").read_text())


@pytest.mark.parametrize("template", ["strands-chat", "strands-plain"])
def test_renders_match_the_fixture(pinned_gemma4_tokenizer, template):
    tokenizer = pinned_gemma4_tokenizer
    mismatched = []
    for case in FIXTURE["cases"]:
        row, expected = case["row"], case[template]
        try:
            q = core.render(
                {"state": row["state"], "questions": {"q": core_question(row["question"])}},
                tokenizer,
                template,
            ).questions["q"]
            got = {
                "ids_sha256": hashlib.sha256(json.dumps(list(q.ids)).encode()).hexdigest(),
                "n": len(q.ids),
                "S": q.state_len,
                "codes": len(q.code_token_ids),
                "options_match": list(q.options) == bound_options(row["question"]),
            }
        except Exception as e:
            got = {"refused": type(e).__name__}
        want = {k: v for k, v in expected.items() if k != "same_as_trained_core"}
        if got != want:
            mismatched.append(row["id"])
    assert not mismatched, f"{len(mismatched)} rows render differently: {mismatched[:5]}"


def test_the_fixture_matches_the_trained_core_but_for_nonfinite_and_surrogate_states():
    """The fixture records, per row, whether the core the trained models used rendered it as this core does (recorded
    when the fixture was made; this test renders nothing): all but the states holding NaN, Infinity or a lone
    surrogate, which the newer core refuses."""
    for case in FIXTURE["cases"]:
        if not case["strands-chat"]["same_as_trained_core"]:
            assert case["row"]["id"].split("/")[1] in ("14", "15", "16"), case["row"]["id"]
