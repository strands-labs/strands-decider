import pytest

from strands_decider.naming import parse, release_name


@pytest.mark.parametrize("base, size", [
    ("Qwen/Qwen3.5-2B-Base", "2B"),
    ("google/gemma-4-E2B-it", "E2B"),
    ("google/gemma-4-E4B-it", "E4B"),
    ("google/gemma-4-12B-it", "12B"),
    ("google/gemma-4-26B-A4B-it", "26B-A4B"),
])
def test_size_token_is_the_hub_ids(base, size):
    name = release_name(base, 1, "2610")
    assert name.startswith(f"strands-decider-{size}-")
    assert parse(name)[0] == size


def test_this_release():
    assert release_name("Qwen/Qwen3.5-2B-Base", 1, "2610") == "strands-decider-2B-qwen3.5-v1-2610"
    assert parse("StrandsAgents/strands-decider-2B-qwen3.5-v1-2610") == ("2B", "qwen3.5", 1, "2610")
    assert parse("strands-decider-E4B-gemma4-v2-2611") == ("E4B", "gemma4", 2, "2611")


@pytest.mark.parametrize("bad", [
    "strands-decider-2B-hobson-v21",  # released before the convention; keeps its name
    "strands-decider-2b-qwen3.5-v1-2610",  # size as the Hub id writes it
    "strands-decider-2B-qwen3.5-v0-2610",
    "strands-decider-2B-qwen3.5-v1-2613",
    "strands-decider-2B-qwen3.5-v1-261015",  # a same-month update is a Hub revision tag
])
def test_parse_refuses(bad):
    with pytest.raises(ValueError):
        parse(bad)


@pytest.mark.parametrize("args", [("meta-llama/Llama-3-8B", 1, "2610"),
                                  ("Qwen/Qwen3.5-2B-Base", 0, "2610"),
                                  ("Qwen/Qwen3.5-2B-Base", 1, "2600")])
def test_release_name_refuses(args):
    with pytest.raises(ValueError):
        release_name(*args)
