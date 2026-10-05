"""Every shipped config loads, and the v19 route's configs cannot drift from their records.

No test read a config before this one, so a renamed or removed TrainConfig field failed
only on the GPU host. configs/train.yaml is what the released v19 weights trained from;
its header claims it equals configs/experiments/v19.yaml except three keys, and
configs/train-parent.yaml claims the same of v14.yaml. These tests hold both claims,
comparing the parsed configs, not the text.

tests/fixtures/v19-p5-run1/ holds the hobson_config.json and train_config.json that the
v19 checkpoint saved: hyperparameters and relative paths only. A saved train_config.json
is a reproduction path (`strands-decider train --config <ckpt>/train_config.json`).
"""

import dataclasses
import glob
import os

import pytest

from strands_decider.modeling import StrandsDeciderConfig
from strands_decider.train import TrainConfig

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIGS = sorted(glob.glob(os.path.join(ROOT, "configs", "**", "*.yaml"), recursive=True))
FIXTURE = os.path.join(ROOT, "tests", "fixtures", "v19-p5-run1")


def _load(*parts):
    return dataclasses.asdict(TrainConfig.from_yaml(os.path.join(ROOT, *parts)))


def _differing_keys(a, b):
    return {k for k in a if a[k] != b[k]}


@pytest.mark.parametrize("path", CONFIGS, ids=[os.path.relpath(p, ROOT) for p in CONFIGS])
def test_every_config_loads(path):
    # from_yaml rejects unknown keys, so this also fails when a field is removed or renamed.
    TrainConfig.from_yaml(path)


def test_train_yaml_is_v19_except_three_keys():
    train, v19 = _load("configs", "train.yaml"), _load("configs", "experiments", "v19.yaml")
    assert _differing_keys(train, v19) == {"teacher_file", "output_dir", "max_length"}


def test_train_parent_yaml_is_v14_except_two_keys():
    parent, v14 = _load("configs", "train-parent.yaml"), _load("configs", "experiments", "v14.yaml")
    assert _differing_keys(parent, v14) == {"teacher_file", "output_dir"}
    # The file that `training/recipe.sh teacher` writes and run_recipe.sh check_teacher checksums.
    assert parent["teacher_file"] == "data/teacher_multistep_v14_train.jsonl"


def test_v19_seed1_is_v19_except_seed_and_output_dir():
    seed1, v19 = _load("configs", "experiments", "v19-seed1.yaml"), _load("configs", "experiments", "v19.yaml")
    assert _differing_keys(seed1, v19) == {"seed", "output_dir"}


def test_v19_saved_configs_load():
    StrandsDeciderConfig.from_json(os.path.join(FIXTURE, "hobson_config.json"))
    saved = dataclasses.asdict(TrainConfig.from_yaml(os.path.join(FIXTURE, "train_config.json")))
    # The released run was configs/train.yaml under training/run_recipe.sh FAST=1, which
    # turns gradient checkpointing off and precomputes the frozen-KL reference (speed only).
    assert _differing_keys(saved, _load("configs", "train.yaml")) == {"gradient_checkpointing", "precompute_frozen_kl"}


def _config(path: str) -> dict:
    """Any shipped config, parsed by the class its directory belongs to."""
    return dataclasses.asdict(TrainConfig.from_yaml(path))


SEED_REPLICATES = sorted(glob.glob(os.path.join(ROOT, "configs", "**", "*-seed[0-9].yaml"), recursive=True))


@pytest.mark.parametrize("path", SEED_REPLICATES, ids=[os.path.relpath(p, ROOT) for p in SEED_REPLICATES])
def test_seed_replicates_differ_only_in_seed_and_output_dir(path):
    seed = int(path[-len("0.yaml")])
    rep, ref = _config(path), _config(path[: -len("-seed0.yaml")] + ".yaml")
    assert _differing_keys(rep, ref) == {"seed", "output_dir"}
    assert rep["seed"] == seed and str(seed) in rep["output_dir"]


def test_v20_is_v19_yn27b_for_a_full_epoch_from_one_initialisation():
    v20, yn = _load("configs", "experiments", "strands-decider-2B-hobson-v20.yaml"), _load(
        "configs", "experiments", "v19-yn27b.yaml")
    assert _differing_keys(v20, yn) == {"max_steps", "init_seed", "output_dir"}
    assert (v20["continue_from"], v20["max_steps"], v20["init_seed"]) == ("checkpoints/v19", 3738, 0)
