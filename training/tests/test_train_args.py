import pytest

from training import train

ROOT = train.HERE


def args(*extra):
    return train.resolve(["--train", "t.jsonl", "--dev", "d.jsonl", *extra])


def test_defaults_are_the_recipe():
    a = args()
    assert (
        a.rank,
        a.alpha,
        a.dropout,
        a.epochs,
        a.lr,
        a.warmup_ratio,
        a.max_length,
        a.save_every,
        a.rows_per_step,
    ) == (64, 128.0, 0.05, 1.0, 1e-4, 0.05, 16384, 100, 3072)
    assert a.prompt == "strands-chat" and a.seed == 20260920 and a.head == "token_readout"
    assert a.lr_floor == 0.1


@pytest.mark.parametrize(
    "size,lr,batch,budget,split,experts",
    [
        ("e2b", 1e-4, 32, 32768, 1024, False),
        ("e4b", 1e-4, 32, 32768, 1024, False),
        ("12b", 3e-5, 16, 16384, 0, False),
        ("26b-a4b", 1e-4, 8, 8192, 0, True),
        ("31b", 1e-4, 8, 8192, 0, False),
    ],
)
def test_size_configs(size, lr, batch, budget, split, experts):
    a = args("--config", str(ROOT / "configs" / f"{size}.yaml"))
    assert (a.lr, a.batch_size, a.token_budget, a.attn_split, a.expert_lora) == (
        lr,
        batch,
        budget,
        split,
        experts,
    )
    assert a.base_model.startswith("google/gemma-4-") and len(a.revision) == 40
    assert a.tokenizer_sha256 == "cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f"
    per_layer = {"per_layer_input_gate", "per_layer_projection"} <= set(a.targets.split(","))
    assert per_layer == (size in ("e2b", "e4b"))


def test_flags_override_the_config():
    a = args(
        "--config",
        str(ROOT / "configs" / "e2b.yaml"),
        "--lr",
        "2e-4",
        "--batch-size",
        "16",
        "--seed",
        "5",
    )
    assert (a.lr, a.batch_size, a.seed) == (2e-4, 16, 5)


@pytest.mark.parametrize("world,accum", [(8, 12), (96, 1), (32, 3), (16, 6)])
def test_accum_from_the_world_size(world, accum):
    assert train.resolve_accum(args("--batch-size", "32"), world) == accum


def test_accum_refuses_a_world_that_does_not_divide():
    with pytest.raises(SystemExit):
        train.resolve_accum(args("--batch-size", "32"), 64)


def test_unknown_config_keys_are_refused(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("lr: 1e-4\nlearning_rate: 2\n")
    with pytest.raises(SystemExit, match="unknown"):
        args("--config", str(bad))


def test_config_booleans_can_be_turned_off():
    a = args("--config", str(ROOT / "configs" / "e2b.yaml"), "--merge", "--augment")
    assert a.no_merge is False and a.no_augment is False
    b = args("--config", str(ROOT / "configs" / "26b-a4b.yaml"), "--no-expert-lora")
    assert b.expert_lora is False and b.expert_rank == 0


def test_config_values_are_typed(tmp_path):
    good = tmp_path / "good.yaml"
    good.write_text("lr: 1e-4\nexpert_lora: true\n")  # YAML 1.1 reads 1e-4 as a string
    assert args("--config", str(good)).lr == 1e-4
    bad = tmp_path / "bad.yaml"
    bad.write_text("expert_lora: yes please\n")
    with pytest.raises(SystemExit, match="true or false"):
        args("--config", str(bad))


def test_negative_accum_is_refused():
    with pytest.raises(SystemExit):
        args("--accum", "-1")


def test_the_removed_cluster_options_are_unknown(tmp_path):
    for key in ("balance", "shard_base", "release_runs"):
        bad = tmp_path / f"{key}.yaml"
        bad.write_text(f"{key}: true\n")
        with pytest.raises(SystemExit, match="unknown"):
            args("--config", str(bad))
    with pytest.raises(SystemExit):
        args("--balance")


@pytest.mark.parametrize(
    "name,base,bos",
    [("pointer-2b", "Qwen/Qwen3.5-2B-Base", False), ("pointer-e2b", "google/gemma-4-E2B-it", True)],
)
def test_pointer_configs(name, base, bos):
    a = args("--config", str(ROOT / "configs" / f"{name}.yaml"))
    assert (a.head, a.prompt, a.base_model, a.force_bos) == ("pointer", "numbered", base, bos)
    assert (a.rank, a.alpha, a.lr, a.head_lr, a.weight_decay, a.warmup_ratio, a.lr_floor) == (
        16,
        32.0,
        1e-4,
        1e-3,
        0.01,
        0.03,
        0.0,
    )
    assert (a.rows_per_step, a.batch_size, a.max_length, a.pointer_dim, a.head_dropout) == (
        32,
        4,
        4096,
        256,
        0.05,
    )
    assert (
        len(a.revision) == 40
        and len(a.tokenizer_sha256) == 64
        and len(a.expected_base_sha256) == 64
    )
    for world in (1, 2, 4, 8):  # the same 8 microbatches a step
        assert train.resolve_accum(a, world) * world == 8


@pytest.mark.parametrize(
    "extra",
    [
        ("--head", "pointer"),
        ("--prompt", "numbered"),
        ("--head", "pointer", "--prompt", "numbered", "--expert-lora"),
        ("--head", "pointer", "--prompt", "numbered", "--attn-split", "1024"),
    ],
)
def test_a_head_takes_only_its_own_prompt_and_options(extra):
    with pytest.raises(SystemExit):
        args(*extra)


def test_lr_floor_is_checked():
    assert args("--lr-floor", "0").lr_floor == 0.0
    with pytest.raises(SystemExit):
        args("--lr-floor", "1.5")
