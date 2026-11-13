import random

from training import batching


def lengths_for(case):
    rnd = random.Random(case["lengths_seed"])
    return [
        rnd.randint(30, 16384) if rnd.random() < 0.1 else rnd.randint(30, 3000) for _ in range(5000)
    ]


def test_row_length_matches_previous(previous):
    case = previous["row_length"]
    assert [batching.row_length(r) for r in case["rows"]] == case["out"]


def test_pack_microbatches_match_previous(previous):
    for case in previous["pack_microbatches"]:
        lengths = case["lengths"]
        got = batching.pack_microbatches(
            list(range(len(lengths))), case["batch"], case["budget"], length=lengths.__getitem__
        )
        assert got == case["out"]


def test_microbatch_limits():
    lengths = [100, 5000, 200, 300, 9000, 50, 60]
    for mb in batching.pack_microbatches(
        list(range(len(lengths))), 3, 8192, length=lambda i: lengths[i]
    ):
        assert len(mb) <= 3
        assert len(mb) == 1 or max(lengths[i] for i in mb) * len(mb) <= 8192


def test_plans_match_previous(previous):
    # The plan does not depend on the previous trainer's GPU-placement balancing, which this trainer does not have, so
    # every case's plan is checked; the placement only where the case did not balance. The fixture's statistics also
    # hold the balancing's keys, which this trainer does not report.
    dropped = {"rank_load_max_over_mean", "balance", "attn_tokens"}
    for case in previous["plans"]:
        lengths = lengths_for(case)
        plan = batching.build_plan(
            lengths,
            case["epochs"],
            case["seed"],
            case["world"],
            case["accum"],
            case["batch"],
            case["budget"],
            case["window"],
        )
        placement = batching.Placement(case["world"], case["accum"])
        stats = batching.plan_statistics(plan, lengths, placement)
        assert stats == {k: v for k, v in case["statistics"].items() if k not in dropped}
        if not case["balance"]:
            steps = plan[:5] + plan[-3:]
            work = [[list(placement.local_work(s, r)) for r in range(case["world"])] for s in steps]
            assert _sha(work) == case["work_sha"]


def _sha(value):
    import hashlib
    import json

    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def test_plan_depends_on_the_step_not_the_world():
    """The same rows per step give the same plan on 8 GPUs x 3 microbatches or 24 x 1 (the window is the step)."""
    rnd = random.Random(5)
    lengths = [rnd.randint(30, 6000) for _ in range(3000)]
    a = batching.build_plan(lengths, 1.0, 9, 8, 3, 16, 16384, 8 * 3 * 16)
    b = batching.build_plan(lengths, 1.0, 9, 24, 1, 16, 16384, 24 * 1 * 16)
    assert batching.plan_sha256(a) == batching.plan_sha256(b)


def test_placement_runs_every_microbatch_once():
    rnd = random.Random(6)
    lengths = [rnd.randint(30, 16000) for _ in range(2000)]
    plan = batching.build_plan(lengths, 1.0, 3, 8, 4, 8, 8192, 8 * 4 * 8)
    placement = batching.Placement(8, 4)
    for step in plan:
        js = sorted(j for r in range(8) for _mb, real, j in placement.local_work(step, r) if real)
        assert js == list(range(len(step[1])))
        assert all(len(placement.local_work(step, r)) == 4 for r in range(8))


def test_fractional_epoch_takes_its_first_steps():
    lengths = [100] * 640
    full = batching.build_plan(lengths, 2.0, 1, 2, 1, 8, 8192, 16)
    half = batching.build_plan(lengths, 1.5, 1, 2, 1, 8, 8192, 16)
    first = [s for s in full if s[0] == 0]
    assert half[: len(first)] == first
    assert len(half) == len(first) + len(first) // 2
