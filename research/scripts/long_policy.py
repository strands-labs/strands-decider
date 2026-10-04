#!/usr/bin/env python3
"""Print the task-level `long_policy` report from the committed JevBench CSVs.

This reads the saved results only; it does not load a checkpoint or run JevBench.

    python research/scripts/long_policy.py
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "research" / "data"
LINEAGE = ("v16", "v17", "v18", "v19")
REFERENCES = ("decider-2b v10", "decider-2b v11")
RUNS = (*LINEAGE, *REFERENCES)


def read_csv(name: str) -> list[dict[str, str]]:
    with open(DATA / name, encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def print_table(headers: tuple[str, ...], rows: list[tuple[object, ...]]) -> None:
    print("| " + " | ".join(headers) + " |")
    print("| " + " | ".join("---" for _ in headers) + " |")
    for row in rows:
        print("| " + " | ".join(str(value) for value in row) + " |")


def main() -> None:
    tasks = {row["task_id"]: row["family"] for row in read_csv("jevbench_tasks.csv")}
    results = {
        (row["run"], row["task_id"]): int(row["correct"])
        for row in read_csv("jevbench_results.csv")
        if row["run"] in RUNS
    }
    long_policy = sorted(task for task, family in tasks.items() if family == "long_policy")
    assert len(long_policy) == 19
    assert all((run, task) in results for run in RUNS for task in long_policy)

    print("## long_policy tasks\n")
    print_table(
        ("task_id", *RUNS),
        [(task, *(results[run, task] for run in RUNS)) for task in long_policy],
    )

    print("\n## v19 against decider-2b\n")
    by_family: dict[str, list[str]] = defaultdict(list)
    for task, family in tasks.items():
        by_family[family].append(task)
    for reference in REFERENCES:
        rows = []
        for family, family_tasks in sorted(by_family.items()):
            v19 = sum(results["v19", task] for task in family_tasks)
            peer = sum(results[reference, task] for task in family_tasks)
            only_v19 = sum(
                results["v19", task] and not results[reference, task] for task in family_tasks
            )
            only_peer = sum(
                results[reference, task] and not results["v19", task] for task in family_tasks
            )
            if only_v19 or only_peer:
                rows.append(
                    (
                        family,
                        len(family_tasks),
                        v19,
                        peer,
                        only_v19,
                        only_peer,
                        only_v19 - only_peer,
                    )
                )
        print(f"### v19 vs {reference}\n")
        print_table(("family", "n", "v19", "peer", "only v19", "only peer", "net"), rows)
        print()


if __name__ == "__main__":
    main()
