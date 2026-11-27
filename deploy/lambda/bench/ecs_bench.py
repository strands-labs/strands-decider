"""The decision image on ECS: each build on each Graviton generation (EC2 launch type, pinned by
instance type), and on Fargate, where AWS picks the host. Each task runs bench_container.py and
writes its results to S3; this waits for them and downloads them, for graviton_bench.py report.

  python ecs_bench.py --cluster sd-graviton-bench --subnet subnet-... --security-group sg-... \\
      --bucket sd-graviton-bench-...-us-east-1 --labels baseline variants kleidiai \\
      --instance-types c6g.2xlarge c7g.2xlarge c8g.2xlarge c9g.2xlarge --fargate 4 --out ecs.jsonl

Task definitions are `<cluster>-ec2-<label>` and `<cluster>-fargate-<label>`, with a container
named `bench` (bench/README.md). Each run's results go under results/<platform>/<run id>/.
"""

from __future__ import annotations

import argparse
import json
import time
import uuid
from concurrent.futures import ThreadPoolExecutor


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cluster", required=True)
    p.add_argument("--subnet", required=True)
    p.add_argument("--security-group", required=True)
    p.add_argument("--bucket", required=True)
    p.add_argument("--labels", nargs="+", required=True)
    p.add_argument("--instance-types", nargs="*", default=[])
    p.add_argument("--fargate", type=int, default=0, help="Fargate tasks per label")
    p.add_argument("--region", default="us-east-1")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    import boto3

    ecs = boto3.client("ecs", region_name=args.region)
    # Each run's tasks write under results/<platform>/<run>/, so overlapping runs don't collect
    # each other's results (bench_container.py reads BENCH_RUN_ID).
    run_id = time.strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:6]
    overrides = {
        "containerOverrides": [
            {"name": "bench", "environment": [{"name": "BENCH_RUN_ID", "value": run_id}]}
        ]
    }
    print(f"run {run_id}", flush=True)
    s3 = boto3.client("s3", region_name=args.region)

    def wait(arns: list[str]) -> list[dict]:
        done: list[dict] = []
        while arns:
            time.sleep(30)
            tasks = ecs.describe_tasks(cluster=args.cluster, tasks=arns)["tasks"]
            for t in tasks:
                if t["lastStatus"] == "STOPPED":
                    c = t["containers"][0]
                    done.append(
                        {
                            "arn": t["taskArn"],
                            "exit": c.get("exitCode"),
                            "reason": t.get("stoppedReason"),
                        }
                    )
                    arns.remove(t["taskArn"])
        return done

    def on_instance(itype: str) -> list[dict]:
        """Each label in turn on this instance type: one task fits on a host at a time."""
        out = []
        for label in args.labels:
            r = ecs.run_task(
                cluster=args.cluster,
                taskDefinition=f"{args.cluster}-ec2-{label}",
                launchType="EC2",
                overrides=overrides,
                placementConstraints=[
                    {"type": "memberOf", "expression": f"attribute:ecs.instance-type == {itype}"}
                ],
            )
            if r["failures"]:
                out.append({"itype": itype, "label": label, "failure": r["failures"]})
                continue
            out += [dict(x, itype=itype, label=label) for x in wait([r["tasks"][0]["taskArn"]])]
            print(itype, label, out[-1], flush=True)
        return out

    def on_fargate(label: str) -> list[dict]:
        arns = []
        for _ in range(args.fargate):
            r = ecs.run_task(
                cluster=args.cluster,
                taskDefinition=f"{args.cluster}-fargate-{label}",
                launchType="FARGATE",
                overrides=overrides,
                networkConfiguration={
                    "awsvpcConfiguration": {
                        "subnets": [args.subnet],
                        "securityGroups": [args.security_group],
                        "assignPublicIp": "ENABLED",  # ECR and S3 without a NAT; nothing listens
                    }
                },
            )
            arns += [t["taskArn"] for t in r["tasks"]]
        res = [dict(x, label=label, itype="fargate") for x in wait(arns)]
        print("fargate", label, res, flush=True)
        return res

    with ThreadPoolExecutor(len(args.instance_types) + len(args.labels)) as pool:
        futures = [pool.submit(on_instance, i) for i in args.instance_types]
        if args.fargate:
            futures += [pool.submit(on_fargate, label) for label in args.labels]
        runs = [x for f in futures for x in f.result()]
    print(json.dumps(runs, indent=1))

    with open(args.out, "a") as out:
        for where in ("ec2", "fargate"):
            prefix = f"results/{where}/{run_id}/"
            for page in s3.get_paginator("list_objects_v2").paginate(
                Bucket=args.bucket, Prefix=prefix
            ):
                for obj in page.get("Contents", []):
                    body = s3.get_object(Bucket=args.bucket, Key=obj["Key"])["Body"].read().decode()
                    for line in body.splitlines():
                        row = json.loads(line)
                        row["where"] = "ecs-ec2" if where == "ec2" else "fargate"
                        row["run"] = run_id
                        out.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
