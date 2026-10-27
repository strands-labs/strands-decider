"""Custom resource for NeuronCompile (../neuron-compile.ts): compile a model on a temporary inf2 host.

on_event starts the host and returns; is_complete reports done once the host has written its
marker to S3. The host (host-user-data.sh, beside this file) copies the model data from SOURCE, runs
`strands-decider compile-neuron` in the serving image, uploads the served layout to TARGET, writes
TARGET_SUCCESS or TARGET_FAILED, and terminates itself.

A target that already holds _SUCCESS is not compiled again: it was compiled by the same image for
the same model data, both of which are in its path.
"""

from __future__ import annotations

import base64
from pathlib import Path

import boto3
from botocore.exceptions import ClientError

ec2 = boto3.client("ec2")
s3 = boto3.client("s3")

# run_instances errors after which the next subnet (another Availability Zone) may still work
CAPACITY_ERRORS = {"InsufficientInstanceCapacity", "Unsupported", "InstanceLimitExceeded"}
TAG = "strands-decider:neuron-compile"
USER_DATA = (Path(__file__).parent / "host-user-data.sh").read_text()


def _exists(bucket: str, key: str) -> bool:
    try:
        s3.head_object(Bucket=bucket, Key=key)
        return True
    except ClientError as e:
        if e.response["Error"]["Code"] in ("404", "NoSuchKey", "NotFound"):
            return False
        raise


def _instances(target: str, states: list[str]) -> list[str]:
    pages = ec2.get_paginator("describe_instances").paginate(
        Filters=[
            {"Name": f"tag:{TAG}", "Values": [target]},
            {"Name": "instance-state-name", "Values": states},
        ]
    )
    return [i["InstanceId"] for p in pages for r in p["Reservations"] for i in r["Instances"]]


def _launch(props: dict) -> str:
    target = f"s3://{props['Bucket']}/{props['Target']}"
    user_data = (
        USER_DATA.replace("__REGION__", props["Region"])
        .replace("__LOG_GROUP__", props["LogGroup"])
        .replace("__SOURCE__", f"s3://{props['Bucket']}/{props['Source']}")
        .replace("__TARGET__", target)
        .replace("__IMAGE__", props["Image"])
        .replace("__CHECKPOINT__", props["Checkpoint"])
    )
    errors = []
    for subnet in props["Subnets"]:
        try:
            r = ec2.run_instances(
                LaunchTemplate={"LaunchTemplateId": props["LaunchTemplate"], "Version": "$Latest"},
                SubnetId=subnet,
                MinCount=1,
                MaxCount=1,
                UserData=base64.b64encode(user_data.encode()).decode(),
                TagSpecifications=[
                    {
                        "ResourceType": "instance",
                        "Tags": [
                            {"Key": TAG, "Value": props["Target"]},
                            {"Key": "Name", "Value": f"neuron-compile {props['Target']}"},
                        ],
                    }
                ],
            )
            return r["Instances"][0]["InstanceId"]
        except ClientError as e:
            if e.response["Error"]["Code"] not in CAPACITY_ERRORS:
                raise
            errors.append(f"{subnet}: {e.response['Error']['Code']}")
    raise RuntimeError(f"no {props['InstanceType']} capacity in any subnet: {errors}")


def on_event(event: dict, context: object) -> dict:
    props = event["ResourceProperties"]
    target = props["Target"]
    if event["RequestType"] == "Delete":
        # Stop a host that is still compiling. The compiled data goes with the bucket.
        running = _instances(event["PhysicalResourceId"], ["pending", "running"])
        if running:
            ec2.terminate_instances(InstanceIds=running)
        return {"PhysicalResourceId": event["PhysicalResourceId"]}
    if not _exists(props["Bucket"], target + "_SUCCESS"):
        if _exists(props["Bucket"], target + "_FAILED"):
            s3.delete_object(Bucket=props["Bucket"], Key=target + "_FAILED")  # a retry
        if not _instances(target, ["pending", "running"]):
            print(f"started {_launch(props)} for s3://{props['Bucket']}/{target}")
    return {"PhysicalResourceId": target}


def is_complete(event: dict, context: object) -> dict:
    if event["RequestType"] == "Delete":
        return {"IsComplete": True}
    props = event["ResourceProperties"]
    bucket, target = props["Bucket"], props["Target"]
    if _exists(bucket, target + "_SUCCESS"):
        return {"IsComplete": True}
    if _exists(bucket, target + "_FAILED"):
        raise RuntimeError(
            f"compiling failed: see s3://{bucket}/{target}compile.log and the log group "
            f"{props['LogGroup']}"
        )
    if not _instances(target, ["pending", "running", "shutting-down", "stopping"]):
        raise RuntimeError(f"the compile host for s3://{bucket}/{target} stopped without a result")
    return {"IsComplete": False}
