from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
import sys
import tarfile
import tempfile
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import boto3
from botocore.exceptions import ClientError, WaiterError

SSM_POLICY_ARN = "arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore"
TERMINAL_COMMAND_STATES = {"Success", "Cancelled", "TimedOut", "Failed", "Cancelling"}
ANSI_ESCAPE = re.compile(r"\x1b(?:\[[0-?]*[ -/]*[@-~]|\][^\x07]*(?:\x07|\x1b\\))")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a SageMaker benchmark from a temporary, same-region EC2 client. "
            "The client has no inbound access and is managed through SSM."
        )
    )
    parser.add_argument("environment", help="benchmark environment name or environment.yaml path")
    parser.add_argument("model", nargs="?", help="one configured model; omit to run all models")
    parser.add_argument(
        "--account",
        required=True,
        help="expected AWS account ID; the script refuses to use a different account",
    )
    parser.add_argument(
        "--region",
        help="expected target region; defaults to and must match sagemaker.region",
    )
    parser.add_argument(
        "--instance-type",
        default="t3.medium",
        help="temporary load-generator instance type (default: t3.medium)",
    )
    parser.add_argument(
        "--ref",
        default="HEAD",
        help="committed Git revision copied to EC2 (default: HEAD)",
    )
    parser.add_argument("--smoke", action="store_true", help="run the benchmark smoke preset")
    parser.add_argument(
        "--command-timeout-seconds",
        type=int,
        default=7200,
        help="SSM benchmark timeout (default: 7200)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate and print the resource plan without creating AWS resources",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="confirm creation of temporary paid AWS resources",
    )
    return parser.parse_args()


def repo_root() -> Path:
    return Path(
        subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    ).resolve()


def load_environment(root: Path, value: str) -> tuple[Path, dict[str, Any]]:
    candidate = Path(value).expanduser()
    if candidate.is_file():
        path = candidate.resolve()
    elif candidate.is_dir():
        path = (candidate / "environment.yaml").resolve()
    else:
        path = (root / "benchmark" / "environments" / value / "environment.yaml").resolve()
    if not path.is_file():
        raise ValueError(f"environment configuration not found: {path}")

    # Keep this helper runnable with the repository's benchmark extra.
    import yaml

    raw = yaml.safe_load(path.read_text()) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return path, raw


def validate_environment(
    root: Path,
    path: Path,
    raw: dict[str, Any],
    requested_model: str | None,
    requested_region: str | None,
) -> tuple[str, list[str], list[str], str]:
    try:
        relative_environment = path.relative_to(root)
    except ValueError as error:
        raise ValueError(
            "the environment file must be inside the current Git repository"
        ) from error

    sagemaker = raw.get("sagemaker")
    if not isinstance(sagemaker, dict):
        raise ValueError(f"{path} is not a SageMaker environment")
    if sagemaker.get("profile"):
        raise ValueError(
            f"{path}: remove sagemaker.profile so the EC2 client uses its instance role"
        )

    region = sagemaker.get("region")
    if not isinstance(region, str) or not region:
        raise ValueError(f"{path}: sagemaker.region is required")
    if requested_region and requested_region != region:
        raise ValueError(f"--region {requested_region} does not match sagemaker.region {region}")

    models = raw.get("models")
    if (
        not isinstance(models, list)
        or not models
        or not all(isinstance(model, str) and model for model in models)
    ):
        raise ValueError(f"{path}: models must be a non-empty list")
    selected_models = [requested_model] if requested_model else models
    if requested_model and requested_model not in models:
        raise ValueError(f"{requested_model} is not configured in {path}")
    endpoint_name = sagemaker.get("endpoint_name")
    endpoints = sagemaker.get("endpoints")
    if endpoint_name:
        if len(models) != 1:
            raise ValueError("sagemaker.endpoint_name is valid only for a single-model environment")
        selected_endpoints = [str(endpoint_name)]
    elif isinstance(endpoints, dict):
        missing = [model for model in selected_models if not endpoints.get(model)]
        if missing:
            raise ValueError(f"missing SageMaker endpoints for: {', '.join(missing)}")
        selected_endpoints = sorted({str(endpoints[model]) for model in selected_models})
    else:
        raise ValueError("configure sagemaker.endpoint_name or sagemaker.endpoints")

    return str(relative_environment), selected_models, selected_endpoints, region


def wait_for_ssm(ssm: Any, instance_id: str, timeout_seconds: int = 600) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        rows = ssm.describe_instance_information(
            Filters=[{"Key": "InstanceIds", "Values": [instance_id]}]
        )["InstanceInformationList"]
        if rows and rows[0]["PingStatus"] == "Online":
            return
        time.sleep(10)
    raise TimeoutError(f"{instance_id} did not become available in SSM")


def read_remote_progress(ssm: Any, instance_id: str) -> str | None:
    try:
        command_id = ssm.send_command(
            InstanceIds=[instance_id],
            DocumentName="AWS-RunShellScript",
            TimeoutSeconds=30,
            Parameters={"commands": ["tail -c 12000 /tmp/benchmark.log 2>/dev/null || true"]},
            Comment="Read strands-decider benchmark progress",
        )["Command"]["CommandId"]
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            try:
                result = ssm.get_command_invocation(
                    CommandId=command_id,
                    InstanceId=instance_id,
                )
            except ssm.exceptions.InvocationDoesNotExist:
                time.sleep(1)
                continue
            if result["Status"] in TERMINAL_COMMAND_STATES:
                text = ANSI_ESCAPE.sub("", result.get("StandardOutputContent", ""))
                lines = [
                    line.strip() for line in text.replace("\r", "\n").splitlines() if line.strip()
                ]
                progress_lines = [line for line in lines if re.search(r"\b\d+/\d+\b", line)]
                if progress_lines:
                    return progress_lines[-1]
                messages = [
                    line
                    for line in lines
                    if line.startswith(("calibrating ", "warmup:", "latency ", "concurrency "))
                ]
                return messages[-1] if messages else None
            time.sleep(1)
    except ClientError:
        return None
    return None


def wait_for_command(ssm: Any, command_id: str, instance_id: str) -> dict[str, Any]:
    last_status: str | None = None
    last_progress: str | None = None
    while True:
        try:
            result = ssm.get_command_invocation(
                CommandId=command_id,
                InstanceId=instance_id,
            )
        except ssm.exceptions.InvocationDoesNotExist:
            time.sleep(5)
            continue
        if result["Status"] != last_status:
            print(f"remote benchmark: {result['Status']}", flush=True)
            last_status = result["Status"]
        if result["Status"] in TERMINAL_COMMAND_STATES:
            return result
        progress = read_remote_progress(ssm, instance_id)
        if progress and progress != last_progress:
            print(progress, flush=True)
            last_progress = progress
        time.sleep(10)


def make_source_archive(root: Path, ref: str, destination: Path) -> str:
    revision = subprocess.run(
        ["git", "rev-parse", "--verify", f"{ref}^{{commit}}"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    subprocess.run(
        ["git", "archive", "--format=tar.gz", f"--output={destination}", revision],
        cwd=root,
        check=True,
    )
    return revision


def create_bucket(s3: Any, bucket_name: str, region: str) -> None:
    kwargs: dict[str, Any] = {
        "Bucket": bucket_name,
        "ObjectOwnership": "BucketOwnerEnforced",
    }
    if region != "us-east-1":
        kwargs["CreateBucketConfiguration"] = {"LocationConstraint": region}
    s3.create_bucket(**kwargs)


def secure_bucket(s3: Any, bucket_name: str) -> None:
    s3.put_public_access_block(
        Bucket=bucket_name,
        PublicAccessBlockConfiguration={
            "BlockPublicAcls": True,
            "IgnorePublicAcls": True,
            "BlockPublicPolicy": True,
            "RestrictPublicBuckets": True,
        },
    )
    s3.put_bucket_encryption(
        Bucket=bucket_name,
        ServerSideEncryptionConfiguration={
            "Rules": [
                {
                    "ApplyServerSideEncryptionByDefault": {"SSEAlgorithm": "AES256"},
                    "BucketKeyEnabled": True,
                }
            ]
        },
    )


def build_remote_script(
    *,
    bucket_name: str,
    object_prefix: str,
    environment_path: str,
    model: str | None,
    region: str,
    smoke: bool,
) -> str:
    benchmark_arguments = [environment_path]
    if model:
        benchmark_arguments.append(model)
    if smoke:
        benchmark_arguments.append("--smoke")
    command = " ".join(shlex.quote(argument) for argument in benchmark_arguments)
    environment_dir = str(Path(environment_path).parent)
    benchmark_command = (
        "env PYTHONPATH=src /usr/local/bin/uv run --project benchmark --locked --python 3.12 "
        f"python -m benchmark.runner run {command}"
    )

    return f"""set -e
dnf install -y gzip tar util-linux
command -v aws
curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/usr/local/bin sh
rm -rf /opt/strands-decider
mkdir -p /opt/strands-decider
aws s3 cp \
  {shlex.quote(f"s3://{bucket_name}/{object_prefix}/source.tgz")} \
  /tmp/strands-decider-source.tgz \
  --region {shlex.quote(region)}
tar -xzf /tmp/strands-decider-source.tgz -C /opt/strands-decider
cd /opt/strands-decider
set +e
script --quiet --return --command {shlex.quote(benchmark_command)} /tmp/benchmark.log
benchmark_status=$?
set -e
if [ -d {shlex.quote(environment_dir + "/runs")} ]; then
  tar -czf /tmp/benchmark-results.tgz {shlex.quote(environment_dir + "/runs")}
  aws s3 cp \
    /tmp/benchmark-results.tgz \
    {shlex.quote(f"s3://{bucket_name}/{object_prefix}/benchmark-results.tgz")} \
    --region {shlex.quote(region)}
fi
aws s3 cp \
  /tmp/benchmark.log \
  {shlex.quote(f"s3://{bucket_name}/{object_prefix}/benchmark.log")} \
  --region {shlex.quote(region)}
tail -n 100 /tmp/benchmark.log
exit $benchmark_status
"""


def empty_and_delete_bucket(s3: Any, bucket_name: str) -> None:
    continuation_token: str | None = None
    while True:
        kwargs: dict[str, Any] = {"Bucket": bucket_name}
        if continuation_token:
            kwargs["ContinuationToken"] = continuation_token
        response = s3.list_objects_v2(**kwargs)
        objects = response.get("Contents", [])
        if objects:
            s3.delete_objects(
                Bucket=bucket_name,
                Delete={"Objects": [{"Key": item["Key"]} for item in objects]},
            )
        if not response.get("IsTruncated"):
            break
        continuation_token = response["NextContinuationToken"]
    s3.delete_bucket(Bucket=bucket_name)


def extract_results(archive_path: Path, destination: Path) -> None:
    destination = destination.resolve()
    with tarfile.open(archive_path, "r:gz") as archive:
        members = archive.getmembers()
        for member in members:
            target = (destination / member.name).resolve()
            if destination != target and destination not in target.parents:
                raise ValueError(f"unsafe path in result archive: {member.name}")
            if not (member.isfile() or member.isdir()):
                raise ValueError(f"unsupported entry in result archive: {member.name}")
        archive.extractall(destination, members=members)


def main() -> int:
    args = parse_args()
    if not args.dry_run and not args.yes:
        raise ValueError("pass --yes to confirm creation of temporary paid AWS resources")
    if args.smoke and args.model is None:
        raise ValueError("smoke runs require a model")
    if args.command_timeout_seconds < 60:
        raise ValueError("--command-timeout-seconds must be at least 60")

    root = repo_root()
    environment_file, environment = load_environment(root, args.environment)
    environment_path, selected_models, endpoints, region = validate_environment(
        root,
        environment_file,
        environment,
        args.model,
        args.region,
    )

    plan = {
        "account": args.account,
        "region": region,
        "instance_type": args.instance_type,
        "environment": environment_path,
        "models": selected_models,
        "endpoints": endpoints,
        "git_ref": args.ref,
        "smoke": args.smoke,
        "network": "default VPC, public egress, no inbound rules",
        "cleanup": "terminate instance and delete temporary IAM, S3, and security group",
    }
    print(json.dumps(plan, indent=2), flush=True)
    if args.dry_run:
        return 0

    session = boto3.Session(region_name=region)
    identity = session.client("sts").get_caller_identity()
    if identity["Account"] != args.account:
        raise RuntimeError(
            f"refusing to create resources in {identity['Account']}; expected {args.account}"
        )

    ec2 = session.client("ec2")
    iam = session.client("iam")
    s3 = session.client("s3")
    ssm = session.client("ssm")
    suffix = uuid.uuid4().hex[:10]
    role_name = f"strands-decider-benchmark-{suffix}"
    profile_name = role_name
    policy_name = "BenchmarkInvokeEndpoints"
    group_name = role_name
    bucket_name = f"strands-decider-benchmark-{args.account}-{suffix}"
    object_prefix = f"runs/{suffix}"
    delete_after = (datetime.now(timezone.utc) + timedelta(hours=8)).isoformat(timespec="seconds")
    instance_id: str | None = None
    group_id: str | None = None
    role_created = False
    profile_created = False
    bucket_created = False

    with tempfile.TemporaryDirectory(prefix="strands-decider-ec2-") as temporary_directory:
        temporary = Path(temporary_directory)
        source_archive = temporary / "source.tgz"
        result_archive = temporary / "benchmark-results.tgz"
        log_path = temporary / "benchmark.log"
        revision = make_source_archive(root, args.ref, source_archive)
        print(f"copying committed revision {revision}", flush=True)

        try:
            create_bucket(s3, bucket_name, region)
            bucket_created = True
            secure_bucket(s3, bucket_name)
            s3.upload_file(
                str(source_archive),
                bucket_name,
                f"{object_prefix}/source.tgz",
                ExtraArgs={"ServerSideEncryption": "AES256"},
            )
            print(f"created private transfer bucket {bucket_name}", flush=True)

            trust = {
                "Version": "2012-10-17",
                "Statement": [
                    {
                        "Effect": "Allow",
                        "Principal": {"Service": "ec2.amazonaws.com"},
                        "Action": "sts:AssumeRole",
                    }
                ],
            }
            iam.create_role(
                RoleName=role_name,
                AssumeRolePolicyDocument=json.dumps(trust),
                Description="Temporary in-region client for strands-decider benchmark",
                MaxSessionDuration=3600,
                Tags=[
                    {"Key": "Purpose", "Value": "temporary-benchmark-client"},
                    {"Key": "DeleteAfter", "Value": delete_after},
                ],
            )
            role_created = True
            iam.attach_role_policy(RoleName=role_name, PolicyArn=SSM_POLICY_ARN)
            endpoint_arns = [
                f"arn:aws:sagemaker:{region}:{args.account}:endpoint/{endpoint}"
                for endpoint in endpoints
            ]
            inline_policy = {
                "Version": "2012-10-17",
                "Statement": [
                    {
                        "Sid": "DescribeEndpoints",
                        "Effect": "Allow",
                        "Action": [
                            "sagemaker:DescribeEndpoint",
                            "sagemaker:DescribeEndpointConfig",
                        ],
                        "Resource": "*",
                    },
                    {
                        "Sid": "InvokeConfiguredEndpoints",
                        "Effect": "Allow",
                        "Action": "sagemaker:InvokeEndpoint",
                        "Resource": endpoint_arns,
                    },
                    {
                        "Sid": "TransferBenchmarkFiles",
                        "Effect": "Allow",
                        "Action": ["s3:GetObject", "s3:PutObject"],
                        "Resource": f"arn:aws:s3:::{bucket_name}/{object_prefix}/*",
                    },
                ],
            }
            iam.put_role_policy(
                RoleName=role_name,
                PolicyName=policy_name,
                PolicyDocument=json.dumps(inline_policy),
            )
            iam.create_instance_profile(InstanceProfileName=profile_name)
            profile_created = True
            iam.add_role_to_instance_profile(
                InstanceProfileName=profile_name,
                RoleName=role_name,
            )
            print(f"created temporary role and instance profile {role_name}", flush=True)

            vpcs = ec2.describe_vpcs(Filters=[{"Name": "is-default", "Values": ["true"]}])["Vpcs"]
            if len(vpcs) != 1:
                raise RuntimeError(f"expected one default VPC, found {len(vpcs)}")
            vpc_id = vpcs[0]["VpcId"]
            group_id = ec2.create_security_group(
                GroupName=group_name,
                Description="No-ingress temporary benchmark client",
                VpcId=vpc_id,
                TagSpecifications=[
                    {
                        "ResourceType": "security-group",
                        "Tags": [
                            {"Key": "Purpose", "Value": "temporary-benchmark-client"},
                            {"Key": "DeleteAfter", "Value": delete_after},
                        ],
                    }
                ],
            )["GroupId"]
            print(f"created no-ingress security group {group_id}", flush=True)

            subnets = sorted(
                ec2.describe_subnets(Filters=[{"Name": "vpc-id", "Values": [vpc_id]}])["Subnets"],
                key=lambda subnet: subnet["AvailabilityZone"],
            )
            if not subnets:
                raise RuntimeError(f"no default-VPC subnet found in {region}")

            instance_type = ec2.describe_instance_types(InstanceTypes=[args.instance_type])[
                "InstanceTypes"
            ][0]
            architectures = instance_type["ProcessorInfo"]["SupportedArchitectures"]
            architecture = "arm64" if architectures == ["arm64"] else "x86_64"
            image_id = ssm.get_parameter(
                Name=(
                    f"/aws/service/ami-amazon-linux-latest/al2023-ami-kernel-default-{architecture}"
                )
            )["Parameter"]["Value"]

            # IAM instance-profile propagation is eventually consistent.
            time.sleep(15)
            instance = ec2.run_instances(
                ImageId=image_id,
                InstanceType=args.instance_type,
                MinCount=1,
                MaxCount=1,
                IamInstanceProfile={"Name": profile_name},
                InstanceInitiatedShutdownBehavior="terminate",
                MetadataOptions={
                    "HttpEndpoint": "enabled",
                    "HttpTokens": "required",
                    "HttpPutResponseHopLimit": 1,
                    "InstanceMetadataTags": "disabled",
                },
                NetworkInterfaces=[
                    {
                        "AssociatePublicIpAddress": True,
                        "DeleteOnTermination": True,
                        "DeviceIndex": 0,
                        "Groups": [group_id],
                        "SubnetId": subnets[0]["SubnetId"],
                    }
                ],
                BlockDeviceMappings=[
                    {
                        "DeviceName": "/dev/xvda",
                        "Ebs": {
                            "DeleteOnTermination": True,
                            "Encrypted": True,
                            "VolumeSize": 30,
                            "VolumeType": "gp3",
                        },
                    }
                ],
                TagSpecifications=[
                    {
                        "ResourceType": "instance",
                        "Tags": [
                            {"Key": "Name", "Value": "strands-decider-benchmark-client"},
                            {"Key": "Purpose", "Value": "temporary-benchmark-client"},
                            {"Key": "DeleteAfter", "Value": delete_after},
                        ],
                    }
                ],
            )["Instances"][0]
            instance_id = instance["InstanceId"]
            print(f"launched {args.instance_type} {instance_id}", flush=True)
            ec2.get_waiter("instance_running").wait(
                InstanceIds=[instance_id],
                WaiterConfig={"Delay": 5, "MaxAttempts": 120},
            )
            wait_for_ssm(ssm, instance_id)
            print(f"{instance_id} is online in SSM", flush=True)

            remote_script = build_remote_script(
                bucket_name=bucket_name,
                object_prefix=object_prefix,
                environment_path=environment_path,
                model=args.model,
                region=region,
                smoke=args.smoke,
            )
            command_id = ssm.send_command(
                InstanceIds=[instance_id],
                DocumentName="AWS-RunShellScript",
                TimeoutSeconds=args.command_timeout_seconds,
                Parameters={"commands": [remote_script]},
                Comment=f"Run strands-decider benchmark from {region}",
            )["Command"]["CommandId"]
            print(f"started SSM command {command_id}", flush=True)
            invocation = wait_for_command(ssm, command_id, instance_id)
            if invocation.get("StandardOutputContent"):
                print(invocation["StandardOutputContent"], flush=True)
            if invocation.get("StandardErrorContent"):
                print(invocation["StandardErrorContent"], file=sys.stderr, flush=True)

            try:
                s3.download_file(
                    bucket_name,
                    f"{object_prefix}/benchmark.log",
                    str(log_path),
                )
            except ClientError as error:
                if error.response["Error"]["Code"] not in {"404", "NoSuchKey"}:
                    raise
                log_path.write_text(
                    invocation.get("StandardOutputContent", "")
                    + "\n"
                    + invocation.get("StandardErrorContent", "")
                )
            if invocation["Status"] != "Success":
                failed_log = root / "benchmark-ec2-failure.log"
                failed_log.write_bytes(log_path.read_bytes())
                raise RuntimeError(
                    f"remote benchmark failed with {invocation['Status']}; log: {failed_log}"
                )

            s3.download_file(
                bucket_name,
                f"{object_prefix}/benchmark-results.tgz",
                str(result_archive),
            )
            extract_results(result_archive, root)
            print(f"downloaded results into {environment_file.parent / 'runs'}", flush=True)
            return 0
        finally:
            if instance_id:
                try:
                    ec2.terminate_instances(InstanceIds=[instance_id])
                    print(f"termination requested for {instance_id}", flush=True)
                    ec2.get_waiter("instance_terminated").wait(
                        InstanceIds=[instance_id],
                        WaiterConfig={"Delay": 5, "MaxAttempts": 120},
                    )
                    print(f"{instance_id} terminated", flush=True)
                except (ClientError, WaiterError) as error:
                    print(f"instance cleanup warning: {error}", file=sys.stderr, flush=True)

            if group_id:
                for _ in range(30):
                    try:
                        ec2.delete_security_group(GroupId=group_id)
                        print(f"deleted security group {group_id}", flush=True)
                        break
                    except ClientError as error:
                        if error.response["Error"]["Code"] != "DependencyViolation":
                            print(
                                f"security-group cleanup warning: {error}",
                                file=sys.stderr,
                                flush=True,
                            )
                            break
                        time.sleep(5)

            if profile_created:
                try:
                    iam.remove_role_from_instance_profile(
                        InstanceProfileName=profile_name,
                        RoleName=role_name,
                    )
                except ClientError:
                    pass
                try:
                    iam.delete_instance_profile(InstanceProfileName=profile_name)
                    print(f"deleted instance profile {profile_name}", flush=True)
                except ClientError as error:
                    print(f"profile cleanup warning: {error}", file=sys.stderr, flush=True)

            if role_created:
                try:
                    iam.delete_role_policy(RoleName=role_name, PolicyName=policy_name)
                except ClientError:
                    pass
                try:
                    iam.detach_role_policy(RoleName=role_name, PolicyArn=SSM_POLICY_ARN)
                except ClientError:
                    pass
                try:
                    iam.delete_role(RoleName=role_name)
                    print(f"deleted role {role_name}", flush=True)
                except ClientError as error:
                    print(f"role cleanup warning: {error}", file=sys.stderr, flush=True)

            if bucket_created:
                try:
                    empty_and_delete_bucket(s3, bucket_name)
                    print(f"deleted transfer bucket {bucket_name}", flush=True)
                except ClientError as error:
                    print(f"bucket cleanup warning: {error}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
