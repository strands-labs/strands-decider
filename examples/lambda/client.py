"""Call the deployed function's IAM-auth Function URL, signed with SigV4.

usage: python examples/lambda/client.py [--stack NAME] [--region REGION] {health,ask,bench} [N]

Uses boto3's usual credentials; the caller needs lambda:InvokeFunctionUrl on the function.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
import urllib.request

REQUEST = {
    "state": "Help! My payouts have been failing for 3 days!",
    "questions": {
        "is_urgent": {"type": "noul", "instructions": "Does this convey urgency?"},
        "team": {
            "type": "choice",
            "instructions": "Which team should handle this?",
            "criteria": {"billing": "", "sales": "", "retail": ""},
        },
        "frustration": {
            "type": "score",
            "instructions": "How frustrated is the writer?",
            "criteria": ["calm", "frustrated", "depressed"],
        },
    },
}


class Client:
    def __init__(self, stack: str, region: str | None):
        import boto3  # here, so local_test.sh can import REQUEST without boto3

        self.session = boto3.Session(region_name=region)
        outputs = self.session.client("cloudformation").describe_stacks(StackName=stack)["Stacks"][
            0
        ]["Outputs"]
        self.url = next(
            o["OutputValue"] for o in outputs if o["OutputKey"] == "FunctionUrl"
        ).rstrip("/")

    def call(self, method: str, path: str, body: dict | None = None) -> tuple[dict, float]:
        from botocore.auth import SigV4Auth
        from botocore.awsrequest import AWSRequest

        data = json.dumps(body).encode() if body is not None else None
        req = AWSRequest(
            method=method,
            url=self.url + path,
            data=data,
            headers={"content-type": "application/json"},
        )
        creds = self.session.get_credentials().get_frozen_credentials()
        SigV4Auth(creds, "lambda", self.session.region_name).add_auth(req)
        t0 = time.perf_counter()
        with urllib.request.urlopen(
            urllib.request.Request(req.url, data=data, headers=dict(req.headers), method=method),
            timeout=900,
        ) as r:
            out = json.loads(r.read())
        return out, (time.perf_counter() - t0) * 1000


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stack", default="strands-decider-lambda")
    ap.add_argument("--region", default=None)
    ap.add_argument("cmd", choices=["health", "ask", "bench"], nargs="?", default="ask")
    ap.add_argument("n", type=int, nargs="?", default=10)
    args = ap.parse_args()
    c = Client(args.stack, args.region)

    if args.cmd == "health":
        print(json.dumps(c.call("GET", "/health")[0], indent=2))
    elif args.cmd == "ask":
        out, ms = c.call("POST", "/v1/systemone", REQUEST)
        print(json.dumps(out, indent=2))
        print(f"round trip {ms:.0f} ms")
    else:
        rt, srv = [], []
        for _ in range(args.n):
            out, ms = c.call("POST", "/v1/systemone", REQUEST)
            rt.append(ms)
            srv.append(out["latency_ms"])
        print(
            f"n={args.n} server median {statistics.median(srv):.0f} ms, "
            f"round trip median {statistics.median(rt):.0f} ms, cpu_part {out.get('cpu_part')}"
        )


if __name__ == "__main__":
    main()
