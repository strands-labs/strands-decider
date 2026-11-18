#!/usr/bin/env node
import { App } from "aws-cdk-lib";
import { DeciderGpuFunctionStack, stackNameFor } from "../lib/decider-function-stack";

const app = new App();

// One stack per model, so several models can be deployed side by side in one account and region.
new DeciderGpuFunctionStack(app, stackNameFor(app.node.tryGetContext("model")), {
  env: { account: process.env.CDK_DEFAULT_ACCOUNT, region: process.env.CDK_DEFAULT_REGION },
});
