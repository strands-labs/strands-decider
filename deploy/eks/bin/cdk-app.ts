#!/usr/bin/env node
import { App } from "aws-cdk-lib";
import { DeciderEksStack, stackNameFor } from "../lib/decider-eks-stack";

const app = new App();

// One stack and cluster per model, both named after it with an -eks suffix, so it does not clash
// with the SageMaker stack for the same model (../sagemaker).
const name = stackNameFor(app.node.tryGetContext("model"));

new DeciderEksStack(app, name, {
  clusterName: name,
  // The account and region of the current credentials and CLI configuration.
  env: { account: process.env.CDK_DEFAULT_ACCOUNT, region: process.env.CDK_DEFAULT_REGION },
});
