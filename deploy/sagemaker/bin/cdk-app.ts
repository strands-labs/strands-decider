#!/usr/bin/env node
import { App } from "aws-cdk-lib";
import { DeciderEndpointStack, endpointNameFor } from "../lib/decider-endpoint-stack";

const app = new App();

// One stack and endpoint per model, both named after it, so several models can be deployed
// side by side in one account and region.
const endpointName = endpointNameFor(app.node.tryGetContext("model"));

new DeciderEndpointStack(app, endpointName, {
  endpointName,
  // The account and region of the current credentials and CLI configuration.
  env: { account: process.env.CDK_DEFAULT_ACCOUNT, region: process.env.CDK_DEFAULT_REGION },
});
