import { readFileSync } from "fs";
import * as path from "path";
import { CfnOutput, Duration, IgnoreMode, RemovalPolicy, Stack, StackProps } from "aws-cdk-lib";
import * as codebuild from "aws-cdk-lib/aws-codebuild";
import * as ec2 from "aws-cdk-lib/aws-ec2";
import * as ecrAssets from "aws-cdk-lib/aws-ecr-assets";
import * as eks from "aws-cdk-lib/aws-eks-v2";
import * as iam from "aws-cdk-lib/aws-iam";
import * as s3 from "aws-cdk-lib/aws-s3";
import { KubectlV36Layer } from "@aws-cdk/lambda-layer-kubectl-v36";
import { Construct } from "constructs";
import { ContainerImageBuild, NodejsBuild } from "@cdklabs/deploy-time-build";

/**
 * The stack and cluster name for a model's Hugging Face Hub id: its name, lowercased, with every
 * character other than letters, digits and hyphens replaced by a hyphen, and `-eks` appended.
 * StrandsAgents/strands-decider-2B-qwen3.5-v1-2610 -> strands-decider-2b-qwen3-5-v1-2610-eks.
 */
export function stackNameFor(model: string): string {
  const name = (model.split("/").pop() ?? "")
    .toLowerCase()
    .replace(/[^a-z0-9-]+/g, "-")
    .replace(/^-+|-+$/g, "");
  if (!/^[a-z0-9](-*[a-z0-9]){0,58}$/.test(name)) {
    throw new Error(`${model}: cannot make a stack name (1 to 59 letters, digits, hyphens) from it`);
  }
  return `${name}-eks`;
}

/**
 * The vCPUs of a Graviton (arm64) instance type such as m9g.4xlarge, from its size. The image is
 * arm64, so x86 types fail, as do metal sizes, whose names do not give the count.
 */
export function vcpusFor(instanceType: string): number {
  const [family, size] = instanceType.toLowerCase().split(".");
  if (!/^[a-z]+\d+[a-z]*g[a-z]*$/.test(family) || /^g\d/.test(family)) {
    throw new Error(`${instanceType}: not a Graviton CPU instance type, such as m9g.4xlarge; the image is arm64`);
  }
  if (size === "large") return 2;
  const m = /^(\d*)xlarge$/.exec(size ?? "");
  if (!m) throw new Error(`${instanceType}: use a size from large to 48xlarge`);
  return 4 * Number(m[1] || 1);
}

/** A whole-number context value, or an error naming the setting. */
function intContext(scope: Construct, key: string, min: number): number {
  const value = Number(scope.node.tryGetContext(key));
  if (!Number.isInteger(value) || value < min) {
    throw new Error(`${key}=${scope.node.tryGetContext(key)}: must be a whole number, at least ${min}`);
  }
  return value;
}

const REPO_ROOT = path.join(__dirname, "..", "..", "..");

// The SageMaker guide's script, which writes the model and its base model in the layout the
// container loads from /opt/ml/model, run with the versions this package.json pins.
const FETCH_SCRIPTS = path.join(__dirname, "..", "..", "sagemaker", "scripts");
const PACKAGE = JSON.parse(readFileSync(path.join(__dirname, "..", "package.json"), "utf8"));
const FETCH_DEPENDENCIES = ["@huggingface/hub", "tsx"].map(
  (name) => `${name}@${(PACKAGE.devDependencies[name] as string).replace(/^[\^~]/, "")}`,
);

// The build uploads this directory to S3; keep only what the image needs, as
// Dockerfile.dockerignore does for a plain docker build.
const ASSET_EXCLUDE = [
  "*",
  "!pyproject.toml",
  "!README.md",
  "!LICENSE",
  "!src/",
  "!deploy/eks/Dockerfile",
  "**/__pycache__",
  "**/*.egg-info",
];

// Copies the model from S3 into the pod before the server starts. Multi-arch.
const AWS_CLI_IMAGE = "public.ecr.aws/aws-cli/aws-cli:2.38.2";
const NAMESPACE = "strands-decider";
const APP = "strands-decider";
const POOL_LABEL = "strands-decider/pool";
const TAINT = "strands-decider";

export interface DeciderEksStackProps extends StackProps {
  /** The cluster name, from stackNameFor(model). */
  readonly clusterName: string;
}

/**
 * strands-decider on EKS Auto Mode (README.md): a VPC, the cluster, a Graviton node pool, a private
 * bucket with the model, the image, and the Deployment, Service and autoscaler that serve it.
 * Settings are CDK context values (cdk.json, or -c key=value).
 */
export class DeciderEksStack extends Stack {
  constructor(scope: Construct, id: string, props: DeciderEksStackProps) {
    super(scope, id, props);
    const ctx = (key: string) => this.node.tryGetContext(key);
    const modelId: string = ctx("model");
    const instanceTypes = String(ctx("instance_types") ?? "").split(",").map((t) => t.trim().toLowerCase()).filter(Boolean);
    if (instanceTypes.length === 0) throw new Error("instance_types: give at least one, such as m9g.4xlarge");
    const vcpus = Math.max(...instanceTypes.map(vcpusFor));
    const threads = intContext(this, "threads", 1);
    const minReplicas = intContext(this, "min_replicas", 1);
    const maxReplicas = intContext(this, "max_replicas", minReplicas);
    if (threads > vcpus) {
      throw new Error(`threads=${threads}: at most ${vcpus}, the vCPUs of the largest instance type`);
    }
    const internalNlb = String(ctx("internal_nlb") ?? false).toLowerCase();
    if (internalNlb !== "true" && internalNlb !== "false") {
      throw new Error(`internal_nlb=${ctx("internal_nlb")}: must be true or false`);
    }
    // An access entry needs the IAM role or user, not the STS session that assumed the role.
    const admin: string = ctx("admin_principal") ?? "";
    if (admin && !/^arn:aws[a-z-]*:iam::\d{12}:(role|user)\//.test(admin)) {
      throw new Error(
        `admin_principal=${admin}: give an IAM role or user ARN, such as arn:aws:iam::<account>:role/<name>`,
      );
    }

    // The model, copied from the Hugging Face Hub into a private bucket during `cdk deploy`, as
    // in ../sagemaker. Each pod copies it from there (the init container below).
    const bucket = new s3.Bucket(this, "Models", {
      blockPublicAccess: s3.BlockPublicAccess.BLOCK_ALL,
      encryption: s3.BucketEncryption.S3_MANAGED,
      enforceSSL: true,
      removalPolicy: RemovalPolicy.DESTROY,
      autoDeleteObjects: true,
    });
    const modelPrefix = `${props.clusterName}/${ctx("model_revision")}/`;
    const modelData = new NodejsBuild(this, "ModelData", {
      assets: [{ path: FETCH_SCRIPTS, extractPath: "fetch" }],
      nodejsVersion: 22,
      computeType: codebuild.ComputeType.LARGE,
      buildEnvironment: { MODEL: modelId, MODEL_REVISION: ctx("model_revision") },
      buildCommands: [
        `npm install --no-save --no-audit --no-fund ${FETCH_DEPENDENCIES.join(" ")}`,
        'npx tsx fetch-model.ts "$MODEL" "$MODEL_REVISION" model',
      ],
      outputSourceDirectory: "model",
      destinationBucket: bucket,
      destinationKeyPrefix: modelPrefix,
    });

    // Built natively on an arm64 CodeBuild host during `cdk deploy`, and published to an ECR
    // repository in the stack. The image holds no model weights.
    const image = new ContainerImageBuild(this, "Image", {
      directory: REPO_ROOT,
      file: "deploy/eks/Dockerfile",
      exclude: ASSET_EXCLUDE,
      ignoreMode: IgnoreMode.DOCKER,
      platform: ecrAssets.Platform.LINUX_ARM64,
    });

    // Nodes in private subnets. The S3 gateway endpoint keeps model copies off the NAT gateway.
    const vpc = new ec2.Vpc(this, "Vpc", {
      maxAzs: 2,
      natGateways: 1,
      gatewayEndpoints: { S3: { service: ec2.GatewayVpcEndpointAwsService.S3 } },
    });

    const cluster = new eks.Cluster(this, "Cluster", {
      clusterName: props.clusterName,
      version: eks.KubernetesVersion.V1_36,
      vpc,
      vpcSubnets: [{ subnetType: ec2.SubnetType.PRIVATE_WITH_EGRESS }],
      defaultCapacityType: eks.DefaultCapacityType.AUTOMODE,
      kubectlProviderOptions: { kubectlLayer: new KubectlV36Layer(this, "KubectlLayer") },
    });
    // kubectl access for whoever runs it (README.md); the stack can deploy without it.
    if (admin) {
      cluster.grantAccess("Admin", admin, [
        eks.AccessPolicy.fromAccessPolicyName("AmazonEKSClusterAdminPolicy", {
          accessScopeType: eks.AccessScopeType.CLUSTER,
        }),
      ]);
    }
    // The autoscaler reads pod CPU from it.
    const metricsServer = new eks.Addon(this, "MetricsServer", { cluster, addonName: "metrics-server" });

    const namespace = cluster.addManifest("Namespace", {
      apiVersion: "v1",
      kind: "Namespace",
      metadata: { name: NAMESPACE },
    });
    // EKS Pod Identity, which Auto Mode has built in: the pods read the model bucket and nothing
    // else. (cluster.addServiceAccount would also install the agent add-on Auto Mode does not need.)
    const podRole = new iam.Role(this, "PodRole", {
      assumedBy: new iam.ServicePrincipal("pods.eks.amazonaws.com").withSessionTags().withConditions({
        StringEquals: {
          "aws:RequestTag/eks-cluster-name": props.clusterName,
          "aws:RequestTag/kubernetes-namespace": NAMESPACE,
          "aws:RequestTag/kubernetes-service-account": APP,
        },
      }),
    });
    bucket.grantRead(podRole);
    const serviceAccount = cluster.addManifest("ServiceAccount", {
      apiVersion: "v1",
      kind: "ServiceAccount",
      metadata: { name: APP, namespace: NAMESPACE },
    });
    serviceAccount.node.addDependency(namespace);
    const podIdentity = new eks.CfnPodIdentityAssociation(this, "PodIdentity", {
      clusterName: cluster.clusterName,
      namespace: NAMESPACE,
      serviceAccount: APP,
      roleArn: podRole.roleArn,
    });

    // Auto Mode's built-in general-purpose pool is amd64 only, so the model gets its own arm64
    // pool. The taint keeps other pods off it; the CPU limit caps it at max_replicas + 1 nodes of
    // the largest type, the one for the extra pod of a rolling update.
    const nodePool = cluster.addManifest("NodePool", {
      apiVersion: "karpenter.sh/v1",
      kind: "NodePool",
      metadata: { name: APP },
      spec: {
        template: {
          metadata: { labels: { [POOL_LABEL]: APP } },
          spec: {
            nodeClassRef: { group: "eks.amazonaws.com", kind: "NodeClass", name: "default" },
            requirements: [
              { key: "node.kubernetes.io/instance-type", operator: "In", values: instanceTypes },
              { key: "kubernetes.io/arch", operator: "In", values: ["arm64"] },
              { key: "karpenter.sh/capacity-type", operator: "In", values: ["on-demand"] },
            ],
            taints: [{ key: TAINT, value: "true", effect: "NoSchedule" }],
          },
        },
        disruption: { consolidationPolicy: "WhenEmpty", consolidateAfter: "2m" },
        limits: { cpu: String((maxReplicas + 1) * vcpus) },
      },
    });

    const modelDir = `/opt/ml/model/${modelId.split("/").pop()}`;
    const deployment = cluster.addManifest("Deployment", {
      apiVersion: "apps/v1",
      kind: "Deployment",
      metadata: { name: APP, namespace: NAMESPACE, labels: { app: APP } },
      spec: {
        // No replicas: the autoscaler owns the count, and a later `cdk deploy` must not reset it.
        selector: { matchLabels: { app: APP } },
        strategy: { type: "RollingUpdate", rollingUpdate: { maxSurge: 1, maxUnavailable: 0 } },
        template: {
          metadata: { labels: { app: APP } },
          spec: {
            serviceAccountName: APP,
            // Room for the preStop sleep below, then for the requests in flight to finish.
            terminationGracePeriodSeconds: 60,
            nodeSelector: { [POOL_LABEL]: APP },
            tolerations: [{ key: TAINT, operator: "Equal", value: "true", effect: "NoSchedule" }],
            // The model sits in memory: loading it from there is faster than from the node's disk.
            volumes: [{ name: "model", emptyDir: { medium: "Memory", sizeLimit: "8Gi" } }],
            initContainers: [
              {
                name: "fetch-model",
                image: AWS_CLI_IMAGE,
                command: [
                  "sh",
                  "-c",
                  "aws configure set default.s3.max_concurrent_requests 32 && " +
                    'aws s3 cp --recursive --only-show-errors "$MODEL_S3_URI" /opt/ml/model',
                ],
                env: [
                  { name: "MODEL_S3_URI", value: bucket.s3UrlForObject(modelPrefix) },
                  { name: "AWS_REGION", value: this.region },
                ],
                volumeMounts: [{ name: "model", mountPath: "/opt/ml/model" }],
              },
            ],
            containers: [
              {
                name: "decider",
                image: image.imageUri,
                securityContext: {
                  runAsNonRoot: true,
                  allowPrivilegeEscalation: false,
                  capabilities: { drop: ["ALL"] },
                  seccompProfile: { type: "RuntimeDefault" },
                },
                ports: [{ name: "http", containerPort: 8080 }],
                env: [
                  { name: "DECIDER_CHECKPOINT", value: modelDir },
                  { name: "DECIDER_ARGS", value: ctx("serve_args") ?? "" },
                  { name: "OMP_NUM_THREADS", value: String(threads) },
                ],
                volumeMounts: [{ name: "model", mountPath: "/opt/ml/model" }],
                // A CPU request but no CPU limit, so torch's threads are never throttled. The
                // request leaves room for the kubelet, so a node with as many vCPUs as `threads` fits
                // one pod. The 2B models peak at about 9.5 GB in fp32; the model in memory adds 4.5 GB.
                resources: {
                  requests: { cpu: String(Math.max(1, Math.floor(threads * 0.85))), memory: "16Gi" },
                  limits: { memory: "24Gi" },
                },
                // The port opens once the model has loaded. After that, only check the port: /ping
                // shares the server's thread pool with requests, so a busy pod can answer it late,
                // and taking that pod out of the Service, or restarting it, would only lose work.
                startupProbe: { tcpSocket: { port: "http" }, periodSeconds: 5, failureThreshold: 120 },
                readinessProbe: { tcpSocket: { port: "http" }, periodSeconds: 5 },
                // A pod that is stopping keeps serving for 30 s while the Service and the load
                // balancer stop sending it new connections; without this, a request can wait
                // minutes on TCP retries.
                lifecycle: { preStop: { exec: { command: ["sleep", "30"] } } },
              },
            ],
          },
        },
      },
    });
    deployment.node.addDependency(serviceAccount, podIdentity, podRole, nodePool, modelData);
    // The first `cdk deploy` waits for a pod to be ready, and fails (rolling back) if none is in
    // 10 minutes. Later deploys do not wait.
    const ready = new eks.KubernetesObjectValue(this, "Ready", {
      cluster,
      objectType: "deployment",
      objectName: APP,
      objectNamespace: NAMESPACE,
      jsonPath: ".status.readyReplicas",
      timeout: Duration.minutes(10),
    });
    ready.node.addDependency(deployment);

    // Node replacements evict one pod at a time, so with min_replicas 2 one pod keeps serving.
    // With one pod, maxUnavailable 1 still lets the node drain.
    const disruptionBudget = cluster.addManifest("DisruptionBudget", {
      apiVersion: "policy/v1",
      kind: "PodDisruptionBudget",
      metadata: { name: APP, namespace: NAMESPACE },
      spec: { maxUnavailable: 1, selector: { matchLabels: { app: APP } } },
    });
    disruptionBudget.node.addDependency(namespace);

    const service = cluster.addManifest("Service", {
      apiVersion: "v1",
      kind: "Service",
      metadata: { name: APP, namespace: NAMESPACE },
      spec: { selector: { app: APP }, ports: [{ name: "http", port: 8080, targetPort: "http" }] },
    });
    service.node.addDependency(namespace);

    // Adds pods when their CPU use passes 60% of the request; Auto Mode adds nodes for them. CPU
    // shows how busy pods are, not how many requests wait: a pod answering one caller back to back
    // uses about 95%, and gets a second pod.
    const autoscaler = cluster.addManifest("Autoscaler", {
      apiVersion: "autoscaling/v2",
      kind: "HorizontalPodAutoscaler",
      metadata: { name: APP, namespace: NAMESPACE },
      spec: {
        scaleTargetRef: { apiVersion: "apps/v1", kind: "Deployment", name: APP },
        minReplicas,
        maxReplicas,
        metrics: [{ type: "Resource", resource: { name: "cpu", target: { type: "Utilization", averageUtilization: 60 } } }],
      },
    });
    autoscaler.node.addDependency(deployment, metricsServer);

    // An internal Network Load Balancer, for callers in the VPC. The server has no
    // authentication of its own, so it is never internet-facing.
    if (internalNlb === "true") {
      const nlb = cluster.addManifest("InternalNlb", {
        apiVersion: "v1",
        kind: "Service",
        metadata: {
          name: `${APP}-nlb`,
          namespace: NAMESPACE,
          annotations: {
            "service.beta.kubernetes.io/aws-load-balancer-scheme": "internal",
            "service.beta.kubernetes.io/aws-load-balancer-nlb-target-type": "ip",
            // Pods are few and can sit in one zone, so every zone of the NLB sends to all of
            // them. A stopping pod drains for 30 s, while its preStop sleep keeps it serving.
            "service.beta.kubernetes.io/aws-load-balancer-attributes": "load_balancing.cross_zone.enabled=true",
            "service.beta.kubernetes.io/aws-load-balancer-target-group-attributes": "deregistration_delay.timeout_seconds=30",
          },
        },
        spec: {
          type: "LoadBalancer",
          loadBalancerClass: "eks.amazonaws.com/nlb",
          // Only addresses in the VPC can connect; without this the security group allows 0.0.0.0/0.
          loadBalancerSourceRanges: [vpc.vpcCidrBlock],
          selector: { app: APP },
          ports: [{ name: "http", port: 80, targetPort: "http" }],
        },
      });
      nlb.node.addDependency(namespace);
      const nlbAddress = new eks.KubernetesObjectValue(this, "InternalNlbAddress", {
        cluster,
        objectType: "service",
        objectName: `${APP}-nlb`,
        objectNamespace: NAMESPACE,
        jsonPath: ".status.loadBalancer.ingress[0].hostname",
        timeout: Duration.minutes(10),
      });
      nlbAddress.node.addDependency(nlb);
      new CfnOutput(this, "InternalNlbUrl", { value: `http://${nlbAddress.value}` });
    }

    new CfnOutput(this, "ClusterName", { value: cluster.clusterName });
    new CfnOutput(this, "ServiceUrl", { value: `http://${APP}.${NAMESPACE}.svc.cluster.local:8080` });
    new CfnOutput(this, "ModelDataS3Uri", { value: bucket.s3UrlForObject(modelPrefix) });
    new CfnOutput(this, "ImageUri", { value: image.imageUri });
  }
}
