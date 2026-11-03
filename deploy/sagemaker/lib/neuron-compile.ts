import * as path from "path";
import { CustomResource, Duration, Fn, RemovalPolicy, Stack } from "aws-cdk-lib";
import * as ec2 from "aws-cdk-lib/aws-ec2";
import * as ecr from "aws-cdk-lib/aws-ecr";
import * as iam from "aws-cdk-lib/aws-iam";
import * as lambda from "aws-cdk-lib/aws-lambda";
import * as logs from "aws-cdk-lib/aws-logs";
import * as s3 from "aws-cdk-lib/aws-s3";
import * as cr from "aws-cdk-lib/custom-resources";
import { Construct } from "constructs";

/** The Neuron deep learning AMI: Ubuntu with the Neuron driver, which the host needs. */
const NEURON_DLAMI_PARAMETER = "/aws/service/neuron/dlami/multi-framework/ubuntu-24.04/latest/image_id";

export interface NeuronCompileProps {
  /** The bucket with the model data (scripts/fetch-model.ts's layout). */
  readonly bucket: s3.IBucket;
  /** Prefix of the model data in the bucket, ending in "/". */
  readonly sourcePrefix: string;
  /** Prefix to write the model data with its compiled graphs to, ending in "/". */
  readonly targetPrefix: string;
  /** The serving image, which compiles: a graph only loads in the install that compiled it. */
  readonly repository: ecr.IRepository;
  readonly imageUri: string;
  /** The checkpoint's folder name under /opt/ml/model. */
  readonly checkpoint: string;
  /** The EC2 instance type that compiles, the SageMaker type without its "ml." prefix. */
  readonly instanceType: string;
  /**
   * An existing VPC and subnets to compile in, instead of a VPC of its own. The subnets must
   * reach S3, ECR and CloudWatch Logs: public subnets that assign public IPs, or private ones
   * with a NAT gateway or VPC endpoints. Each is tried in turn when one has no capacity.
   */
  readonly network?: { readonly vpcId: string; readonly subnetIds: string[] };
}

/**
 * Compiles a model for Neuron during `cdk deploy`, once per image and model data.
 *
 * A custom resource starts a temporary EC2 instance of the endpoint's instance type, which runs
 * `strands-decider compile-neuron` in the serving image (about 35 minutes for a 2B model on
 * inf2.xlarge, plus 10 to 15 to start), copies the model data with the compiled graphs to
 * `targetPrefix`, and terminates itself. The deploy waits for it, up to two hours. The target
 * prefix names the image, so a new image compiles again and the old graphs are left unused.
 *
 * The instance is in a VPC of its own with public subnets and no NAT gateway, or in `network`.
 * Its security group admits no inbound traffic. The compile container runs without network or AWS credentials.
 */
export class NeuronCompile extends Construct {
  /** Depend on this to wait for the compiled model data. */
  readonly resource: CustomResource;

  constructor(scope: Construct, id: string, props: NeuronCompileProps) {
    super(scope, id);
    const stack = Stack.of(this);

    let vpc: ec2.IVpc;
    let subnetIds: string[];
    if (props.network) {
      // Only the security group needs the VPC, and only its id; the subnets are given.
      vpc = ec2.Vpc.fromVpcAttributes(this, "Vpc", { vpcId: props.network.vpcId, availabilityZones: Fn.getAzs() });
      subnetIds = props.network.subnetIds;
    } else {
      const own = new ec2.Vpc(this, "Vpc", {
        maxAzs: 99, // every zone: the instance type is not offered in all of them
        natGateways: 0,
        subnetConfiguration: [{ name: "public", subnetType: ec2.SubnetType.PUBLIC, cidrMask: 24 }],
      });
      vpc = own;
      subnetIds = own.publicSubnets.map((s) => s.subnetId);
    }
    const securityGroup = new ec2.SecurityGroup(this, "HostSecurityGroup", {
      vpc,
      allowAllOutbound: false,
      description: "strands-decider Neuron compile host: no inbound traffic",
    });
    securityGroup.addEgressRule(ec2.Peer.anyIpv4(), ec2.Port.tcp(443), "S3, ECR and CloudWatch Logs");
    securityGroup.addEgressRule(ec2.Peer.anyIpv4(), ec2.Port.tcp(80), "Ubuntu package mirrors, if the AMI lacks docker");

    const logGroup = new logs.LogGroup(this, "Logs", {
      retention: logs.RetentionDays.ONE_MONTH,
      removalPolicy: RemovalPolicy.DESTROY,
    });
    const hostRole = new iam.Role(this, "HostRole", {
      assumedBy: new iam.ServicePrincipal("ec2.amazonaws.com"),
    });
    props.bucket.grantRead(hostRole, `${props.sourcePrefix}*`);
    props.bucket.grantReadWrite(hostRole, `${props.targetPrefix}*`); // aws s3 sync lists the target
    props.repository.grantPull(hostRole);
    logGroup.grantWrite(hostRole);

    const launchTemplate = new ec2.LaunchTemplate(this, "Host", {
      instanceType: new ec2.InstanceType(props.instanceType),
      machineImage: ec2.MachineImage.fromSsmParameter(NEURON_DLAMI_PARAMETER),
      role: hostRole,
      securityGroup,
      requireImdsv2: true,
      httpPutResponseHopLimit: 1, // the container cannot reach the instance role
      instanceInitiatedShutdownBehavior: ec2.InstanceInitiatedShutdownBehavior.TERMINATE,
      blockDevices: [
        {
          deviceName: "/dev/sda1",
          volume: ec2.BlockDeviceVolume.ebs(128, { volumeType: ec2.EbsDeviceVolumeType.GP3, encrypted: true }),
        },
      ],
    });

    const code = lambda.Code.fromAsset(path.join(__dirname, "neuron-compile"));
    const handler = (name: string) =>
      new lambda.Function(this, name, {
        runtime: lambda.Runtime.PYTHON_3_13,
        code,
        handler: `handler.${name === "OnEvent" ? "on_event" : "is_complete"}`,
        timeout: Duration.minutes(2),
      });
    const onEvent = handler("OnEvent");
    const isComplete = handler("IsComplete");
    const ec2Arn = (resource: string) => `arn:${stack.partition}:ec2:${stack.region}:${stack.account}:${resource}`;
    onEvent.addToRolePolicy(
      new iam.PolicyStatement({
        actions: ["ec2:RunInstances"],
        resources: [
          ec2Arn("instance/*"),
          ec2Arn("volume/*"),
          ec2Arn("network-interface/*"),
          ec2Arn(`launch-template/${launchTemplate.launchTemplateId}`),
          ...subnetIds.map((id) => ec2Arn(`subnet/${id}`)),
          ec2Arn(`security-group/${securityGroup.securityGroupId}`),
          `arn:${stack.partition}:ec2:${stack.region}::image/*`,
        ],
      }),
    );
    onEvent.addToRolePolicy(
      new iam.PolicyStatement({
        actions: ["ec2:CreateTags"],
        resources: [ec2Arn("instance/*"), ec2Arn("volume/*"), ec2Arn("network-interface/*")],
        conditions: { StringEquals: { "ec2:CreateAction": "RunInstances" } },
      }),
    );
    onEvent.addToRolePolicy(
      new iam.PolicyStatement({
        actions: ["ec2:TerminateInstances"],
        resources: [ec2Arn("instance/*")],
        conditions: { StringEquals: { "aws:ResourceTag/strands-decider:neuron-compile": props.targetPrefix } },
      }),
    );
    hostRole.grantPassRole(onEvent.grantPrincipal);
    for (const fn of [onEvent, isComplete]) {
      fn.addToRolePolicy(new iam.PolicyStatement({ actions: ["ec2:DescribeInstances"], resources: ["*"] }));
      props.bucket.grantRead(fn, `${props.targetPrefix}*`);
    }
    props.bucket.grantDelete(onEvent, `${props.targetPrefix}_FAILED`);

    const provider = new cr.Provider(this, "Provider", {
      onEventHandler: onEvent,
      isCompleteHandler: isComplete,
      queryInterval: Duration.minutes(1),
      totalTimeout: Duration.hours(2),
    });
    this.resource = new CustomResource(this, "Resource", {
      serviceToken: provider.serviceToken,
      properties: {
        Bucket: props.bucket.bucketName,
        Source: props.sourcePrefix,
        Target: props.targetPrefix,
        Image: props.imageUri,
        Checkpoint: props.checkpoint,
        InstanceType: props.instanceType,
        LaunchTemplate: launchTemplate.launchTemplateId,
        Subnets: subnetIds,
        Region: stack.region,
        LogGroup: logGroup.logGroupName,
      },
    });
  }
}
