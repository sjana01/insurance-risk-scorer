"""
Bootstrap all AWS IAM roles, policies, and S3 bucket for the Insurance Risk Scorer.

Run this AFTER configuring the AWS CLI (`aws configure`).
It is fully idempotent — safe to re-run.

Usage:
    python infra/bootstrap_iam.py
    python infra/bootstrap_iam.py --region eu-west-1
    python infra/bootstrap_iam.py --dry-run
"""

import argparse
import json
import sys
import time

import boto3
from botocore.exceptions import ClientError

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SM_ROLE_NAME = "InsuranceRiskScorerSageMakerRole"
LAMBDA_ROLE_NAME = "InsuranceRiskScorerLambdaRole"
ENDPOINT_NAME = "insurance-risk-scorer"
PROJECT_TAG = [{"Key": "Project", "Value": "insurance-risk-scorer"}]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_account_id(sts) -> str:
    return sts.get_caller_identity()["Account"]


def role_exists(iam, role_name: str) -> bool:
    try:
        iam.get_role(RoleName=role_name)
        return True
    except ClientError as e:
        if e.response["Error"]["Code"] == "NoSuchEntity":
            return False
        raise


def create_role(iam, role_name: str, trust_service: str, description: str, dry_run: bool) -> str:
    """Create an IAM role with a service trust policy. Returns role ARN."""
    trust_policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Principal": {"Service": f"{trust_service}.amazonaws.com"},
                "Action": "sts:AssumeRole",
            }
        ],
    }

    if role_exists(iam, role_name):
        arn = iam.get_role(RoleName=role_name)["Role"]["Arn"]
        print(f"  [exists]  {role_name} -> {arn}")
        return arn

    if dry_run:
        print(f"  [dry-run] Would create role: {role_name}")
        return f"arn:aws:iam::DRY_RUN:role/{role_name}"

    response = iam.create_role(
        RoleName=role_name,
        AssumeRolePolicyDocument=json.dumps(trust_policy),
        Description=description,
        Tags=PROJECT_TAG,
    )
    arn = response["Role"]["Arn"]
    print(f"  [created] {role_name} -> {arn}")
    return arn


def attach_managed_policy(iam, role_name: str, policy_arn: str, dry_run: bool) -> None:
    if dry_run:
        print(f"  [dry-run] Would attach {policy_arn.split('/')[-1]}")
        return
    try:
        iam.attach_role_policy(RoleName=role_name, PolicyArn=policy_arn)
        print(f"  [attached] {policy_arn.split('/')[-1]}")
    except ClientError as e:
        if "already attached" in str(e).lower() or e.response["Error"]["Code"] == "PolicyNotAttachable":
            print(f"  [already]  {policy_arn.split('/')[-1]}")
        else:
            raise


def put_inline_policy(iam, role_name: str, policy_name: str, policy_doc: dict, dry_run: bool) -> None:
    if dry_run:
        print(f"  [dry-run] Would put inline policy: {policy_name}")
        return
    iam.put_role_policy(
        RoleName=role_name,
        PolicyName=policy_name,
        PolicyDocument=json.dumps(policy_doc),
    )
    print(f"  [inline]   {policy_name}")


def create_s3_bucket(s3, bucket_name: str, region: str, dry_run: bool) -> None:
    if dry_run:
        print(f"  [dry-run] Would create bucket: s3://{bucket_name}")
        return

    try:
        if region == "us-east-1":
            s3.create_bucket(Bucket=bucket_name)
        else:
            s3.create_bucket(
                Bucket=bucket_name,
                CreateBucketConfiguration={"LocationConstraint": region},
            )
        s3.put_bucket_tagging(
            Bucket=bucket_name,
            Tagging={"TagSet": PROJECT_TAG},
        )
        # Block all public access
        s3.put_public_access_block(
            Bucket=bucket_name,
            PublicAccessBlockConfiguration={
                "BlockPublicAcls": True,
                "IgnorePublicAcls": True,
                "BlockPublicPolicy": True,
                "RestrictPublicBuckets": True,
            },
        )
        print(f"  [created] s3://{bucket_name}")
    except ClientError as e:
        code = e.response["Error"]["Code"]
        if code in ("BucketAlreadyOwnedByYou", "BucketAlreadyExists"):
            print(f"  [exists]  s3://{bucket_name}")
        else:
            raise


# ---------------------------------------------------------------------------
# Main bootstrap
# ---------------------------------------------------------------------------

def bootstrap(region: str, dry_run: bool) -> dict:
    session = boto3.Session(region_name=region)
    iam = session.client("iam")
    s3 = session.client("s3", region_name=region)
    sts = session.client("sts")

    print("Verifying AWS credentials...")
    identity = sts.get_caller_identity()
    account_id = identity["Account"]
    print(f"  Account ID : {account_id}")
    print(f"  Caller ARN : {identity['Arn']}")
    print(f"  Region     : {region}")
    print()

    bucket_name = f"insurance-risk-scorer-{account_id}"

    # ---- SageMaker role ------------------------------------------------
    print("Creating SageMaker execution role...")
    sm_role_arn = create_role(
        iam,
        SM_ROLE_NAME,
        trust_service="sagemaker",
        description="SageMaker execution role for Insurance Risk Scorer",
        dry_run=dry_run,
    )
    attach_managed_policy(iam, SM_ROLE_NAME, "arn:aws:iam::aws:policy/AmazonSageMakerFullAccess", dry_run)
    attach_managed_policy(iam, SM_ROLE_NAME, "arn:aws:iam::aws:policy/AmazonS3FullAccess", dry_run)
    print()

    # ---- Lambda role ---------------------------------------------------
    print("Creating Lambda execution role...")
    lambda_role_arn = create_role(
        iam,
        LAMBDA_ROLE_NAME,
        trust_service="lambda",
        description="Lambda execution role for Insurance Risk Scorer API",
        dry_run=dry_run,
    )
    attach_managed_policy(iam, LAMBDA_ROLE_NAME, "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole", dry_run)
    put_inline_policy(
        iam,
        LAMBDA_ROLE_NAME,
        "InvokeSageMakerEndpoint",
        {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Action": "sagemaker:InvokeEndpoint",
                    "Resource": f"arn:aws:sagemaker:*:{account_id}:endpoint/{ENDPOINT_NAME}",
                }
            ],
        },
        dry_run,
    )
    print()

    # ---- S3 bucket -----------------------------------------------------
    print("Creating S3 bucket...")
    create_s3_bucket(s3, bucket_name, region, dry_run)
    print()

    # ---- Wait for roles to propagate -----------------------------------
    if not dry_run:
        print("Waiting 10 s for IAM propagation...")
        time.sleep(10)

    return {
        "account_id": account_id,
        "region": region,
        "sm_role_arn": sm_role_arn,
        "lambda_role_arn": lambda_role_arn,
        "bucket_name": bucket_name,
        "endpoint_name": ENDPOINT_NAME,
    }


def print_env_block(config: dict) -> None:
    """Print ready-to-paste environment variable exports."""
    print("=" * 60)
    print("Environment variables — paste into your shell profile")
    print("=" * 60)

    # PowerShell (Windows)
    print("\n# PowerShell ($PROFILE or run in terminal):")
    for key, val in [
        ("SAGEMAKER_ROLE_ARN", config["sm_role_arn"]),
        ("LAMBDA_ROLE_ARN", config["lambda_role_arn"]),
        ("S3_BUCKET", config["bucket_name"]),
        ("AWS_REGION", config["region"]),
        ("ENDPOINT_NAME", config["endpoint_name"]),
    ]:
        print(f'$env:{key} = "{val}"')

    # Bash / zsh
    print("\n# Bash / zsh (~/.bashrc or ~/.zshrc):")
    for key, val in [
        ("SAGEMAKER_ROLE_ARN", config["sm_role_arn"]),
        ("LAMBDA_ROLE_ARN", config["lambda_role_arn"]),
        ("S3_BUCKET", config["bucket_name"]),
        ("AWS_REGION", config["region"]),
        ("ENDPOINT_NAME", config["endpoint_name"]),
    ]:
        print(f'export {key}="{val}"')

    print("=" * 60)

    print("\nVerification commands:")
    print("  aws sts get-caller-identity")
    print(f"  aws s3 ls s3://{config['bucket_name']}/")
    print('  python -c "import boto3, sagemaker; print(\'OK\')"')


def main():
    parser = argparse.ArgumentParser(description="Bootstrap IAM roles and S3 bucket for Insurance Risk Scorer")
    parser.add_argument("--region", default="us-east-1", help="AWS region (default: us-east-1)")
    parser.add_argument("--dry-run", action="store_true", help="Print actions without creating resources")
    args = parser.parse_args()

    if args.dry_run:
        print("=== DRY RUN — no resources will be created ===\n")

    try:
        config = bootstrap(region=args.region, dry_run=args.dry_run)
    except ClientError as e:
        code = e.response["Error"]["Code"]
        if code in ("InvalidClientTokenId", "AuthFailure", "ExpiredTokenException"):
            print(f"\nERROR: AWS credentials invalid or expired.")
            print("Run `aws configure` to set up your credentials first.")
            sys.exit(1)
        if code == "AccessDenied":
            print(f"\nERROR: Access denied — your IAM user needs IAM and S3 permissions.")
            print("Make sure the user has AdministratorAccess or the required policies.")
            sys.exit(1)
        raise

    if not args.dry_run:
        print_env_block(config)


if __name__ == "__main__":
    main()
