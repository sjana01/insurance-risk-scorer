"""
Cleanup script — deletes all billable AWS resources created by this project.

Resources deleted (in safe order):
  1. API Gateway REST API
  2. Lambda function
  3. SageMaker endpoint          ← stops hourly billing
  4. SageMaker endpoint config
  5. SageMaker model
  6. CloudWatch alarms
  7. (Optional) S3 data prefix

Usage:
    python infra/cleanup.py                    # uses .api_config.json + env vars
    python infra/cleanup.py --delete-s3        # also empties S3 prefix
    python infra/cleanup.py --dry-run          # prints what would be deleted
"""

import argparse
import json
import os

import boto3
from botocore.exceptions import ClientError

REGION = os.environ.get("AWS_REGION", "us-east-1")
BUCKET = os.environ.get("S3_BUCKET", "insurance-risk-scorer-ACCOUNT_ID")
ENDPOINT_NAME = os.environ.get("ENDPOINT_NAME", "insurance-risk-scorer")
FUNCTION_NAME = "insurance-risk-scorer-api"

sm = boto3.client("sagemaker", region_name=REGION)
lambda_client = boto3.client("lambda", region_name=REGION)
apigw = boto3.client("apigateway", region_name=REGION)
cw = boto3.client("cloudwatch", region_name=REGION)
s3 = boto3.client("s3", region_name=REGION)


def safe_delete(name: str, fn, dry_run: bool, **kwargs) -> None:
    if dry_run:
        print(f"[DRY RUN] Would delete: {name}")
        return
    try:
        fn(**kwargs)
        print(f"Deleted: {name}")
    except ClientError as e:
        code = e.response["Error"]["Code"]
        if code in ("ResourceNotFound", "ResourceNotFoundException", "NoSuchEntity", "NotFoundException"):
            print(f"Already gone: {name}")
        else:
            print(f"ERROR deleting {name}: {e}")


def load_api_config() -> dict:
    config_path = os.path.join(os.path.dirname(__file__), "..", "api", ".api_config.json")
    if os.path.exists(config_path):
        with open(config_path) as f:
            return json.load(f)
    return {}


def get_sm_model_name(endpoint_name: str) -> str:
    """Look up the model name from the endpoint config."""
    try:
        ep = sm.describe_endpoint(EndpointName=endpoint_name)
        cfg_name = ep["EndpointConfigName"]
        cfg = sm.describe_endpoint_config(EndpointConfigName=cfg_name)
        return cfg["ProductionVariants"][0]["ModelName"]
    except Exception:
        return endpoint_name  # fallback


def delete_s3_prefix(bucket: str, prefix: str, dry_run: bool) -> None:
    if dry_run:
        print(f"[DRY RUN] Would empty s3://{bucket}/{prefix}")
        return
    paginator = s3.get_paginator("list_objects_v2")
    count = 0
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        objects = [{"Key": obj["Key"]} for obj in page.get("Contents", [])]
        if objects:
            s3.delete_objects(Bucket=bucket, Delete={"Objects": objects})
            count += len(objects)
    print(f"Deleted {count} objects from s3://{bucket}/{prefix}")


def main():
    parser = argparse.ArgumentParser(description="Cleanup insurance-risk-scorer AWS resources")
    parser.add_argument("--dry-run", action="store_true", help="Print actions without executing")
    parser.add_argument("--delete-s3", action="store_true", help="Also empty S3 data prefix")
    args = parser.parse_args()

    if args.dry_run:
        print("=== DRY RUN — no resources will be deleted ===\n")

    api_config = load_api_config()
    api_id = api_config.get("api_id", "")

    print("=== Insurance Risk Scorer — Cleanup ===\n")

    # 1. API Gateway
    if api_id:
        safe_delete(f"API Gateway {api_id}", apigw.delete_rest_api, args.dry_run, restApiId=api_id)
    else:
        # Try to find by name
        apis = apigw.get_rest_apis().get("items", [])
        matches = [a for a in apis if a["name"] == "InsuranceRiskScorerAPI"]
        for api in matches:
            safe_delete(f"API Gateway {api['id']}", apigw.delete_rest_api, args.dry_run, restApiId=api["id"])

    # 2. Lambda
    safe_delete(f"Lambda {FUNCTION_NAME}", lambda_client.delete_function, args.dry_run, FunctionName=FUNCTION_NAME)

    # 3. SageMaker endpoint (most important — stops hourly billing)
    model_name = get_sm_model_name(ENDPOINT_NAME)
    safe_delete(
        f"SageMaker endpoint {ENDPOINT_NAME}",
        sm.delete_endpoint,
        args.dry_run,
        EndpointName=ENDPOINT_NAME,
    )

    # 4. Endpoint config
    safe_delete(
        f"SageMaker endpoint config {ENDPOINT_NAME}",
        sm.delete_endpoint_config,
        args.dry_run,
        EndpointConfigName=ENDPOINT_NAME,
    )

    # 5. Model
    safe_delete(
        f"SageMaker model {model_name}",
        sm.delete_model,
        args.dry_run,
        ModelName=model_name,
    )

    # 6. CloudWatch alarms
    alarm_names = [
        f"{ENDPOINT_NAME}-InvocationErrors",
        f"{ENDPOINT_NAME}-HighLatency",
    ]
    safe_delete(
        f"CloudWatch alarms {alarm_names}",
        cw.delete_alarms,
        args.dry_run,
        AlarmNames=alarm_names,
    )

    # 7. S3 (optional)
    if args.delete_s3:
        delete_s3_prefix(BUCKET, "porto-seguro/", args.dry_run)
        delete_s3_prefix(BUCKET, "data-capture/", args.dry_run)
        delete_s3_prefix(BUCKET, "model-monitor/", args.dry_run)

    print("\nCleanup complete. Verify in AWS Console that the endpoint is gone.")


if __name__ == "__main__":
    main()
