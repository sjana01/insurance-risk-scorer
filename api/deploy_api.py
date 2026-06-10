"""
Deploy the REST API layer: Lambda function + API Gateway.

What this script does (idempotent):
  1. Packages api/lambda_handler.py into a zip
  2. Creates (or updates) a Lambda function
  3. Creates an API Gateway REST API with POST /predict
  4. Wires Lambda proxy integration
  5. Deploys to stage 'prod'
  6. Prints the public invoke URL

Usage:
    python api/deploy_api.py

Required env vars:
    SAGEMAKER_ROLE_ARN   — used to locate the Lambda execution role ARN
    LAMBDA_ROLE_ARN      — ARN of the Lambda execution IAM role
    S3_BUCKET            — for config persistence (optional)
    AWS_REGION           — e.g. us-east-1
    ENDPOINT_NAME        — SageMaker endpoint name (default: insurance-risk-scorer)
"""

import io
import json
import os
import sys
import time
import zipfile

import boto3
from botocore.exceptions import ClientError

REGION = os.environ.get("AWS_REGION", "us-east-1")
LAMBDA_ROLE_ARN = os.environ.get("LAMBDA_ROLE_ARN", "arn:aws:iam::ACCOUNT_ID:role/LambdaExecutionRole")
ENDPOINT_NAME = os.environ.get("ENDPOINT_NAME", "insurance-risk-scorer")
FUNCTION_NAME = "insurance-risk-scorer-api"
API_NAME = "InsuranceRiskScorerAPI"
STAGE_NAME = "prod"

lambda_client = boto3.client("lambda", region_name=REGION)
apigw_client = boto3.client("apigateway", region_name=REGION)
iam_client = boto3.client("iam", region_name=REGION)


def build_lambda_zip() -> bytes:
    """Bundle lambda_handler.py into an in-memory zip."""
    handler_path = os.path.join(os.path.dirname(__file__), "lambda_handler.py")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(handler_path, arcname="lambda_handler.py")
    return buf.getvalue()


def upsert_lambda(zip_bytes: bytes) -> str:
    """Create or update the Lambda function. Returns function ARN."""
    env_vars = {
        "ENDPOINT_NAME": ENDPOINT_NAME,
        "SM_REGION": REGION,
    }

    try:
        fn = lambda_client.get_function(FunctionName=FUNCTION_NAME)
        print(f"Updating existing Lambda: {FUNCTION_NAME}")
        lambda_client.update_function_code(FunctionName=FUNCTION_NAME, ZipFile=zip_bytes)
        lambda_client.update_function_configuration(
            FunctionName=FUNCTION_NAME,
            Environment={"Variables": env_vars},
        )
        return fn["Configuration"]["FunctionArn"]

    except ClientError as e:
        if e.response["Error"]["Code"] != "ResourceNotFoundException":
            raise

    print(f"Creating Lambda: {FUNCTION_NAME}")
    response = lambda_client.create_function(
        FunctionName=FUNCTION_NAME,
        Runtime="python3.10",
        Role=LAMBDA_ROLE_ARN,
        Handler="lambda_handler.handler",
        Code={"ZipFile": zip_bytes},
        Description="Insurance Risk Scorer — API Gateway proxy to SageMaker",
        Timeout=30,
        MemorySize=128,
        Environment={"Variables": env_vars},
        Tags={"Project": "insurance-risk-scorer"},
    )
    # Wait for Lambda to become Active
    waiter = lambda_client.get_waiter("function_active")
    waiter.wait(FunctionName=FUNCTION_NAME)
    return response["FunctionArn"]


def upsert_api(lambda_arn: str) -> str:
    """Create (or reuse) API Gateway REST API. Returns API ID."""
    # Check if API already exists
    apis = apigw_client.get_rest_apis()["items"]
    existing = [a for a in apis if a["name"] == API_NAME]
    if existing:
        api_id = existing[0]["id"]
        print(f"Reusing existing API Gateway: {api_id}")
        return api_id

    print(f"Creating API Gateway: {API_NAME}")
    api = apigw_client.create_rest_api(
        name=API_NAME,
        description="Insurance Risk Scorer REST API",
        endpointConfiguration={"types": ["REGIONAL"]},
    )
    return api["id"]


def wire_lambda_proxy(api_id: str, lambda_arn: str) -> None:
    """Create /predict resource + POST method with Lambda proxy integration."""
    account_id = boto3.client("sts").get_caller_identity()["Account"]
    region = REGION

    # Get root resource
    resources = apigw_client.get_resources(restApiId=api_id)["items"]
    root_id = next(r["id"] for r in resources if r["path"] == "/")

    # Create /predict resource (skip if exists)
    predict_resource = next(
        (r for r in resources if r.get("pathPart") == "predict"), None
    )
    if predict_resource is None:
        predict_resource = apigw_client.create_resource(
            restApiId=api_id,
            parentId=root_id,
            pathPart="predict",
        )
    resource_id = predict_resource["id"]

    # Create POST method (skip if exists)
    try:
        apigw_client.get_method(restApiId=api_id, resourceId=resource_id, httpMethod="POST")
        print("POST /predict already configured")
    except ClientError:
        apigw_client.put_method(
            restApiId=api_id,
            resourceId=resource_id,
            httpMethod="POST",
            authorizationType="NONE",
        )

        lambda_uri = (
            f"arn:aws:apigateway:{region}:lambda:path/2015-03-31"
            f"/functions/{lambda_arn}/invocations"
        )
        apigw_client.put_integration(
            restApiId=api_id,
            resourceId=resource_id,
            httpMethod="POST",
            type="AWS_PROXY",
            integrationHttpMethod="POST",
            uri=lambda_uri,
        )
        print("Created POST /predict with Lambda proxy integration")

    # Grant API Gateway permission to invoke Lambda
    statement_id = "apigateway-invoke"
    try:
        lambda_client.add_permission(
            FunctionName=FUNCTION_NAME,
            StatementId=statement_id,
            Action="lambda:InvokeFunction",
            Principal="apigateway.amazonaws.com",
            SourceArn=f"arn:aws:execute-api:{region}:{account_id}:{api_id}/*/POST/predict",
        )
    except ClientError as e:
        if "already exists" not in str(e):
            raise


def deploy_stage(api_id: str) -> str:
    """Deploy the API to prod stage and return the invoke URL."""
    apigw_client.create_deployment(restApiId=api_id, stageName=STAGE_NAME)
    return f"https://{api_id}.execute-api.{REGION}.amazonaws.com/{STAGE_NAME}/predict"


def main():
    print("=== Deploying Insurance Risk Scorer REST API ===\n")

    zip_bytes = build_lambda_zip()
    print(f"Lambda package size: {len(zip_bytes) / 1024:.1f} KB")

    lambda_arn = upsert_lambda(zip_bytes)
    print(f"Lambda ARN: {lambda_arn}")

    api_id = upsert_api(lambda_arn)
    wire_lambda_proxy(api_id, lambda_arn)
    invoke_url = deploy_stage(api_id)

    print(f"\n{'='*50}")
    print(f"REST API deployed successfully!")
    print(f"\nEndpoint URL:")
    print(f"  {invoke_url}")
    print(f"\nTest with curl:")
    print(f'  curl -X POST {invoke_url} \\')
    print(f'       -H "Content-Type: application/json" \\')
    print(f"       -d '{{\"ps_ind_01\": 2, \"ps_reg_01\": 0.7, \"missing_count\": 0}}'")
    print(f"\nSave these for cleanup:")
    print(f"  API_ID={api_id}")
    print(f"  FUNCTION_NAME={FUNCTION_NAME}")

    # Persist IDs for cleanup script
    config_path = os.path.join(os.path.dirname(__file__), ".api_config.json")
    with open(config_path, "w") as f:
        json.dump({"api_id": api_id, "function_name": FUNCTION_NAME, "invoke_url": invoke_url}, f, indent=2)
    print(f"\nConfig saved to {config_path}")


if __name__ == "__main__":
    main()
