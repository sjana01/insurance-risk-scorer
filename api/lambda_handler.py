"""
AWS Lambda handler: API Gateway → SageMaker endpoint.

Environment variables (set at deploy time):
    ENDPOINT_NAME  — SageMaker endpoint name (e.g. insurance-risk-scorer)
    AWS_REGION     — AWS region (e.g. us-east-1)

API contract:
    POST /predict
    Content-Type: application/json
    Body: {"ps_ind_01": 2, "ps_reg_01": 0.7, ...}

    Response 200:
    {"claim_probability": 0.043, "risk_label": "LOW"}

    Response 400: invalid JSON
    Response 500: SageMaker invocation failure
"""

import json
import logging
import os

import boto3

logger = logging.getLogger()
logger.setLevel(logging.INFO)

ENDPOINT_NAME = os.environ["ENDPOINT_NAME"]
REGION = os.environ.get("AWS_REGION", "us-east-1")

_runtime = None  # lazy-init so cold starts don't fail on missing creds during tests


def _get_runtime():
    global _runtime
    if _runtime is None:
        _runtime = boto3.client("sagemaker-runtime", region_name=REGION)
    return _runtime


def _cors_headers() -> dict:
    return {
        "Content-Type": "application/json",
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "POST,OPTIONS",
        "Access-Control-Allow-Headers": "Content-Type",
    }


def handler(event: dict, context) -> dict:
    """Lambda entry point."""
    # Handle CORS preflight
    if event.get("httpMethod") == "OPTIONS":
        return {"statusCode": 200, "headers": _cors_headers(), "body": ""}

    try:
        body_raw = event.get("body") or "{}"
        body = json.loads(body_raw)
    except (json.JSONDecodeError, TypeError) as e:
        logger.warning(f"Bad request — invalid JSON: {e}")
        return {
            "statusCode": 400,
            "headers": _cors_headers(),
            "body": json.dumps({"error": "Invalid JSON body"}),
        }

    logger.info(f"Invoking endpoint={ENDPOINT_NAME} with {len(body)} features")

    try:
        response = _get_runtime().invoke_endpoint(
            EndpointName=ENDPOINT_NAME,
            ContentType="application/json",
            Accept="application/json",
            Body=json.dumps(body),
        )
        result = json.loads(response["Body"].read())
    except Exception as e:
        logger.error(f"SageMaker invocation error: {e}")
        return {
            "statusCode": 502,
            "headers": _cors_headers(),
            "body": json.dumps({"error": "Model inference failed", "detail": str(e)}),
        }

    return {
        "statusCode": 200,
        "headers": _cors_headers(),
        "body": json.dumps(result),
    }
