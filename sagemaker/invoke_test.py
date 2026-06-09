"""
Smoke test: invoke the live SageMaker endpoint with a synthetic sample row
and verify the response structure.

Usage:
    python sagemaker/invoke_test.py
    python sagemaker/invoke_test.py --endpoint my-endpoint-name
"""

import argparse
import json
import os
import sys

import boto3

ENDPOINT_NAME = os.environ.get("ENDPOINT_NAME", "insurance-risk-scorer")
REGION = os.environ.get("AWS_REGION", "us-east-1")

# A plausible sample row from the Porto Seguro dataset (all _calc_ cols omitted)
SAMPLE_PAYLOAD = {
    "ps_ind_01": 2,
    "ps_ind_02_cat": 2,
    "ps_ind_03": 5,
    "ps_ind_04_cat": 0,
    "ps_ind_05_cat": 0,
    "ps_ind_06_bin": 0,
    "ps_ind_07_bin": 1,
    "ps_ind_08_bin": 0,
    "ps_ind_09_bin": 0,
    "ps_ind_10_bin": 0,
    "ps_ind_11_bin": 0,
    "ps_ind_12_bin": 0,
    "ps_ind_13_bin": 0,
    "ps_ind_14": 0,
    "ps_ind_15": 11,
    "ps_ind_16_bin": 1,
    "ps_ind_17_bin": 0,
    "ps_ind_18_bin": 0,
    "ps_reg_01": 0.7,
    "ps_reg_02": 0.2,
    "ps_reg_03": 0.718,
    "ps_car_01_cat": 10,
    "ps_car_02_cat": 1,
    "ps_car_03_cat": -1,
    "ps_car_04_cat": 0,
    "ps_car_05_cat": 1,
    "ps_car_06_cat": 11,
    "ps_car_07_cat": 1,
    "ps_car_08_cat": 1,
    "ps_car_09_cat": 2,
    "ps_car_10_cat": 1,
    "ps_car_11_cat": 104,
    "ps_car_11": 2,
    "ps_car_12": 0.3795,
    "ps_car_13": 0.664,
    "ps_car_14": 0.4507,
    "ps_car_15": 3.606,
    "missing_count": 2,
}


def invoke(endpoint_name: str, payload: dict, region: str) -> dict:
    client = boto3.client("sagemaker-runtime", region_name=region)
    response = client.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType="application/json",
        Body=json.dumps(payload),
    )
    return json.loads(response["Body"].read())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", default=ENDPOINT_NAME)
    parser.add_argument("--region", default=REGION)
    args = parser.parse_args()

    print(f"Invoking endpoint: {args.endpoint}")
    print(f"Payload keys:      {list(SAMPLE_PAYLOAD.keys())[:5]} ...")

    result = invoke(args.endpoint, SAMPLE_PAYLOAD, args.region)
    print(f"\nResponse: {json.dumps(result, indent=2)}")

    # Assertions
    assert "claim_probability" in result, "Missing claim_probability in response"
    assert "risk_label" in result, "Missing risk_label in response"
    assert 0.0 <= result["claim_probability"] <= 1.0, "Probability out of [0,1] range"
    assert result["risk_label"] in {"HIGH", "LOW"}, "Unexpected risk_label value"

    print("\nAll assertions passed. Endpoint is healthy.")


if __name__ == "__main__":
    main()
