"""
SageMaker training job launcher and endpoint deployer.

Usage:
    # Launch training job only:
    python sagemaker/deploy.py --train-only

    # Deploy existing training job output as endpoint:
    python sagemaker/deploy.py --deploy --job-name insurance-risk-scorer-YYYYMMDD-HHMMSS

    # Train then immediately deploy:
    python sagemaker/deploy.py --train-and-deploy

Required env vars (or edit CONFIG below):
    SAGEMAKER_ROLE_ARN   — ARN of the SageMaker execution IAM role
    S3_BUCKET            — S3 bucket name (no s3:// prefix)
    AWS_REGION           — e.g. us-east-1
"""

import argparse
import os
import time

import boto3
import sagemaker
from sagemaker import image_uris
from sagemaker.estimator import Estimator

# --- Configuration -----------------------------------------------------------
CONFIG = {
    "role": os.environ.get("SAGEMAKER_ROLE_ARN", "arn:aws:iam::ACCOUNT_ID:role/SageMakerExecutionRole"),
    "bucket": os.environ.get("S3_BUCKET", "insurance-risk-scorer-ACCOUNT_ID"),
    "region": os.environ.get("AWS_REGION", "us-east-1"),
    "prefix": "porto-seguro",
    "endpoint_name": "insurance-risk-scorer",
    "train_instance": "ml.m5.xlarge",
    "deploy_instance": "ml.m5.large",
}

HYPERPARAMETERS = {
    "max-depth": 6,
    "eta": 0.05,
    "subsample": 0.8,
    "colsample-bytree": 0.8,
    "min-child-weight": 100,
    "num-boost-round": 500,
    "early-stopping-rounds": 50,
    "n-folds": 5,
    "seed": 42,
}
# ----------------------------------------------------------------------------


def get_session():
    boto_session = boto3.Session(region_name=CONFIG["region"])
    return sagemaker.Session(boto_session=boto_session)


def get_xgb_image(region: str) -> str:
    return image_uris.retrieve(
        framework="xgboost",
        region=region,
        version="1.7-1",
        image_scope="training",
    )


def launch_training(sm_session: sagemaker.Session) -> str:
    """Start a SageMaker training job and return the job name."""
    s3_train_uri = f"s3://{CONFIG['bucket']}/{CONFIG['prefix']}/data/train.csv"
    job_name = f"insurance-risk-scorer-{time.strftime('%Y%m%d-%H%M%S')}"

    print(f"Training data: {s3_train_uri}")
    print(f"Job name:      {job_name}")

    estimator = Estimator(
        image_uri=get_xgb_image(CONFIG["region"]),
        role=CONFIG["role"],
        instance_count=1,
        instance_type=CONFIG["train_instance"],
        output_path=f"s3://{CONFIG['bucket']}/{CONFIG['prefix']}/output/",
        sagemaker_session=sm_session,
        entry_point="train.py",
        source_dir=os.path.dirname(__file__),
        hyperparameters=HYPERPARAMETERS,
        base_job_name="insurance-risk-scorer",
        tags=[{"Key": "Project", "Value": "insurance-risk-scorer"}],
    )

    estimator.fit(
        {"train": s3_train_uri},
        job_name=job_name,
        wait=True,
        logs="All",
    )

    print(f"\nTraining complete. Job: {job_name}")
    return job_name


def deploy_endpoint(sm_session: sagemaker.Session, job_name: str) -> str:
    """Create a real-time endpoint from a completed training job."""
    from sagemaker.estimator import Estimator
    from sagemaker.serializers import JSONSerializer
    from sagemaker.deserializers import JSONDeserializer

    estimator = Estimator.attach(job_name, sagemaker_session=sm_session)

    print(f"Deploying endpoint: {CONFIG['endpoint_name']}")
    print(f"Instance type:      {CONFIG['deploy_instance']}")

    predictor = estimator.deploy(
        initial_instance_count=1,
        instance_type=CONFIG["deploy_instance"],
        endpoint_name=CONFIG["endpoint_name"],
        serializer=JSONSerializer(),
        deserializer=JSONDeserializer(),
    )

    endpoint_url = (
        f"https://runtime.sagemaker.{CONFIG['region']}.amazonaws.com"
        f"/endpoints/{CONFIG['endpoint_name']}/invocations"
    )
    print(f"\nEndpoint live: {CONFIG['endpoint_name']}")
    print(f"Invocation URL: {endpoint_url}")
    print("\nWARNING: Remember to run infra/cleanup.py when done — endpoints bill by the hour.")
    return CONFIG["endpoint_name"]


def parse_args():
    parser = argparse.ArgumentParser(description="SageMaker train/deploy controller")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--train-only", action="store_true")
    group.add_argument("--deploy", action="store_true")
    group.add_argument("--train-and-deploy", action="store_true")
    parser.add_argument("--job-name", type=str, help="Existing job name (required with --deploy)")
    return parser.parse_args()


def main():
    args = parse_args()
    sm_session = get_session()

    if args.train_only:
        launch_training(sm_session)

    elif args.deploy:
        if not args.job_name:
            raise ValueError("--job-name is required with --deploy")
        deploy_endpoint(sm_session, args.job_name)

    elif args.train_and_deploy:
        job_name = launch_training(sm_session)
        deploy_endpoint(sm_session, job_name)


if __name__ == "__main__":
    main()
