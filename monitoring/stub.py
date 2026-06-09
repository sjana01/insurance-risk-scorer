"""
Monitoring stub for the Insurance Risk Scorer.

Sets up:
  1. CloudWatch alarm — endpoint invocation errors
  2. Data capture config (attach at endpoint creation time)
  3. SageMaker Model Monitor skeleton (commented — requires baseline job)

Run after the endpoint is live:
    python monitoring/stub.py
"""

import json
import os

import boto3

REGION = os.environ.get("AWS_REGION", "us-east-1")
BUCKET = os.environ.get("S3_BUCKET", "insurance-risk-scorer-ACCOUNT_ID")
ENDPOINT_NAME = os.environ.get("ENDPOINT_NAME", "insurance-risk-scorer")
ALARM_EMAIL = os.environ.get("ALARM_EMAIL", "")  # optional SNS email

cloudwatch = boto3.client("cloudwatch", region_name=REGION)
sns = boto3.client("sns", region_name=REGION)
sm_client = boto3.client("sagemaker", region_name=REGION)


# ---------------------------------------------------------------------------
# 1. CloudWatch Alarm — Invocation Errors
# ---------------------------------------------------------------------------

def create_error_alarm(sns_topic_arn: str = "") -> None:
    """Alert when ModelError count ≥ 5 within a 5-minute window."""
    alarm_kwargs = dict(
        AlarmName=f"{ENDPOINT_NAME}-InvocationErrors",
        AlarmDescription="SageMaker endpoint invocation errors exceed threshold",
        MetricName="ModelError",
        Namespace="AWS/SageMaker",
        Dimensions=[{"Name": "EndpointName", "Value": ENDPOINT_NAME}],
        Period=300,
        EvaluationPeriods=1,
        Threshold=5,
        Statistic="Sum",
        ComparisonOperator="GreaterThanOrEqualToThreshold",
        TreatMissingData="notBreaching",
    )
    if sns_topic_arn:
        alarm_kwargs["AlarmActions"] = [sns_topic_arn]
        alarm_kwargs["OKActions"] = [sns_topic_arn]

    cloudwatch.put_metric_alarm(**alarm_kwargs)
    print(f"CloudWatch alarm created: {ENDPOINT_NAME}-InvocationErrors")


def create_latency_alarm(threshold_ms: float = 2000.0) -> None:
    """Alert when p99 inference latency exceeds threshold_ms."""
    cloudwatch.put_metric_alarm(
        AlarmName=f"{ENDPOINT_NAME}-HighLatency",
        AlarmDescription=f"p99 latency > {threshold_ms}ms",
        MetricName="ModelLatency",
        Namespace="AWS/SageMaker",
        Dimensions=[
            {"Name": "EndpointName", "Value": ENDPOINT_NAME},
            {"Name": "VariantName", "Value": "AllTraffic"},
        ],
        Period=60,
        EvaluationPeriods=3,
        Threshold=threshold_ms * 1_000,  # SageMaker reports in microseconds
        Statistic="p99",
        ExtendedStatistic="p99",
        ComparisonOperator="GreaterThanThreshold",
        TreatMissingData="notBreaching",
    )
    print(f"CloudWatch alarm created: {ENDPOINT_NAME}-HighLatency")


# ---------------------------------------------------------------------------
# 2. Data Capture Config (to attach at deploy time)
# ---------------------------------------------------------------------------

def get_data_capture_config() -> dict:
    """
    Returns the DataCaptureConfig dict to pass to sagemaker.deploy().

    Usage in sagemaker/deploy.py:
        from sagemaker.model_monitor import DataCaptureConfig
        data_capture_config = DataCaptureConfig(
            enable_capture=True,
            sampling_percentage=20,
            destination_s3_uri=f"s3://{BUCKET}/data-capture/",
            capture_options=["REQUEST", "RESPONSE"],
        )
        predictor = estimator.deploy(..., data_capture_config=data_capture_config)
    """
    return {
        "enable_capture": True,
        "sampling_percentage": 20,
        "destination_s3_uri": f"s3://{BUCKET}/data-capture/{ENDPOINT_NAME}/",
        "capture_options": ["REQUEST", "RESPONSE"],
    }


# ---------------------------------------------------------------------------
# 3. SageMaker Model Monitor Skeleton (uncomment to activate)
# ---------------------------------------------------------------------------

def setup_model_monitor_baseline() -> None:
    """
    TODO: Run a baselining job once you have ~1000+ captured requests.

    Steps:
    1. Wait for data capture to populate s3://{BUCKET}/data-capture/...
    2. Run baselining job (creates statistics.json + constraints.json)
    3. Create monitoring schedule (hourly or daily)

    Uncomment and run after the endpoint has served real traffic.
    """
    # from sagemaker.model_monitor import DefaultModelMonitor, CronExpressionGenerator

    # monitor = DefaultModelMonitor(
    #     role=os.environ["SAGEMAKER_ROLE_ARN"],
    #     instance_count=1,
    #     instance_type="ml.m5.xlarge",
    #     volume_size_in_gb=20,
    #     max_runtime_in_seconds=3600,
    # )

    # Baseline job — fit on captured training data distribution
    # monitor.suggest_baseline(
    #     baseline_dataset=f"s3://{BUCKET}/porto-seguro/data/train.csv",
    #     dataset_format=sagemaker.model_monitor.DatasetFormat.csv(header=True),
    #     output_s3_uri=f"s3://{BUCKET}/model-monitor/baseline/",
    #     wait=True,
    # )

    # Monitoring schedule — check for drift hourly
    # monitor.create_monitoring_schedule(
    #     monitor_schedule_name=f"{ENDPOINT_NAME}-monitor",
    #     endpoint_input=ENDPOINT_NAME,
    #     output_s3_uri=f"s3://{BUCKET}/model-monitor/reports/",
    #     statistics=monitor.baseline_statistics(),
    #     constraints=monitor.suggested_constraints(),
    #     schedule_cron_expression=CronExpressionGenerator.hourly(),
    # )
    # print(f"Model Monitor schedule created: {ENDPOINT_NAME}-monitor")
    pass


# ---------------------------------------------------------------------------
# 4. Simple statistical drift check (no SageMaker dependency)
# ---------------------------------------------------------------------------

def check_input_drift(recent_df, baseline_df, threshold: float = 0.1) -> list:
    """
    Compares mean of each numeric feature between recent and baseline.
    Returns list of features drifted beyond threshold (relative change).
    """
    import numpy as np

    drifted = []
    for col in baseline_df.select_dtypes("number").columns:
        if col not in recent_df.columns:
            continue
        base_mean = baseline_df[col].mean()
        recent_mean = recent_df[col].mean()
        if base_mean == 0:
            continue
        rel_change = abs(recent_mean - base_mean) / abs(base_mean)
        if rel_change > threshold:
            drifted.append({"feature": col, "relative_change": round(rel_change, 4)})

    if drifted:
        print(f"WARNING: {len(drifted)} features drifted > {threshold*100:.0f}%:")
        for d in drifted:
            print(f"  {d['feature']}: {d['relative_change']:.2%}")
    else:
        print("No significant drift detected.")

    return drifted


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"Setting up monitoring for endpoint: {ENDPOINT_NAME}\n")

    # Optional: create SNS topic for email alerts
    topic_arn = ""
    if ALARM_EMAIL:
        topic = sns.create_topic(Name=f"{ENDPOINT_NAME}-alerts")
        topic_arn = topic["TopicArn"]
        sns.subscribe(TopicArn=topic_arn, Protocol="email", Endpoint=ALARM_EMAIL)
        print(f"SNS topic created: {topic_arn}")
        print(f"Check {ALARM_EMAIL} to confirm SNS subscription.")

    create_error_alarm(sns_topic_arn=topic_arn)
    create_latency_alarm()

    data_capture_cfg = get_data_capture_config()
    print(f"\nData capture config (add to deploy.py):")
    print(json.dumps(data_capture_cfg, indent=2))

    print("\nModel Monitor baseline: uncomment setup_model_monitor_baseline() after capturing traffic.")
    print("\nMonitoring setup complete.")


if __name__ == "__main__":
    main()
