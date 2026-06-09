# IAM Setup Guide

Complete these steps once before running any training or deployment scripts.

---

## Prerequisites

- AWS account with console access
- AWS CLI installed: `pip install awscli` or via conda
- Your AWS account ID (find it in AWS Console → top-right account menu)

---

## Step 1 — Create S3 Bucket

In the AWS Console → S3 → Create bucket:

- **Name:** `insurance-risk-scorer-{YOUR_ACCOUNT_ID}` (must be globally unique)
- **Region:** `us-east-1` (or your preferred region — keep consistent throughout)
- Block all public access: ✓ enabled
- Versioning: optional

---

## Step 2 — Create SageMaker Execution Role

AWS Console → IAM → Roles → Create role

**Trust policy** (select "Custom trust policy"):
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Service": "sagemaker.amazonaws.com"
      },
      "Action": "sts:AssumeRole"
    }
  ]
}
```

**Permissions to attach:**
- `AmazonSageMakerFullAccess`
- `AmazonS3FullAccess`

**Name:** `InsuranceRiskScorerSageMakerRole`

Copy the role ARN — you'll need it as `SAGEMAKER_ROLE_ARN`.

---

## Step 3 — Create Lambda Execution Role

AWS Console → IAM → Roles → Create role

**Trusted entity:** AWS service → Lambda

**Permissions to attach:**
- `AWSLambdaBasicExecutionRole`
- Add this inline policy (allows invoking only your SageMaker endpoint):

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": "sagemaker:InvokeEndpoint",
      "Resource": "arn:aws:sagemaker:*:*:endpoint/insurance-risk-scorer"
    }
  ]
}
```

**Name:** `InsuranceRiskScorerLambdaRole`

Copy the role ARN — you'll need it as `LAMBDA_ROLE_ARN`.

---

## Step 4 — Create IAM User for Local Development

AWS Console → IAM → Users → Create user

**Name:** `insurance-risk-dev`

**Permissions to attach (directly):**
- `AmazonSageMakerFullAccess`
- `AmazonS3FullAccess`
- `AWSLambdaFullAccess`
- `AmazonAPIGatewayAdministrator`

Create access key (for CLI):
- User → Security credentials → Create access key → CLI use

---

## Step 5 — Configure AWS CLI

```bash
aws configure
# AWS Access Key ID:     [paste from step 4]
# AWS Secret Access Key: [paste from step 4]
# Default region name:   us-east-1
# Default output format: json
```

Verify:
```bash
aws sts get-caller-identity
```

---

## Step 6 — Set Environment Variables

Add to your shell profile (`~/.zshrc`, `~/.bashrc`, or Windows environment):

```bash
export SAGEMAKER_ROLE_ARN="arn:aws:iam::ACCOUNT_ID:role/InsuranceRiskScorerSageMakerRole"
export LAMBDA_ROLE_ARN="arn:aws:iam::ACCOUNT_ID:role/InsuranceRiskScorerLambdaRole"
export S3_BUCKET="insurance-risk-scorer-ACCOUNT_ID"
export AWS_REGION="us-east-1"
export ENDPOINT_NAME="insurance-risk-scorer"
```

---

## Step 7 — Set Up Kaggle API

1. Log in to [kaggle.com](https://kaggle.com) → Account → Create New API Token
2. Download `kaggle.json`
3. Place it at `~/.kaggle/kaggle.json`
4. Set permissions (Linux/Mac): `chmod 600 ~/.kaggle/kaggle.json`
5. Accept the competition rules at:
   https://www.kaggle.com/c/porto-seguro-safe-driver-prediction → Rules → I Understand and Accept

Test:
```bash
kaggle competitions list
```

---

## Cost Guardrails

- SageMaker endpoints bill **per hour** while running. Always run `python infra/cleanup.py` after demos.
- Set a billing alert: AWS Console → Billing → Budgets → Create budget → $10/month threshold.
- The training job (ml.m5.xlarge, ~25 min) costs ~$0.10 one-time.
- The endpoint (ml.m5.large) costs ~$0.115/hr — **delete it when not in use**.

---

## Verification Checklist

- [ ] `aws sts get-caller-identity` returns your account ID
- [ ] `aws s3 ls s3://insurance-risk-scorer-{ACCOUNT_ID}/` returns empty (not error)
- [ ] `kaggle competitions list` returns results
- [ ] Environment variables are set (`echo $SAGEMAKER_ROLE_ARN`)
