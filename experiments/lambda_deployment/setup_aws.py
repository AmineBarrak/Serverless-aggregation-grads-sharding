#!/usr/bin/env python3
"""
AWS Infrastructure Setup for Lambda Validation Experiment.
Works on Windows, macOS, and Linux.

Creates: S3 bucket, IAM role, 4 Lambda functions.

Usage:
    python setup_aws.py
"""

import json
import time
import sys
import os
import zipfile
import tempfile

try:
    import boto3
    from botocore.exceptions import ClientError
except ImportError:
    print("ERROR: boto3 not installed. Run: pip install boto3")
    sys.exit(1)

REGION = "us-east-1"
ACCOUNT_ID = "961341528585"
BUCKET_NAME = f"grads-sharding-exp-{ACCOUNT_ID}"
ROLE_NAME = "grads-sharding-lambda-role"
FUNCTION_PREFIX = "grads-sharding-agg"

# Lambda memory & timeout per model
FUNC_CONFIGS = {
    "resnet18":    {"memory": 512,  "timeout": 120},
    "vgg16":       {"memory": 3008, "timeout": 900},
    "gpt2_medium": {"memory": 2048, "timeout": 600},
    "gpt2_large":  {"memory": 3008, "timeout": 900},
}

# AWS-provided layer with numpy (AWSSDKPandas includes numpy+pandas)
NUMPY_LAYER = f"arn:aws:lambda:{REGION}:336392948345:layer:AWSSDKPandas-Python312:16"


def create_s3_bucket(s3_client):
    """Create S3 bucket if it doesn't exist."""
    print(f"\n[Step 1/4] Creating S3 bucket: {BUCKET_NAME}")
    try:
        s3_client.head_bucket(Bucket=BUCKET_NAME)
        print(f"  Already exists, skipping.")
    except ClientError:
        # us-east-1 doesn't need LocationConstraint
        if REGION == "us-east-1":
            s3_client.create_bucket(Bucket=BUCKET_NAME)
        else:
            s3_client.create_bucket(
                Bucket=BUCKET_NAME,
                CreateBucketConfiguration={"LocationConstraint": REGION}
            )
        print(f"  Created.")


def create_iam_role(iam_client):
    """Create IAM role for Lambda with S3 access."""
    print(f"\n[Step 2/4] Creating IAM role: {ROLE_NAME}")

    trust_policy = {
        "Version": "2012-10-17",
        "Statement": [{
            "Effect": "Allow",
            "Principal": {"Service": "lambda.amazonaws.com"},
            "Action": "sts:AssumeRole"
        }]
    }

    try:
        response = iam_client.get_role(RoleName=ROLE_NAME)
        role_arn = response['Role']['Arn']
        print(f"  Already exists: {role_arn}")
    except ClientError:
        response = iam_client.create_role(
            RoleName=ROLE_NAME,
            AssumeRolePolicyDocument=json.dumps(trust_policy),
            Description="Role for GradsSharding Lambda aggregation experiment"
        )
        role_arn = response['Role']['Arn']
        print(f"  Created: {role_arn}")

    # Attach basic Lambda execution policy
    print(f"  Attaching policies...")
    try:
        iam_client.attach_role_policy(
            RoleName=ROLE_NAME,
            PolicyArn="arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
        )
    except ClientError:
        pass

    # S3 access policy
    s3_policy = {
        "Version": "2012-10-17",
        "Statement": [{
            "Effect": "Allow",
            "Action": ["s3:GetObject", "s3:PutObject", "s3:ListBucket"],
            "Resource": [
                f"arn:aws:s3:::{BUCKET_NAME}",
                f"arn:aws:s3:::{BUCKET_NAME}/*"
            ]
        }]
    }
    iam_client.put_role_policy(
        RoleName=ROLE_NAME,
        PolicyName="S3GradientAccess",
        PolicyDocument=json.dumps(s3_policy)
    )
    print(f"  Policies attached.")

    # Wait for role propagation
    print(f"  Waiting 10s for IAM role propagation...")
    time.sleep(10)

    return role_arn


def create_deployment_package():
    """Create a zip file containing the Lambda function code."""
    print(f"\n[Step 3/4] Creating deployment package...")

    lambda_code_path = os.path.join(os.path.dirname(__file__), "lambda_function.py")
    if not os.path.exists(lambda_code_path):
        print(f"  ERROR: lambda_function.py not found at {lambda_code_path}")
        sys.exit(1)

    zip_path = os.path.join(tempfile.gettempdir(), "grads_sharding_lambda.zip")
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.write(lambda_code_path, "lambda_function.py")

    size_kb = os.path.getsize(zip_path) / 1024
    print(f"  Created: {zip_path} ({size_kb:.1f} KB)")
    return zip_path


def create_lambda_functions(lambda_client, role_arn, zip_path):
    """Create or update Lambda functions for each model."""
    print(f"\n[Step 4/4] Creating Lambda functions...")

    with open(zip_path, 'rb') as f:
        zip_bytes = f.read()

    for model, config in FUNC_CONFIGS.items():
        func_name = f"{FUNCTION_PREFIX}-{model}"
        mem = config['memory']
        timeout = config['timeout']

        print(f"  {func_name} (memory={mem}MB, timeout={timeout}s)...", end=" ")

        try:
            # Check if function exists
            lambda_client.get_function(FunctionName=func_name)

            # Wait until function is not in update state
            while True:
                resp = lambda_client.get_function(FunctionName=func_name)
                state = resp['Configuration'].get('LastUpdateStatus', 'Successful')
                if state in ('Successful', 'Failed'):
                    break
                print(f"waiting ({state})...", end=" ", flush=True)
                time.sleep(3)

            # Update existing
            lambda_client.update_function_code(
                FunctionName=func_name,
                ZipFile=zip_bytes,
            )
            # Wait for code update to finish
            while True:
                resp = lambda_client.get_function(FunctionName=func_name)
                state = resp['Configuration'].get('LastUpdateStatus', 'Successful')
                if state in ('Successful', 'Failed'):
                    break
                time.sleep(3)

            lambda_client.update_function_configuration(
                FunctionName=func_name,
                MemorySize=mem,
                Timeout=timeout,
                Layers=[NUMPY_LAYER],
            )
            print("updated.")

        except ClientError as e:
            if 'ResourceNotFoundException' in str(e):
                # Create new
                lambda_client.create_function(
                    FunctionName=func_name,
                    Runtime='python3.12',
                    Role=role_arn,
                    Handler='lambda_function.lambda_handler',
                    Code={'ZipFile': zip_bytes},
                    MemorySize=mem,
                    Timeout=timeout,
                    Layers=[NUMPY_LAYER],
                )
                print("created.")
            else:
                raise

    # Clean up zip
    os.remove(zip_path)


def main():
    print("=" * 60)
    print("  GradsSharding Lambda Validation - AWS Setup")
    print("=" * 60)
    print(f"  Region:   {REGION}")
    print(f"  Account:  {ACCOUNT_ID}")
    print(f"  Bucket:   {BUCKET_NAME}")
    print("=" * 60)

    s3_client = boto3.client('s3', region_name=REGION)
    iam_client = boto3.client('iam', region_name=REGION)
    lambda_client = boto3.client('lambda', region_name=REGION)

    create_s3_bucket(s3_client)
    role_arn = create_iam_role(iam_client)
    zip_path = create_deployment_package()
    create_lambda_functions(lambda_client, role_arn, zip_path)

    print(f"\n{'=' * 60}")
    print(f"  SETUP COMPLETE")
    print(f"{'=' * 60}")
    print(f"  Bucket:    {BUCKET_NAME}")
    print(f"  Functions: {FUNCTION_PREFIX}-{{resnet18,vgg16,gpt2_medium,gpt2_large}}")
    print(f"")
    print(f"  Next steps:")
    print(f"    1. python generate_and_upload_gradients.py --bucket {BUCKET_NAME}")
    print(f"    2. python run_lambda_experiment.py --bucket {BUCKET_NAME}")
    print(f"    3. python teardown_aws.py   (when done)")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
