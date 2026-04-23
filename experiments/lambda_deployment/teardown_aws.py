#!/usr/bin/env python3
"""
Teardown all AWS resources created by setup_aws.py.
Works on Windows, macOS, and Linux.

Usage:
    python teardown_aws.py
"""

import time
import sys

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
MODELS = ["resnet18", "vgg16", "gpt2_medium", "gpt2_large"]


def main():
    print("=" * 60)
    print("  GradsSharding Lambda Validation - TEARDOWN")
    print("=" * 60)
    print("  This will DELETE all experiment resources.")
    print("  Press Ctrl+C within 5 seconds to cancel...")
    print("=" * 60)
    time.sleep(5)

    lambda_client = boto3.client('lambda', region_name=REGION)
    s3_client = boto3.client('s3', region_name=REGION)
    s3_resource = boto3.resource('s3', region_name=REGION)
    iam_client = boto3.client('iam', region_name=REGION)

    # 1. Delete Lambda functions
    print("\n[1/3] Deleting Lambda functions...")
    for model in MODELS:
        func_name = f"{FUNCTION_PREFIX}-{model}"
        try:
            lambda_client.delete_function(FunctionName=func_name)
            print(f"  Deleted: {func_name}")
        except ClientError:
            print(f"  Not found: {func_name}")

    # 2. Empty and delete S3 bucket
    print("\n[2/3] Emptying and deleting S3 bucket...")
    try:
        bucket = s3_resource.Bucket(BUCKET_NAME)
        bucket.objects.all().delete()
        bucket.delete()
        print(f"  Deleted: {BUCKET_NAME}")
    except ClientError:
        print(f"  Not found: {BUCKET_NAME}")

    # 3. Delete IAM role
    print("\n[3/3] Deleting IAM role...")
    try:
        # Detach managed policies
        try:
            iam_client.detach_role_policy(
                RoleName=ROLE_NAME,
                PolicyArn="arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
            )
        except ClientError:
            pass

        # Delete inline policies
        try:
            iam_client.delete_role_policy(
                RoleName=ROLE_NAME,
                PolicyName="S3GradientAccess"
            )
        except ClientError:
            pass

        iam_client.delete_role(RoleName=ROLE_NAME)
        print(f"  Deleted: {ROLE_NAME}")
    except ClientError:
        print(f"  Not found: {ROLE_NAME}")

    print(f"\n{'=' * 60}")
    print(f"  TEARDOWN COMPLETE - All resources deleted")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
