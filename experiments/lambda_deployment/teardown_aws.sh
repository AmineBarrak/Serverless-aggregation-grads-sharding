#!/bin/bash
# ============================================================
# Teardown AWS resources created by setup_aws.sh
# ============================================================

set -e

REGION="us-east-1"
BUCKET_NAME="grads-sharding-exp-$(aws sts get-caller-identity --query Account --output text)"
ROLE_NAME="grads-sharding-lambda-role"
FUNCTION_PREFIX="grads-sharding-agg"

echo "============================================================"
echo "  GradsSharding Lambda Validation - TEARDOWN"
echo "============================================================"
echo "  This will delete all experiment resources."
echo "  Press Ctrl+C within 5 seconds to cancel..."
sleep 5

# Delete Lambda functions
echo ""
echo "[1/3] Deleting Lambda functions..."
for model in resnet18 vgg16 gpt2_medium gpt2_large; do
    func_name="${FUNCTION_PREFIX}-${model}"
    if aws lambda get-function --function-name "$func_name" 2>/dev/null; then
        aws lambda delete-function --function-name "$func_name"
        echo "  Deleted: $func_name"
    fi
done

# Empty and delete S3 bucket
echo ""
echo "[2/3] Emptying and deleting S3 bucket..."
if aws s3api head-bucket --bucket "$BUCKET_NAME" 2>/dev/null; then
    aws s3 rm "s3://$BUCKET_NAME" --recursive --quiet
    aws s3 rb "s3://$BUCKET_NAME"
    echo "  Deleted: $BUCKET_NAME"
else
    echo "  Bucket not found, skipping."
fi

# Delete IAM role
echo ""
echo "[3/3] Deleting IAM role..."
if aws iam get-role --role-name "$ROLE_NAME" 2>/dev/null; then
    # Detach policies first
    aws iam detach-role-policy --role-name "$ROLE_NAME" \
        --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole 2>/dev/null || true
    aws iam delete-role-policy --role-name "$ROLE_NAME" \
        --policy-name "S3GradientAccess" 2>/dev/null || true
    aws iam delete-role --role-name "$ROLE_NAME"
    echo "  Deleted: $ROLE_NAME"
else
    echo "  Role not found, skipping."
fi

echo ""
echo "============================================================"
echo "  TEARDOWN COMPLETE - All resources deleted"
echo "============================================================"
