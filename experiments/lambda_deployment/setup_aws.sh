#!/bin/bash
# ============================================================
# AWS Infrastructure Setup for Lambda Validation Experiment
# ============================================================
#
# This script creates:
#   1. An S3 bucket for gradient storage
#   2. An IAM role for the Lambda function
#   3. A Lambda deployment package with numpy layer
#   4. Four Lambda functions (one per model/memory config)
#
# Prerequisites:
#   - AWS CLI configured (aws configure)
#   - Sufficient IAM permissions to create roles, Lambda, S3
#
# Usage:
#   chmod +x setup_aws.sh
#   ./setup_aws.sh
# ============================================================

set -e

REGION="us-east-1"
BUCKET_NAME="grads-sharding-exp-$(aws sts get-caller-identity --query Account --output text)"
ROLE_NAME="grads-sharding-lambda-role"
FUNCTION_PREFIX="grads-sharding-agg"

echo "============================================================"
echo "  GradsSharding Lambda Validation - AWS Setup"
echo "============================================================"
echo "  Region:  $REGION"
echo "  Bucket:  $BUCKET_NAME"
echo "  Role:    $ROLE_NAME"
echo "============================================================"

# ---- Step 1: Create S3 Bucket ----
echo ""
echo "[Step 1/4] Creating S3 bucket..."
if aws s3api head-bucket --bucket "$BUCKET_NAME" 2>/dev/null; then
    echo "  Bucket $BUCKET_NAME already exists, skipping."
else
    aws s3 mb "s3://$BUCKET_NAME" --region "$REGION"
    echo "  Created bucket: $BUCKET_NAME"
fi

# ---- Step 2: Create IAM Role ----
echo ""
echo "[Step 2/4] Creating IAM role..."

TRUST_POLICY='{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Principal": {"Service": "lambda.amazonaws.com"},
    "Action": "sts:AssumeRole"
  }]
}'

if aws iam get-role --role-name "$ROLE_NAME" 2>/dev/null; then
    echo "  Role $ROLE_NAME already exists, skipping."
else
    aws iam create-role \
        --role-name "$ROLE_NAME" \
        --assume-role-policy-document "$TRUST_POLICY" \
        --description "Role for GradsSharding Lambda aggregation experiment" \
        > /dev/null
    echo "  Created role: $ROLE_NAME"
fi

# Attach policies
echo "  Attaching policies..."
aws iam attach-role-policy --role-name "$ROLE_NAME" \
    --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole 2>/dev/null || true

# S3 access policy
S3_POLICY='{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Action": ["s3:GetObject", "s3:PutObject", "s3:ListBucket"],
    "Resource": ["arn:aws:s3:::'$BUCKET_NAME'", "arn:aws:s3:::'$BUCKET_NAME'/*"]
  }]
}'

aws iam put-role-policy --role-name "$ROLE_NAME" \
    --policy-name "S3GradientAccess" \
    --policy-document "$S3_POLICY"
echo "  Policies attached."

# Wait for role propagation
echo "  Waiting 10s for role propagation..."
sleep 10

ROLE_ARN=$(aws iam get-role --role-name "$ROLE_NAME" --query 'Role.Arn' --output text)
echo "  Role ARN: $ROLE_ARN"

# ---- Step 3: Create Lambda Deployment Package ----
echo ""
echo "[Step 3/4] Creating Lambda deployment package..."

DEPLOY_DIR=$(mktemp -d)
cp lambda_function.py "$DEPLOY_DIR/"
cd "$DEPLOY_DIR"
zip -q deployment.zip lambda_function.py
cd -
echo "  Package created: $DEPLOY_DIR/deployment.zip"

# Get the numpy Lambda layer ARN (AWS-provided)
# Using the AWSSDKPandas layer which includes numpy
NUMPY_LAYER="arn:aws:lambda:${REGION}:336392948345:layer:AWSSDKPandas-Python312:16"
echo "  Using numpy layer: $NUMPY_LAYER"

# ---- Step 4: Create Lambda Functions ----
echo ""
echo "[Step 4/4] Creating Lambda functions..."

# Function configs: name -> memory_mb, timeout_seconds
# Memory sized for streaming aggregation: 2 * gradient_size + overhead
declare -A FUNC_CONFIGS
FUNC_CONFIGS=(
    ["resnet18"]="512:120"         # 42.7 MB grad -> 512 MB plenty
    ["vgg16"]="2048:300"           # 512 MB grad -> 2 GB for streaming
    ["gpt2_medium"]="2048:600"     # 338 MB per shard (M=4) -> 2 GB
    ["gpt2_large"]="3072:900"      # 738 MB per shard (M=4) -> 3 GB
)

for model in resnet18 vgg16 gpt2_medium gpt2_large; do
    IFS=':' read -r mem timeout <<< "${FUNC_CONFIGS[$model]}"
    func_name="${FUNCTION_PREFIX}-${model}"

    echo "  Creating $func_name (memory=${mem}MB, timeout=${timeout}s)..."

    if aws lambda get-function --function-name "$func_name" 2>/dev/null; then
        aws lambda update-function-code \
            --function-name "$func_name" \
            --zip-file "fileb://$DEPLOY_DIR/deployment.zip" \
            > /dev/null
        aws lambda update-function-configuration \
            --function-name "$func_name" \
            --memory-size "$mem" \
            --timeout "$timeout" \
            --layers "$NUMPY_LAYER" \
            > /dev/null 2>&1 || true
        echo "    Updated existing function."
    else
        aws lambda create-function \
            --function-name "$func_name" \
            --runtime python3.12 \
            --role "$ROLE_ARN" \
            --handler lambda_function.lambda_handler \
            --zip-file "fileb://$DEPLOY_DIR/deployment.zip" \
            --memory-size "$mem" \
            --timeout "$timeout" \
            --layers "$NUMPY_LAYER" \
            --region "$REGION" \
            > /dev/null
        echo "    Created."
    fi
done

# Cleanup temp dir
rm -rf "$DEPLOY_DIR"

echo ""
echo "============================================================"
echo "  SETUP COMPLETE"
echo "============================================================"
echo "  Bucket:    $BUCKET_NAME"
echo "  Functions: ${FUNCTION_PREFIX}-{resnet18,vgg16,gpt2_medium,gpt2_large}"
echo ""
echo "  Next steps:"
echo "    1. Upload gradients:"
echo "       python generate_and_upload_gradients.py --bucket $BUCKET_NAME"
echo ""
echo "    2. Run the experiment:"
echo "       python run_lambda_experiment.py --bucket $BUCKET_NAME"
echo ""
echo "    3. When done, clean up:"
echo "       ./teardown_aws.sh"
echo "============================================================"
