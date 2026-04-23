#!/usr/bin/env python3
"""
AWS Infrastructure Setup for RQ3: Cross-Architecture Comparison.

Creates Lambda functions for all three architectures across all model sizes:
  - GradsSharding:  one function per model (processes one shard at a time)
  - λ-FL:           one function per model (leaf + root use same function)
  - LIFL:           one function per model (all levels use same function)

All functions use the SAME unified handler (lambda_rq3.py) with a 'mode' field
to dispatch to the correct logic.

Memory configurations:
  - GradsSharding: sized for one shard (2 × shard_size + overhead)
  - λ-FL / LIFL:   sized for full gradient (2 × grad_size + overhead)
  - Models exceeding Lambda's 3008 MB limit for full gradients get
    λ-FL/LIFL functions created but marked as INFEASIBLE (for documentation)

Usage:
    python setup_rq3_aws.py
    python setup_rq3_aws.py --skip-role   # if role already exists
"""

import json
import time
import sys
import os
import zipfile
import tempfile
import argparse
import math

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

# AWS-provided layer with numpy
NUMPY_LAYER = f"arn:aws:lambda:{REGION}:336392948345:layer:AWSSDKPandas-Python312:16"

# Lambda max memory
LAMBDA_MAX_MEMORY_MB = 3008
OVERHEAD_MB = 450  # Python runtime + boto3 + AWSSDKPandas (numpy+pandas) layer

# Model definitions
MODELS = {
    'resnet18': {
        'params': 11_181_642,     # ~42.7 MB
        'num_shards': 4,
    },
    'vgg16': {
        'params': 134_301_514,    # ~512.3 MB
        'num_shards': 4,
    },
    'gpt2_large': {
        'params': 774_030_080,    # ~2952.7 MB
        'num_shards': 4,
    },
    'synthetic_5gb': {
        'params': 1_342_177_280,  # ~5120 MB
        'num_shards': 8,
    },
}


def grad_size_mb(num_params):
    return num_params * 4 / (1024 * 1024)


def compute_memory_configs(model_name, config):
    """Compute Lambda memory for each architecture for a given model.

    Returns dict: architecture -> {memory_mb, timeout, feasible, reason}
    """
    num_params = config['params']
    num_shards = config['num_shards']
    gsize = grad_size_mb(num_params)
    shard_size = gsize / num_shards

    configs = {}

    # Memory during streaming: running_sum + S3 body + parsed array = 3 × input_size
    # GradsSharding: processes one shard at a time
    gs_mem_needed = math.ceil(3 * shard_size + OVERHEAD_MB)
    gs_mem = min(max(gs_mem_needed, 128), LAMBDA_MAX_MEMORY_MB)  # clamp to [128, 3008]
    gs_feasible = gs_mem_needed <= LAMBDA_MAX_MEMORY_MB
    configs['grads_sharding'] = {
        'memory_mb': gs_mem,
        'timeout': 900 if gsize > 500 else 300,
        'feasible': gs_feasible,
        'reason': f'shard={shard_size:.0f}MB, peak_mem={3*shard_size:.0f}MB + {OVERHEAD_MB}MB overhead',
    }

    # λ-FL and LIFL: process full gradients
    full_mem_needed = math.ceil(3 * gsize + OVERHEAD_MB)
    full_mem = min(max(full_mem_needed, 128), LAMBDA_MAX_MEMORY_MB)
    full_feasible = full_mem_needed <= LAMBDA_MAX_MEMORY_MB

    for arch in ['lambda_fl', 'lifl']:
        configs[arch] = {
            'memory_mb': full_mem,
            'timeout': 900 if gsize > 500 else 300,
            'feasible': full_feasible,
            'reason': f'full_grad={gsize:.0f}MB, peak_mem={3*gsize:.0f}MB + {OVERHEAD_MB}MB overhead'
                      + (f' > {LAMBDA_MAX_MEMORY_MB}MB LIMIT' if not full_feasible else ''),
        }

    return configs


def create_deployment_package():
    """Create zip with lambda_rq3.py as the handler."""
    print(f"\nCreating deployment package...")

    # Use the unified RQ3 handler
    handler_path = os.path.join(os.path.dirname(__file__), "lambda_rq3.py")
    if not os.path.exists(handler_path):
        print(f"  ERROR: lambda_rq3.py not found at {handler_path}")
        sys.exit(1)

    zip_path = os.path.join(tempfile.gettempdir(), "rq3_lambda.zip")
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.write(handler_path, "lambda_rq3.py")

    size_kb = os.path.getsize(zip_path) / 1024
    print(f"  Created: {zip_path} ({size_kb:.1f} KB)")
    return zip_path


def ensure_iam_role(iam_client):
    """Ensure IAM role exists with proper permissions."""
    print(f"\nChecking IAM role: {ROLE_NAME}")

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
        print(f"  Exists: {role_arn}")
        return role_arn
    except ClientError:
        print(f"  Creating role...")
        response = iam_client.create_role(
            RoleName=ROLE_NAME,
            AssumeRolePolicyDocument=json.dumps(trust_policy),
            Description="Role for GradsSharding Lambda experiments"
        )
        role_arn = response['Role']['Arn']

        iam_client.attach_role_policy(
            RoleName=ROLE_NAME,
            PolicyArn="arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
        )

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

        print(f"  Created: {role_arn}")
        print(f"  Waiting 10s for IAM propagation...")
        time.sleep(10)
        return role_arn


def create_or_update_function(lambda_client, func_name, role_arn, zip_bytes,
                               memory_mb, timeout):
    """Create or update a single Lambda function."""
    try:
        lambda_client.get_function(FunctionName=func_name)

        # Wait for any in-progress updates
        while True:
            resp = lambda_client.get_function(FunctionName=func_name)
            state = resp['Configuration'].get('LastUpdateStatus', 'Successful')
            if state in ('Successful', 'Failed'):
                break
            time.sleep(3)

        lambda_client.update_function_code(
            FunctionName=func_name,
            ZipFile=zip_bytes,
        )

        # Wait for code update
        while True:
            resp = lambda_client.get_function(FunctionName=func_name)
            state = resp['Configuration'].get('LastUpdateStatus', 'Successful')
            if state in ('Successful', 'Failed'):
                break
            time.sleep(3)

        lambda_client.update_function_configuration(
            FunctionName=func_name,
            MemorySize=memory_mb,
            Timeout=timeout,
            Layers=[NUMPY_LAYER],
        )
        return "updated"

    except ClientError as e:
        if 'ResourceNotFoundException' in str(e):
            lambda_client.create_function(
                FunctionName=func_name,
                Runtime='python3.12',
                Role=role_arn,
                Handler='lambda_rq3.lambda_handler',
                Code={'ZipFile': zip_bytes},
                MemorySize=memory_mb,
                Timeout=timeout,
                Layers=[NUMPY_LAYER],
            )
            return "created"
        raise


def main():
    parser = argparse.ArgumentParser(description="Setup AWS for RQ3 experiments")
    parser.add_argument('--skip-role', action='store_true', help='Skip IAM role creation')
    parser.add_argument('--dry-run', action='store_true', help='Show plan only')
    args = parser.parse_args()

    print("=" * 65)
    print("  RQ3 AWS INFRASTRUCTURE SETUP")
    print("=" * 65)
    print(f"  Region:  {REGION}")
    print(f"  Account: {ACCOUNT_ID}")
    print(f"  Bucket:  {BUCKET_NAME}")
    print(f"  Handler: lambda_rq3.py (unified)")
    print("=" * 65)

    # Plan: show all functions to create
    all_functions = []
    print(f"\n  FUNCTION PLAN:")
    print(f"  {'Function Name':<45} {'Mem':>5} {'T/O':>5} {'Feasible':>8}")
    print(f"  {'-'*70}")

    for model_name, config in MODELS.items():
        mem_configs = compute_memory_configs(model_name, config)
        for arch, ac in mem_configs.items():
            func_name = f"rq3-{arch.replace('_', '-')}-{model_name.replace('_', '-')}"
            status = "YES" if ac['feasible'] else "SKIP"
            print(f"  {func_name:<45} {ac['memory_mb']:>5} {ac['timeout']:>5} {status:>8}")
            if ac['feasible']:
                all_functions.append({
                    'name': func_name,
                    'memory_mb': ac['memory_mb'],
                    'timeout': ac['timeout'],
                    'model': model_name,
                    'arch': arch,
                })

    print(f"\n  Total functions to create: {len(all_functions)}")
    infeasible = sum(1 for m in MODELS for a, c in compute_memory_configs(m, MODELS[m]).items()
                     if not c['feasible'])
    print(f"  Infeasible (skipped):     {infeasible}")

    if args.dry_run:
        print("\n  DRY RUN complete.")
        return

    # Setup
    s3_client = boto3.client('s3', region_name=REGION)
    iam_client = boto3.client('iam', region_name=REGION)
    lambda_client = boto3.client('lambda', region_name=REGION)

    # S3 bucket
    print(f"\nChecking S3 bucket: {BUCKET_NAME}")
    try:
        s3_client.head_bucket(Bucket=BUCKET_NAME)
        print(f"  Exists.")
    except ClientError:
        if REGION == "us-east-1":
            s3_client.create_bucket(Bucket=BUCKET_NAME)
        else:
            s3_client.create_bucket(
                Bucket=BUCKET_NAME,
                CreateBucketConfiguration={"LocationConstraint": REGION}
            )
        print(f"  Created.")

    # IAM role
    if args.skip_role:
        role_arn = f"arn:aws:iam::{ACCOUNT_ID}:role/{ROLE_NAME}"
        print(f"\nSkipping IAM role (using {role_arn})")
    else:
        role_arn = ensure_iam_role(iam_client)

    # Deployment package
    zip_path = create_deployment_package()
    with open(zip_path, 'rb') as f:
        zip_bytes = f.read()
    os.remove(zip_path)

    # Create functions
    print(f"\nCreating {len(all_functions)} Lambda functions...")
    for func_info in all_functions:
        fn = func_info['name']
        print(f"  {fn} (mem={func_info['memory_mb']}MB, timeout={func_info['timeout']}s)...",
              end=" ", flush=True)
        status = create_or_update_function(
            lambda_client, fn, role_arn, zip_bytes,
            func_info['memory_mb'], func_info['timeout']
        )
        print(status)

    # Summary
    print(f"\n{'='*65}")
    print(f"  SETUP COMPLETE")
    print(f"{'='*65}")
    print(f"  Functions created/updated: {len(all_functions)}")
    print(f"  Infeasible (not created): {infeasible}")
    print(f"\n  Next steps:")
    print(f"    1. python generate_rq3_gradients.py --bucket {BUCKET_NAME}")
    print(f"    2. python run_rq3_lambda.py --bucket {BUCKET_NAME}")
    print(f"{'='*65}")

    # Save function registry for the orchestrator
    registry = {}
    for func_info in all_functions:
        model = func_info['model']
        arch = func_info['arch']
        if model not in registry:
            registry[model] = {}
        registry[model][arch] = {
            'function_name': func_info['name'],
            'memory_mb': func_info['memory_mb'],
            'timeout': func_info['timeout'],
        }

    # Add infeasible entries
    for model_name, config in MODELS.items():
        mem_configs = compute_memory_configs(model_name, config)
        for arch, ac in mem_configs.items():
            if not ac['feasible']:
                if model_name not in registry:
                    registry[model_name] = {}
                registry[model_name][arch] = {
                    'function_name': None,
                    'memory_mb': ac['memory_mb'],
                    'timeout': ac['timeout'],
                    'infeasible': True,
                    'reason': ac['reason'],
                }

    reg_path = os.path.join(os.path.dirname(__file__), "rq3_function_registry.json")
    with open(reg_path, 'w') as f:
        json.dump(registry, f, indent=2)
    print(f"\n  Function registry saved: {reg_path}")


if __name__ == '__main__':
    main()
