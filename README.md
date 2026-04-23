# Replication Package: GradsSharding

**Paper:** *Shard the Gradient, Scale the Model: Serverless Federated Aggregation via Gradient Partitioning*
**Authors:** Amine Barrak (Oakland University)

## Overview

This replication package contains the source code, experiment scripts, raw results, and figure generation scripts for reproducing all experiments reported in the paper. The paper compares three serverless federated learning aggregation architectures:

- **GradsSharding** (our contribution): partitions the gradient tensor into M shards across parallel Lambda functions
- **λ-FL** (baseline): two-level tree aggregation with √N leaf aggregators
- **LIFL** (baseline): three-level hierarchical aggregation with shared-memory optimization

## Directory Structure

```
replication-package/
├── systems/                        # FL system implementations
│   ├── grads-sharding/             # GradsSharding (gradient partitioning)
│   ├── lambda-fl/                  # λ-FL (tree-based aggregation)
│   ├── lifl/                       # LIFL (hierarchical + shared memory)
│   └── shared/                     # Common code (models, datasets, training, metrics)
├── experiments/                    # Experiment runner scripts
│   ├── rq1_ps_idle_time.py         # RQ1: Parameter server idle ratio
│   ├── rq1_shard_ablation.py       # RQ2 Part A: Shard ablation on HPC
│   ├── runner.py                   # System execution helper
│   ├── run_experiment.py           # Main experiment orchestrator
│   └── lambda_deployment/          # AWS Lambda deployment code
│       ├── run_vgg16_shard_sweep.py    # RQ2 Part B: Lambda shard sweep
│       ├── run_rq3_lambda.py           # RQ3: Cross-architecture comparison
│       ├── lambda_rq3.py               # Lambda handler for RQ3
│       ├── lambda_function.py          # Generic Lambda handler
│       ├── generate_rq3_gradients.py   # Synthetic gradient generation
│       ├── generate_and_upload_gradients.py
│       ├── setup_rq3_aws.py            # AWS resource provisioning
│       ├── setup_aws.py / setup_aws.sh
│       └── teardown_aws.py / teardown_aws.sh
├── results/                        # Raw experimental results (JSON)
│   ├── rq1_ps_idle_time/           # RQ1: per-model idle ratios
│   ├── rq2_shard_ablation_hpc/     # RQ2 Part A: HPC shard sweep (ResNet-18, VGG-16)
│   ├── rq2_lambda_shard_sweep/     # RQ2 Part B: Lambda shard sweep (VGG-16, M=1..16)
│   └── rq3_cross_architecture/     # RQ3: all 3 architectures × 4 model sizes
├── figures/
│   ├── scripts/                    # Python scripts to generate paper figures
│   └── figures/                    # figures (PDF + PNG)
├── hpc/                            # SLURM scripts for HPC cluster execution
├── paper/                          # LaTeX source of the paper
│   ├── 00.main2.tex                # Main document
│   ├── 01.Abstract.tex ... 09.Conclusion.tex
│   ├── references.bib
│   └── images/                     # Figures used in the paper
└── setup.py                        # Package installation
```

## Research Questions

### RQ1: Serverless Motivation (Parameter Server Idle Time)

**Question:** What fraction of each FL round does the parameter server spend idle?

**How to reproduce:**

*HPC (Part A):* Requires GPU node with PyTorch.
```bash
python -m experiments.rq1_ps_idle_time
```

*Lambda (Part B):* Uses pre-recorded Lambda round times. Results in `results/rq1_ps_idle_time/`.

**Paper reference:** Section 5.1, Table 1, Figure 2

### RQ2: Shard Ablation (Effect of M on Memory, Latency, Cost)

**Question:** How does shard count M affect GradsSharding performance?

**How to reproduce:**

*Part A — HPC ablation:*
```bash
python -m experiments.rq1_shard_ablation
```

*Part B — Lambda shard sweep (VGG-16, M ∈ {1,2,4,8,16}):*
```bash
cd experiments/lambda_deployment
python setup_rq3_aws.py        # Provision AWS resources
python run_vgg16_shard_sweep.py # Run sweep on real Lambda
python teardown_aws.py          # Clean up
```

**Paper reference:** Section 5.2, Tables 2–3, Figures 3–5

### RQ3: Cross-Architecture Comparison on AWS Lambda

**Question:** How do the three architectures compare in cost, latency, and feasibility?

**How to reproduce:**
```bash
cd experiments/lambda_deployment
python setup_rq3_aws.py            # Provision Lambda functions + S3 buckets
python generate_rq3_gradients.py   # Generate synthetic gradients for all model sizes
python run_rq3_lambda.py           # Deploy all 3 architectures × 4 model sizes
python teardown_aws.py             # Clean up AWS resources
```

**Models tested:** ResNet-18 (43 MB), VGG-16 (512 MB), GPT-2 Large (2,953 MB), Synthetic 5 GB

**Paper reference:** Section 5.3, Table 4, Figure 6

## Prerequisites

### Software
- Python 3.10+
- PyTorch 2.0+ (for HPC experiments)
- AWS CLI configured with credentials (for Lambda experiments)
- Dependencies: `pip install -r systems/shared/requirements.txt`

### Hardware
- **HPC experiments (RQ1 Part A, RQ2 Part A):** GPU node (tested on NVIDIA Tesla V100, 16 GB)
- **Lambda experiments (RQ1 Part B, RQ2 Part B, RQ3):** AWS account with Lambda and S3 access

### AWS Configuration
Lambda experiments use:
- **Region:** us-east-1
- **Runtime:** Python 3.12 with AWSSDKPandas layer
- **Memory:** Configured per architecture (up to 10,240 MB)
- **Timeout:** 900 seconds

## Regenerating Figures

All paper figures can be regenerated from the raw results:

```bash
cd figures/scripts
python generate_rq3_figure.py         # Figure 6: RQ3 cross-architecture comparison
python gen_shard_sweep_plots.py       # Figures 3-5: RQ2 shard sweep
python generate_rq1_figure.py         # Figure 2: RQ1 PS idle time
python generate_paper_figures.py      # All figures
```


## Key Results Summary

| RQ | Finding |
|----|---------|
| RQ1 | Parameter server idle 80–99.6% of each FL round |
| RQ2 | Near-linear latency scaling: 16.2× speedup at M=16; S3 I/O accounts for >99% of aggregation time |
| RQ3 | Cost crossover at ~500 MB; GradsSharding 2.7× cheaper at VGG-16; only viable architecture beyond ~3 GB gradient size |

## Contact

Amine Barrak — aminebarrak@gmail.com
