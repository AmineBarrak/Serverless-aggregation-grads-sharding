#!/usr/bin/env python3
"""
Setup script for the Federated Learning Benchmarking Suite

Install with: pip install -e .
"""

from setuptools import setup, find_packages

with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='fl-benchmark',
    version='1.0.0',
    description='Federated Learning Aggregation Benchmarks: Lambda-FL vs LIFL vs GradsSharding',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='FL Benchmark Authors',
    python_requires='>=3.10',
    packages=find_packages(include=[
        'shared',
        'shared.*',
        'lambda_fl',
        'lambda_fl.*',
        'lifl',
        'lifl.*',
        'grads_sharding',
        'grads_sharding.*',
        'experiments',
        'experiments.*',
    ]),
    install_requires=[
        'numpy>=1.21.0',
        'scipy>=1.7.0',
        'scikit-learn>=0.24.0',
        'torch>=1.10.0',
        'torchvision>=0.11.0',
        'matplotlib>=3.3.0',
        'pandas>=1.3.0',
        'tqdm>=4.50.0',
    ],
    extras_require={
        'dev': [
            'pytest>=6.2.0',
            'pytest-cov>=2.12.0',
            'black>=21.0',
            'flake8>=3.9.0',
            'mypy>=0.910',
        ],
    },
    entry_points={
        'console_scripts': [
            'fl-benchmark=experiments.run_experiments:main',
        ],
    },
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    keywords='federated learning aggregation lambda-fl lifl grads-sharding benchmark',
    url='https://github.com/your-org/grads-sharding',
    project_urls={
        'Documentation': 'https://github.com/your-org/grads-sharding#readme',
        'Source': 'https://github.com/your-org/grads-sharding',
        'Tracker': 'https://github.com/your-org/grads-sharding/issues',
    },
    include_package_data=True,
    zip_safe=False,
)
