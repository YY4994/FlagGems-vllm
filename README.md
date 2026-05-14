[English|[中文版](./README_cn.md)]

## Introduction

FlagDNN is part of [FlagOS](https://flagos.io/).
FlagDNN is a deep neural network computing library oriented towards multiple chip backends. It provides high-performance implementations of common deep learning operators, supporting efficient computation in fields such as deep learning, computer vision, natural language processing, and artificial intelligence.

FlagDNN is a high-performance deep learning operator library implemented using the [Triton programming language](https://github.com/openai/triton) launched by OpenAI.

## Features

- Operators have undergone deep performance tuning
- Triton kernel call optimization
- Flexible multi-backend support mechanism
- Support for common deep learning operators (ReLU, etc.)

## Quick Installation
### Install Dependencies
```shell
pip install -U scikit-build-core>=0.11 pybind11 ninja cmake
```
### Install FlagDNN
```shell
git clone https://github.com/flagos-ai/FlagDNN.git
cd FlagDNN
pip install  .
```

## Usage Example

```python
import torch
import flaggems_vllm

# Create a tensor
x = torch.randn(1024, device='cuda')

# Apply ReLU activation
y = flaggems_vllm.ops.relu(x)
```

## Tests and Benchmark Quick Start

The following commands are verified in this repository and can be used for quick validation after installation.

### Run tests

```shell
cd /workspace/FlagGems-vllm
pytest -q tests --collect-only
pytest -q tests/test_outer.py --quick
```

### Run benchmark

```shell
cd /workspace/FlagGems-vllm
pytest -q benchmark --collect-only
pytest -q benchmark/test_outer.py::test_outer --level core --iter 1 --warmup 1
```

### Notes

- Most tests/benchmarks require a CUDA-capable GPU runtime.
- `--collect-only` is recommended first to quickly check import and discovery.

This project is licensed under the [Apache (version 2.0) License](./LICENSE).
