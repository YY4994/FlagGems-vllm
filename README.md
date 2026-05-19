[English|[中文版](./README_cn.md)]

## Introduction

FlagGems-vllm is part of [FlagOS](https://flagos.io/).
FlagGems-vllm is a high-performance operator library designed for multiple hardware backends. It provides optimized implementations of common vLLM operators and supports high-performance inference and deployment for a variety of widely used models.

FlagGems-vllm is a high-performance deep learning operator library implemented using the [Triton programming language](https://github.com/openai/triton) launched by OpenAI.

## Features

- Operators have undergone deep performance tuning
- Triton kernel call optimization
- Flexible multi-backend support mechanism
- Support for common vllm operators (moe_align_block_size, etc.)

## Quick Installation
### Install Dependencies
```shell
pip install -U scikit-build-core>=0.11 pybind11 ninja cmake
```
### Install FlagGems-vllm
```shell
git clone https://github.com/flagos-ai/FlagGems-vllm.git
cd FlagGems-vllm
pip install  .
```

## Usage Example

```python
import torch
import flaggems_vllm

# Prepare a simple topk_ids tensor for MoE routing
num_tokens = 128
topk = 2
num_experts = 16
block_size = 32

topk_ids = torch.randint(
	low=0,
	high=num_experts,
	size=(num_tokens, topk),
	device='cuda',
	dtype=torch.int32,
)

# Align tokens by expert and block size
sorted_ids, expert_ids, num_tokens_post_pad = flaggems_vllm.ops.moe_align_block_size(
	topk_ids=topk_ids,
	block_size=block_size,
	num_experts=num_experts,
)

print(sorted_ids.shape, expert_ids.shape, num_tokens_post_pad)
```

## Tests and Benchmark Quick Start

The following commands are verified in this repository and can be used for quick validation after installation.

### Run tests

```shell
cd /workspace/FlagGems-vllm
pytest -q tests --collect-only
pytest -q tests/test_moe_align_block_size.py --quick
```

### Run benchmark

```shell
cd /workspace/FlagGems-vllm
pytest -q benchmark --collect-only
pytest -q benchmark/test_moe_align_block_size_triton.py::test_moe_align_block_size_triton --level core --iter 1 --warmup 1
```

### Notes

- Most tests/benchmarks require a CUDA-capable GPU runtime.
- `--collect-only` is recommended first to quickly check import and discovery.

This project is licensed under the [Apache (version 2.0) License](./LICENSE).
