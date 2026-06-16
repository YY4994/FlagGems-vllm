import pytest
import torch
import triton

import flaggems_vllm
from benchmark.base import Benchmark


class ParallelNSACompressionBenchmark(Benchmark):
    DEFAULT_DTYPES = [torch.bfloat16, torch.float16]
    # 形状: (B, T, H, HQ, D)
    DEFAULT_SHAPES = [
        # T 扫描 — 长序列到超长序列 (B=1, H=4, HQ=64, D=64)
        (1, 8192, 4, 64, 64),
        (1, 16384, 4, 64, 64),
        (1, 32768, 4, 64, 64),
        (1, 65536, 4, 64, 64),
        # D 对比 — 宽/窄 head (B=1, T=16K, H=4, HQ=64)
        (1, 16384, 4, 64, 64),
        (1, 16384, 4, 64, 128),
        # 多序列场景 (B=2, T=16K, H=4, HQ=64, D=64)
        (2, 16384, 4, 64, 64),
    ]
    DEFAULT_SHAPE_DESC = "B, T, H, HQ, D"

    def set_more_shapes(self):
        return self.DEFAULT_SHAPES

    def set_shapes(self, shape_file_path=None):
        self.shapes = self.DEFAULT_SHAPES
        self.shape_desc = self.DEFAULT_SHAPE_DESC

    def get_input_iter(self, cur_dtype):
        for B, T, H, HQ, D in self.shapes:
            yield self._build_inputs(B, T, H, HQ, D, cur_dtype)

    def _build_inputs(
        self, B: int, T: int, H: int, HQ: int, D: int, dtype: torch.dtype
    ):
        device = flaggems_vllm.device
        block_size = 64
        # 压缩后的 token 数
        TC = triton.cdiv(T, block_size)

        q = torch.randn(B, T, HQ, D, device=device, dtype=dtype)
        k = torch.randn(B, TC, H, D, device=device, dtype=dtype)
        v = torch.randn(B, TC, H, D, device=device, dtype=dtype)
        scale = D**-0.5

        return (
            q,
            k,
            v,
            block_size,
            scale,
            None,  # cu_seqlens
        )


@pytest.mark.parallel_nsa_compression
def test_perf_parallel_nsa_compression():
    bench = ParallelNSACompressionBenchmark(
        op_name="parallel_nsa_compression",
        torch_op=flaggems_vllm.parallel_nsa_compression,
    )
    bench.set_gems(flaggems_vllm.parallel_nsa_compression)
    bench.run()
