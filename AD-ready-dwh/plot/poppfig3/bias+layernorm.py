import torch
import triton
import triton.language as tl
import time
import os
import torch.nn.functional as F
import tilelang
from tilelang.autotuner import *
import tilelang.language as T
import itertools
import argparse
from functools import partial

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 64}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
    ],
    key=['M'],  
)
@triton.jit
def batched_bias_layer_norm_kernel(
    X, Y, Bias, Mean, Rstd,
    batch_stride, seq_stride, hidden_size, eps,
    M,
    BLOCK_SIZE: tl.constexpr,
):
    batch_id = tl.program_id(0)
    seq_id = tl.program_id(1)
    batch_offset = batch_id * batch_stride
    seq_offset = seq_id * seq_stride
    x_ptr = X + batch_offset + seq_offset
    y_ptr = Y + batch_offset + seq_offset

    # 计算 mean
    _mean = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, hidden_size, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < hidden_size
        x = tl.load(x_ptr + cols, mask=mask, other=0.).to(tl.float32)
        bias = tl.load(Bias + cols, mask=mask, other=0.).to(tl.float32)
        biased_val = x + bias
        _mean += tl.where(mask, biased_val, 0.0)
    mean = tl.sum(_mean, axis=0) / hidden_size

    # 计算方差
    _var = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, hidden_size, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < hidden_size
        x = tl.load(x_ptr + cols, mask=mask, other=0.).to(tl.float32)
        bias = tl.load(Bias + cols, mask=mask, other=0.).to(tl.float32)
        biased_val = x + bias
        diff = tl.where(mask, biased_val - mean, 0.0)
        _var += diff * diff
    var = tl.sum(_var, axis=0) / hidden_size
    rstd = 1.0 / tl.sqrt(var + eps)

    # 写入 mean/rstd
    mean_idx = batch_id * seq_stride + seq_id
    tl.store(Mean + mean_idx, mean)
    tl.store(Rstd + mean_idx, rstd)

    # 归一化并写入结果
    for off in range(0, hidden_size, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < hidden_size
        x = tl.load(x_ptr + cols, mask=mask, other=0.).to(tl.float32)
        bias = tl.load(Bias + cols, mask=mask, other=0.).to(tl.float32)
        biased_val = x + bias
        x_hat = (biased_val - mean) * rstd
        tl.store(y_ptr + cols, x_hat, mask=mask)

def triton_batched_bias_layernorm(x, bias, eps=1e-5):
    batch_size, seq_len, hidden_size = x.shape
    M = seq_len
    y = torch.empty_like(x)
    mean = torch.empty((batch_size * seq_len,), dtype=torch.float32, device=x.device)
    rstd = torch.empty((batch_size * seq_len,), dtype=torch.float32, device=x.device)

    # BLOCK_SIZE = min(128, triton.next_power_of_2(hidden_size))
    grid = (batch_size, seq_len)

    batched_bias_layer_norm_kernel[grid](
        x, y, bias, mean, rstd,
        x.stride(0), x.stride(1), hidden_size, eps,
        M,
    )
    return y

def get_configs():
    iter_params = dict(blk_m=[1, 2, 4, 8], num_stages=[ 1, 2, 3], threads=[128, 256])
    return [dict(zip(iter_params, values)) for values in itertools.product(*iter_params.values())]

@autotune(configs=get_configs(), warmup=0, rep=1)
@tilelang.jit(out_idx=[-1], pass_configs={"tl.disable_tma_lower": True})
def bias_layernorm(B, M, N, blk_m=1,num_stages=1,threads=128):
    
    dtype = "float16"

    @T.prim_func
    def main(A: T.Tensor((B, M, N), dtype),
             Bias: T.Tensor((N,), dtype),
             Out: T.Tensor((B, M, N), dtype)):

        with T.Kernel(B, T.ceildiv(M, blk_m), threads=threads) as (bz, bx):
            A_shared = T.alloc_shared((blk_m, N), dtype)
            Bias_shared = T.alloc_shared((N,), dtype)
            A_local = T.alloc_fragment((blk_m, N), dtype)
            A_pow_local = T.alloc_fragment((blk_m, N), dtype)
            A_sum = T.alloc_fragment((blk_m,), dtype)
            A_powsum = T.alloc_fragment((blk_m,), dtype)

            T.copy(A[bz, bx * blk_m:(bx + 1) * blk_m, :], A_shared)
            T.copy(Bias[:], Bias_shared)

            for i, j in T.Parallel(blk_m, N):
                A_local[i, j] = A_shared[i, j] + Bias_shared[j]
            for i, j in T.Parallel(blk_m, N):
                A_pow_local[i, j] = A_local[i, j] * A_local[i, j]

            T.reduce_sum(A_local, A_sum, dim=1)
            T.reduce_sum(A_pow_local, A_powsum, dim=1)

            for i in T.Parallel(blk_m):
                mean_val = A_sum[i] / N
                mean_sq_val = A_powsum[i] / N
                var_val = mean_sq_val - mean_val * mean_val
                rstd_val = T.rsqrt(var_val + 1e-5)
                A_sum[i] = mean_val
                A_powsum[i] = rstd_val

            for i, j in T.Parallel(blk_m, N):
                mean_i = A_sum[i]
                rstd_i = A_powsum[i]
                A_local[i, j] = (A_local[i, j] - mean_i) * rstd_i

            T.copy(A_local, Out[bz, bx * blk_m:(bx + 1) * blk_m, :])

    return main



# def ref_program(x, bias):
#     x_biased = x + bias
#     mean = x_biased.mean(-1, keepdim=True)
#     variance = x_biased.var(-1, keepdim=True, unbiased=False)
#     return (x_biased - mean) * torch.rsqrt(variance + 1e-5)

def ref_program(x, bias, eps=1e-5):
    x_biased = x + bias
    norm_shape = (x.shape[2],)  # Normalize over hidden dimension
    y = torch.nn.functional.layer_norm(x_biased, norm_shape, weight=None, bias=None, eps=eps)
    return y


import itertools
import numpy as np

def benchmark_implementations():
    batch_sizes = [1, 8]
    seq_lens = [128, 512, 1024, 4096]
    hidden_sizes = [512, 1024]
    
    results = []
    
    for batch_size, hidden_size, seq_len in itertools.product(batch_sizes, hidden_sizes, seq_lens ):
        print(f"Benchmarking batch={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}")
        
        # Create random input data
        x = torch.randn((batch_size, seq_len, hidden_size), device="cuda", dtype=torch.float16)
        bias = torch.randn((hidden_size,), device="cuda", dtype=torch.float16)
        

        tilelang_kernel = bias_layernorm(batch_size, seq_len, hidden_size)

        # Warm up
        for _ in range(10):
            y_triton = triton_batched_bias_layernorm(x, bias)
            y_pytorch = ref_program(x, bias)
        
        # Verify correctness
        y_triton = tilelang_kernel(x, bias)
        y_pytorch = ref_program(x, bias)
        max_error = torch.max(torch.abs(y_triton - y_pytorch)).item()

        # Benchmark Tilelang implementation
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(100):
            y_tilelang = tilelang_kernel(x, bias)
        torch.cuda.synchronize()
        tilelang_time = (time.time() - start) / 100
        
        # Benchmark Triton implementation
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(100):
            y_triton = triton_batched_bias_layernorm(x, bias)
        torch.cuda.synchronize()
        triton_time = (time.time() - start) / 100
        
        # Benchmark PyTorch implementation
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(100):
            y_pytorch = ref_program(x, bias)
        torch.cuda.synchronize()
        pytorch_time = (time.time() - start) / 100
        
        speedup = pytorch_time / triton_time
        
        results.append((batch_size, seq_len, hidden_size, tilelang_time, triton_time, pytorch_time, speedup, max_error))
        
        print(f"Tilelang time: {tilelang_time*1000:.3f} ms")
        print(f"Triton time: {triton_time*1000:.3f} ms")
        print(f"PyTorch time: {pytorch_time*1000:.3f} ms")
        print(f"Speedup: {speedup:.2f}x")
        print(f"Max absolute error: {max_error}\n")
    
    # Print summary table
    print("\nPerformance Summary:")
    print("Batch | SeqLen | Hidden | Tilelang (ms) | Triton (ms) | PyTorch (ms) | Speedup")
    print("-" * 80)
    for r in results:
        print(f"{r[0]:5} | {r[1]:6} | {r[2]:6} | {r[3]*1000:.4f}  | {r[4]*1000:.4f}   | {r[5]*1000:.4f}   | {r[6]:.6f}")

if __name__ == "__main__":
    benchmark_implementations()
