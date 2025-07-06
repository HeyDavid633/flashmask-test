# 2025.7.02 Wed.
#  
# 以下三角为例，测量具体的性能；以Torch中的FlashAttn2为对比对象
# python bench1.py --batch_size 1 --head_num 1 --head_size 64 --seq_len 256

import sys
import os
import argparse
import torch
from ops.package_op import block_attn_mask_func, binding_attn_func  # Our kernel

from einops import rearrange, repeat
import math

import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel        # Orignal FlashAttn2
from torch.nn.attention.flex_attention import flex_attention  # FlexAttn

from util.masks import create_block_mask_cached, flex_causal_mask, generate_causal_mask, generate_full_mask
from util.utils import set_dtype, seqlen_to_mask, torch_cuda_identify, time_stamp_cudasync

def attention_pytorch(q, k, v, dropout_p=0.0, causal=True):
    """
    Arguments:
        q, k, v: (batch_size, seqlen, nheads, head_dim)
        dropout_p: float
    Output:
        output: (batch_size, seqlen, nheads, head_dim)
    """
    batch_size, seqlen, nheads, d = q.shape
    q = rearrange(q, 'b t h d -> (b h) t d')
    k = rearrange(k, 'b s h d -> (b h) d s')
    softmax_scale = 1.0 / math.sqrt(d)
    
    scores = torch.empty(batch_size * nheads, seqlen, seqlen, dtype=q.dtype, device=q.device)
    scores = rearrange(torch.baddbmm(scores, q, k, beta=1.0, alpha=softmax_scale),
                       '(b h) t s -> b h t s', h=nheads)
    
    if causal:
        causal_mask = torch.triu(torch.full((seqlen, seqlen), -10000.0, device=scores.device), 1)
        scores = scores + causal_mask.to(dtype=scores.dtype)
    
    attention = torch.softmax(scores, dim=-1)
    attention_drop = F.dropout(attention, dropout_p)
    output = torch.einsum('bhts,bshd->bthd', attention_drop , v)
    return output.to(dtype=q.dtype)



if __name__ == "__main__":
    torch.manual_seed(0)
    torch.cuda.empty_cache()
    running_device = torch_cuda_identify(print_info = True)
    torch._dynamo.config.cache_size_limit = 64

    parser = argparse.ArgumentParser(description="Give the parameters for the attention test (with Mask)")
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size (default: 1)')
    parser.add_argument('--head_num', type=int, default=12, help='Number of heads (default: 12)')
    parser.add_argument('--head_size', type=int, default=64, help='Head size (default: 64)')
    parser.add_argument('--seq_len', type=int, default=256, help='Sequence length (default: 256)')
    args = parser.parse_args() 

    warmup_iters = 10
    running_iters = 100

    batch_size = args.batch_size
    nheads   = args.head_num
    headdim  = args.head_size # hidden_dim = nheads * headdim
    seqlen    = args.seq_len
    dropout_p = 0.0
    dtype = torch.float16
    is_causal = True


    avg_seq_len = seqlen
    low, high = (2 * avg_seq_len - seqlen, seqlen + 1)
    input_lens = torch.randint(low=low, high=high, size=(batch_size,))
    seqlen_mask = seqlen_to_mask(input_lens, seqlen)
    attr_mask   = set_dtype(torch.tile(seqlen_mask, dims=(seqlen,)).reshape(batch_size, seqlen, seqlen).cuda(), "fp16")
    if is_causal: 
        mask_name = 'Causal_Mask'
        mask_mod = flex_causal_mask
        score_mod = None
        mask = generate_causal_mask(attr_mask).cuda()
    else:
        mask_name = 'Full_Mask'

    q = torch.randn(batch_size, seqlen, nheads, headdim, device=running_device, dtype=dtype, requires_grad=False)
    k = torch.randn(batch_size, seqlen, nheads, headdim, device=running_device, dtype=dtype, requires_grad=False)
    v = torch.randn(batch_size, seqlen, nheads, headdim, device=running_device, dtype=dtype, requires_grad=False)
    q1 = q.permute(0, 2, 1, 3)
    k1 = k.permute(0, 2, 1, 3)
    v1 = v.permute(0, 2, 1, 3)

    
    # PyTorch Naive  ---------------------------------------
    for i in range(warmup_iters + running_iters):
        if i == warmup_iters:    
            t0_start = time_stamp_cudasync()
        pt_out = attention_pytorch(q, k, v, dropout_p=dropout_p, causal=is_causal)
    t0_end = time_stamp_cudasync()
    base_time = (t0_end - t0_start) * 1000 / running_iters
    print(" bs:{} | seq:{} |  Torch Naive : {:.3f} ms / iter".format(batch_size, seqlen, base_time)) 

    # FlashAttn2   Note: Torch >= 2.2.0
    for i in range(warmup_iters + running_iters):
        if i == warmup_iters:    
            t2_start = time_stamp_cudasync()
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            FA2_out = torch.nn.functional.scaled_dot_product_attention(q1, k1, v1, is_causal=is_causal)
        FA2_out1 = FA2_out.permute(0, 2, 1, 3)
    t2_end = time_stamp_cudasync()
    flashAttn2_time = (t2_end - t2_start) * 1000 / running_iters
    print(" bs:{} | seq:{} |  FlashAttn2  : {:.3f} ms / iter".format(batch_size, seqlen, flashAttn2_time)) 

    # FlexAttn  ---------------------------------------
    compiled_flex_attention = torch.compile(flex_attention, mode="default", dynamic=False)
    block_mask = create_block_mask_cached(mask_mod, 1, 1, seqlen, seqlen, device=q.device)
    for i in range(warmup_iters + running_iters):
        if i == warmup_iters:    
            t3_start = time_stamp_cudasync()
        flex_output = compiled_flex_attention(q1, k1, v1, score_mod=score_mod, block_mask=block_mask)
        flex_output1 = flex_output.permute(0, 2, 1, 3)
    t3_end = time_stamp_cudasync()
    flexattn_time = (t3_end - t3_start) * 1000 / running_iters
    print(" bs:{} | seq:{} |   FlexAttn   : {:.3f} ms / iter".format(batch_size, seqlen, flexattn_time)) 
    
    # Binding Attn -------------------------------------
    for i in range(warmup_iters + running_iters):
        if i == warmup_iters:    
            t_start = time_stamp_cudasync()
        binding_out = binding_attn_func(q, k, v, dropout_p=dropout_p, causal=is_causal)
    t_end = time_stamp_cudasync()
    ourkernel_time = (t_end - t_start) * 1000 / running_iters
    print(" bs:{} | seq:{} |  Bind Kernel : {:.3f} ms / iter |  Speedup/FA2: {:.3f}".format(batch_size, seqlen, ourkernel_time, flashAttn2_time/ourkernel_time)) 
        