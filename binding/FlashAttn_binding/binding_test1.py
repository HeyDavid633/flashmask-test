# 2025.6.29 binding_test1.py
# 
# 针对 FA_binding_test.py 展开写多个版本
# 
# 方式1: 直接使用已 pip 安装好的 flash_attn


import argparse
from typing import Optional, Union
from einops import rearrange, repeat
import math
import torch
import torch
import torch.nn as nn
import torch.nn.functional as F

from flash_attn import flash_attn_func


def attention_pytorch(q, k, v, dropout_p=0.0, causal=True, attn_bias=None):
    """
    Arguments:
        q, k, v: (batch_size, seqlen, nheads, head_dim)
        dropout_p: float
        attn_bias: (batch_size, nheads, seqlen, seqlen) or (1, nheads, seqlen, seqlen)
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
    torch.random.manual_seed(0)
    parser = argparse.ArgumentParser(description="Give the parameters for the attention test (with Mask)")
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size (default: 1)')
    parser.add_argument('--head_num', type=int, default=12, help='Number of heads (default: 12)')
    parser.add_argument('--head_size', type=int, default=64, help='Head size (default: 64)')
    parser.add_argument('--seq_len', type=int, default=256, help='Sequence length (default: 256)')
    args = parser.parse_args() 

    batch_size = args.batch_size
    nheads   = args.head_num
    headdim  = args.head_size # hidden_dim= nheads * headdim
    seqlen    = args.seq_len
    causal = True
    dropout_p = 0.0
    device = "cuda"
    dtype = torch.float16 
    
    q = torch.randn(batch_size, seqlen, nheads, headdim, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(batch_size, seqlen, nheads, headdim, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(batch_size, seqlen, nheads, headdim, device=device, dtype=dtype, requires_grad=True)
    
    out = flash_attn_func(q, k, v, dropout_p=dropout_p, causal=causal)
    pt_out = attention_pytorch(q, k, v, dropout_p=dropout_p, causal=causal)
    
    print(f"Output max diff: {(out - pt_out).abs().max().item()}")
    # print(f"Output mean diff: {(out - pt_out).abs().mean().item()}")
    