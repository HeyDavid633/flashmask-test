# 2025.6.29 binding_test3.py
# 
# 原 FA_binding_test.py
# conduct FlashAttn 同步于Tri-Dap Rep
# 其调用方式 主要参考自 /flash-attention/flash_attn/flash_attn_interface.py
# 从 flash-attn 源代码中截取出来前向的部分，以setup.py的方式将以前TriDao仓库中所安装的 flash_attn_2_cuda 给修改成了我的 flashattn_binding； 并在代码中更换了调用方式为 flashattn_binding.forward

import argparse
from typing import Optional, Sequence, Tuple, Union
from einops import rearrange, repeat
import math
import torch
import torch
import torch.nn as nn
import torch.nn.functional as F

import flashattn_binding
# import flash_attn_2_cuda as flash_attn_cuda
# from flash_attn import flash_attn_func

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


def _flash_attn_forward(
    q, k, v, dropout_p, softmax_scale, causal, window_size, alibi_slopes, return_softmax
):
    maybe_contiguous = lambda x: x.contiguous() if x.stride(-1) != 1 else x
    q, k, v = [maybe_contiguous(x) for x in (q, k, v)]
    
    # 相比于 binding_test2.py 的修改为： flash_attn_cuda.fwd --- flashattn_binding.forward
    out, q, k, v, out_padded, softmax_lse, S_dmask, rng_state = flashattn_binding.forward(
        q, k, v, None, alibi_slopes, dropout_p, softmax_scale, 
        causal, window_size[0], window_size[1], return_softmax, None,)
    
    return out, q, k, v, out_padded, softmax_lse, S_dmask, rng_state


def _flash_attn_forward(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
    dropout_p: float,
    softmax_scale: float,
    causal: bool,
    window_size_left: int, window_size_right: int,
    softcap: float,
    alibi_slopes: Optional[torch.Tensor],
    return_softmax: bool
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    
    maybe_contiguous = lambda x: x.contiguous() if x.stride(-1) != 1 else x
    q, k, v = [maybe_contiguous(x) for x in (q, k, v)]
    
    # 相比于 binding_test2.py 的修改为： flash_attn_cuda.fwd --- flashattn_binding.forward
    out, softmax_lse, S_dmask, rng_state = flashattn_binding.forward(
        q, k, v, None, alibi_slopes, dropout_p, softmax_scale, causal,
        window_size_left, window_size_right, softcap, return_softmax, None,
    )
    return out, softmax_lse, S_dmask, rng_state


class FlashAttnFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        softcap,
        alibi_slopes,
        deterministic,
        return_softmax,
        is_grad_enabled,
    ):
        is_grad = is_grad_enabled and any(
            x.requires_grad for x in [q, k, v]
        )
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)
        head_size_og = q.size(3)
        if head_size_og % 8 != 0:
            q = torch.nn.functional.pad(q, [0, 8 - head_size_og % 8])
            k = torch.nn.functional.pad(k, [0, 8 - head_size_og % 8])
            v = torch.nn.functional.pad(v, [0, 8 - head_size_og % 8])
        out_padded, softmax_lse, S_dmask, rng_state = _flash_attn_forward(
            q,
            k,
            v,
            dropout_p,
            softmax_scale,
            causal=causal,
            window_size_left=window_size[0],
            window_size_right=window_size[1],
            softcap=softcap,
            alibi_slopes=alibi_slopes,
            return_softmax=return_softmax and dropout_p > 0,
        )
        if is_grad:
            ctx.save_for_backward(q, k, v, out_padded, softmax_lse, rng_state)
            ctx.dropout_p = dropout_p
            ctx.softmax_scale = softmax_scale
            ctx.causal = causal
            ctx.window_size = window_size
            ctx.softcap = softcap
            ctx.alibi_slopes = alibi_slopes
            ctx.deterministic = deterministic
        out = out_padded[..., :head_size_og]
        return out if not return_softmax else (out, softmax_lse, S_dmask)

def flash_attn_func(
    q, k, v,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite context window
    softcap=0.0, # 0.0 means deactivated
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
):
    """dropout_p should be set to 0.0 during evaluation
    Supports multi-query and grouped-query attention (MQA/GQA) by passing in KV with fewer heads
    than Q. Note that the number of heads in Q must be divisible by the number of heads in KV.
    For example, if Q has 6 heads and K, V have 2 heads, head 0, 1, 2 of Q will attention to head
    0 of K, V, and head 3, 4, 5 of Q will attention to head 1 of K, V.

    If causal=True, the causal mask is aligned to the bottom right corner of the attention matrix.
    For example, if seqlen_q = 2 and seqlen_k = 5, the causal mask (1 = keep, 0 = masked out) is:
        1 1 1 1 0
        1 1 1 1 1
    If seqlen_q = 5 and seqlen_k = 2, the causal mask is:
        0 0
        0 0
        0 0
        1 0
        1 1
    If the row of the mask is all zero, the output will be zero.

    If window_size != (-1, -1), implements sliding window local attention. Query at position i
    will only attend to keys between
    [i + seqlen_k - seqlen_q - window_size[0], i + seqlen_k - seqlen_q + window_size[1]] inclusive.

    Arguments:
        q: (batch_size, seqlen, nheads, headdim)
        k: (batch_size, seqlen, nheads_k, headdim)
        v: (batch_size, seqlen, nheads_k, headdim)
        dropout_p: float. Dropout probability.
        softmax_scale: float. The scaling of QK^T before applying softmax.
            Default to 1 / sqrt(headdim).
        causal: bool. Whether to apply causal attention mask (e.g., for auto-regressive modeling).
        window_size: (left, right). If not (-1, -1), implements sliding window local attention.
        alibi_slopes: (nheads,) or (batch_size, nheads), fp32. A bias of
            (-alibi_slope * |i + seqlen_k - seqlen_q - j|)
            is added to the attention score of query i and key j.
        deterministic: bool. Whether to use the deterministic implementation of the backward pass,
            which is slightly slower and uses more memory. The forward pass is always deterministic.
        return_attn_probs: bool. Whether to return the attention probabilities. This option is for
           testing only. The returned probabilities are not guaranteed to be correct
           (they might not have the right scaling).
    Return:
        out: (batch_size, seqlen, nheads, headdim).
        softmax_lse [optional, if return_attn_probs=True]: (batch_size, nheads, seqlen). The
            logsumexp of each row of the matrix QK^T * scaling (e.g., log of the softmax
            normalization factor).
        S_dmask [optional, if return_attn_probs=True]: (batch_size, nheads, seqlen, seqlen).
            The output of softmax (possibly with different scaling). It also encodes the dropout
            pattern (negative means that location was dropped, nonnegative means it was kept).
    """
    return FlashAttnFunc.apply(
        q,
        k,
        v,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        softcap,
        alibi_slopes,
        deterministic,
        return_attn_probs,
        torch.is_grad_enabled(),
    )

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
    dropout_p = 0.0
    device = "cuda"
    dtype = torch.float16 
    
    q = torch.randn(batch_size, seqlen, nheads, headdim, device=device, dtype=dtype, requires_grad=False)
    k = torch.randn(batch_size, seqlen, nheads, headdim, device=device, dtype=dtype, requires_grad=False)
    v = torch.randn(batch_size, seqlen, nheads, headdim, device=device, dtype=dtype, requires_grad=False)
    
    # causal = True
    # out = flash_attn_func(q, k, v, dropout_p=dropout_p, causal=causal)
    # pt_out = attention_pytorch(q, k, v, dropout_p=dropout_p, causal=causal)
    # print(f"Casual: {causal} |  Output max diff: {(out - pt_out).abs().max().item()}")
    # # print(f"Output mean diff: {(out - pt_out).abs().mean().item()}")

    # causal = False
    # out2 = flash_attn_func(q, k, v, dropout_p=dropout_p, causal=causal)
    # pt_out2 = attention_pytorch(q, k, v, dropout_p=dropout_p, causal=causal)
    # print(f"Casual: {causal} |  Output max diff: {(out2 - pt_out2).abs().max().item()}")
    
    
    with torch.cuda.stream(torch.cuda.Stream()):
        causal = True
        out = flash_attn_func(q, k, v, dropout_p=dropout_p, causal=causal)
        pt_out = attention_pytorch(q, k, v, dropout_p=dropout_p, causal=causal)
        print(f"Casual: {causal} |  Output max diff: {(out - pt_out).abs().max().item()}")


    with torch.cuda.stream(torch.cuda.Stream()):
        causal = False
        out = flash_attn_func(q, k, v, dropout_p=dropout_p, causal=causal)
        pt_out = attention_pytorch(q, k, v, dropout_p=dropout_p, causal=causal)
        print(f"Casual: {causal} |  Output max diff: {(out - pt_out).abs().max().item()}")
