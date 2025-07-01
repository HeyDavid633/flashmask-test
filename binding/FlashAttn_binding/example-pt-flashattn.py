# 2025.06.26 Thu.
# 
# 原存在于  4080-laptop/home/david/Documents/fusion-SC25/model-candidate/my_test2_pt_slidingwin.py
# 
# 通过 my_test1_pytest.py 的例子追踪调用栈；发现PyTorch中的Flash-attn kernel是支持sliding windows的
# 并且packed qkv与否也并不重要
# 此处希望调用起来真实的 PyTorch中FA2 ！
import torch
import torch.nn.functional as F
import math
from einops import rearrange, repeat
# from flash_attn import flash_attn_qkvpacked_func


def construct_local_mask(
    seqlen_q,
    seqlen_k,
    window_size=(-1, -1),  # -1 means infinite window size
    device=None,
):
    row_idx = rearrange(torch.arange(seqlen_q, device=device, dtype=torch.long), "s -> s 1")
    col_idx = torch.arange(seqlen_k, device=device, dtype=torch.long)
    
    sk = seqlen_k
    sq = seqlen_q
        
    if window_size[0] < 0:
        return col_idx > row_idx + sk - sq + window_size[1]
    
    else:
        sk = torch.full_like(col_idx, seqlen_k)
        return torch.logical_or(
            col_idx > torch.minimum(row_idx + sk - sq + window_size[1], sk),
            col_idx < row_idx + sk - sq - window_size[0],
        )

def attention_qkvpacked_ref(
    qkv,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite window size
):
    return attention_ref(
        qkv[:, :, 0],
        qkv[:, :, 1],
        qkv[:, :, 2],
        causal=causal,
        window_size=window_size,
    )

def attention_ref(
    q,
    k,
    v,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite window size
    key_leftpad=None,
):
    """
    Arguments:
        q: (batch_size, seqlen_q, nheads, head_dim)
        k: (batch_size, seqlen_k, nheads_k, head_dim)
        v: (batch_size, seqlen_k, nheads_k, head_dim)
        query_padding_mask: (batch_size, seqlen_q)
        key_padding_mask: (batch_size, seqlen_k)
        causal: whether to apply causal masking
        window_size: (int, int), left and right window size
    Output:
        output: (batch_size, seqlen_q, nheads, head_dim)
        attention: (batch_size, nheads, seqlen_q, seqlen_k), softmax after dropout
    """
    if causal:
        window_size = (window_size[0], 0)
    
    
    print("(before) k.shape: ", k.shape)
    
    
    seqlen_q, seqlen_k = q.shape[1], k.shape[1]
    k = repeat(k, "b s h d -> b s (h g) d", g=q.shape[2] // k.shape[2])
    v = repeat(v, "b s h d -> b s (h g) d", g=q.shape[2] // v.shape[2])
    d = q.shape[-1]
    
    print("(after) k.shape: ", k.shape)

    scores = torch.einsum("bthd,bshd->bhts", q / math.sqrt(d), k)
   
    if window_size[0] >= 0 or window_size[1] >= 0:
        local_mask = construct_local_mask(
            seqlen_q,
            seqlen_k,
            window_size,
            q.device,
        )
        
        # print(local_mask)
        
        scores.masked_fill_(local_mask, float("-inf"))
        
        # print(scores)
        
                    
    attention = torch.softmax(scores, dim=-1).to(v.dtype)
    
    # Some rows might be completely masked out so we fill them with zero instead of NaN
    if window_size[0] >= 0 or window_size[1] >= 0:
        attention = attention.masked_fill(torch.all(local_mask, dim=-1, keepdim=True), 0.0)


    output = torch.einsum("bhts,bshd->bthd", attention, v)

    
    return output, attention


from typing import Optional, Tuple
import flash_attn_2_cuda as flash_attn_gpu

def maybe_contiguous(x):
    return x.contiguous() if x is not None and x.stride(-1) != 1 else x

# @_torch_custom_op_wrapper("flash_attn::_flash_attn_forward", mutates_args=(), device_types="cuda")
def _flash_attn_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dropout_p: float,
    softmax_scale: float,
    causal: bool,
    window_size_left: int,
    window_size_right: int,
    softcap: float,
    alibi_slopes: Optional[torch.Tensor],
    return_softmax: bool
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    q, k, v = [maybe_contiguous(x) for x in (q, k, v)]
    out, softmax_lse, S_dmask, rng_state = flash_attn_gpu.fwd(
        q,
        k,
        v,
        None,
        alibi_slopes,
        dropout_p,
        softmax_scale,
        causal,
        window_size_left,
        window_size_right,
        softcap,
        return_softmax,
        None,
    )
    return out, softmax_lse, S_dmask, rng_state



if __name__ == "__main__":
    torch.random.manual_seed(0)
    print(f"torch.__version__: {torch.__version__}")
    
    dtype = torch.float16
    local = True
    causal = True
    hidden_dim = 64
    seqlen = 256
    batch_size = 1
    head_num = 12
    device = "cuda"
    
    dropout_p = 0.0
    
    
    softcap = 0.0
    alibi_slopes = None
    softmax_scale = None
    return_attn_probs = False
    
    # 生成一个形状为 (2,) 的张量，其中的生成的整数将在 [0, seqlen) 之间
    window_size = (-1, -1) if not local else torch.randint(0, seqlen, (2,))
    # window_size = (2, 0)
    
    # 生成一个形状为 (batch_size, seqlen, 3, head_num, head_num) 的张量，元素为标准正态分布（均值为0，方差为1）
    qkv = torch.randn(batch_size, seqlen, 3, head_num, hidden_dim, device=device, dtype=dtype, requires_grad=False)


    out_ref, attn_ref = attention_qkvpacked_ref(
        qkv, 
        causal=causal, 
        window_size=window_size
    )    
    
    
    if softmax_scale is None:
        softmax_scale = qkv.shape[-1] ** (-0.5)
    q, k, v = qkv[:, :, 0].detach(), qkv[:, :, 1].detach(), qkv[:, :, 2].detach()
    head_size_og = q.size(3)
    if head_size_og % 8 != 0:
        q = torch.nn.functional.pad(q, [0, 8 - head_size_og % 8])
        k = torch.nn.functional.pad(k, [0, 8 - head_size_og % 8])
        v = torch.nn.functional.pad(v, [0, 8 - head_size_og % 8])
    
    

    # out_padded, softmax_lse, S_dmask, rng_state = _flash_attn_forward(
    #         q,
    #         k,
    #         v,
    #         dropout_p,
    #         softmax_scale,
    #         causal=causal,
    #         window_size_left=window_size[0],
    #         window_size_right=window_size[1],
    #         softcap=softcap,
    #         alibi_slopes=alibi_slopes,
    #         return_softmax=return_attn_probs and dropout_p > 0,
    #     )
    
    # _wrapped_flash_attn_forward = torch.ops.flash_attn._flash_attn_forward
    # out_padded, softmax_lse, S_dmask, rng_state = _wrapped_flash_attn_forward(
    #         q,
    #         k,
    #         v,
    #         dropout_p,
    #         softmax_scale,
    #         causal=causal,
    #         window_size_left=window_size[0],
    #         window_size_right=window_size[1],
    #         softcap=softcap,
    #         alibi_slopes=alibi_slopes,
    #         return_softmax=return_attn_probs and dropout_p > 0,
    #     )
    
    # pt_real_FA2_out = out_padded[..., :head_size_og]
    # print("pt_real_FA2_out.shape: ", pt_real_FA2_out.shape)
    print("out_ref.shape:", out_ref.shape)
    
    # print("Hello !")
    # print(torch.ops.flash_attn)
    
    
    
    
    # print(f"Output max diff: {(pt_real_FA2_out - out_ref).abs().max().item()}")
    # print(f"Output mean diff: {(pt_real_FA2_out - out_ref).abs().mean().item()}")