import torch
from torch.autograd import Function
from typing import Optional, Sequence, Tuple, Union
import flashattn_binding

__all__ = ['flashattn_binding_func']


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
    
    out, softmax_lse, S_dmask, rng_state = flashattn_binding.forward(
        q, k, v, None, alibi_slopes, dropout_p, softmax_scale, causal,
        window_size_left, window_size_right, softcap, return_softmax, None,
    )
    return out, softmax_lse, S_dmask, rng_state
    
class FlashAttnFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx,
        q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
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
            q, k, v,
            dropout_p,
            softmax_scale,
            causal=causal,
            window_size_left=window_size[0],
            window_size_right=window_size[1],
            softcap=softcap,
            alibi_slopes=alibi_slopes,
            return_softmax=return_softmax and dropout_p > 0,
        )
        
        out = out_padded[..., :head_size_og]
        return out if not return_softmax else (out, softmax_lse, S_dmask)


#  q, k, v: (batch_size, seqlen, nheads, headdim)
#      out: (batch_size, seqlen, nheads, headdim)
def flashattn_binding_func(
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
        False,  # torch.is_grad_enabled() 不保存反向时需要的中间变量，减少显存占用
    )

