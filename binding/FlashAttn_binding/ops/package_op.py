import torch
from torch.autograd import Function
import flashattn_binding

__all__ = ['flashattn_binding_op']

# 输入尺寸待对齐
class Flashattn_Binding(Function):
    @staticmethod
    def forward(ctx, 
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        dropout_p: float,
        softmax_scale: float,
        causal: bool,
        window_size_left: int,
        window_size_right: int,
        softcap: float,
        alibi_slopes: torch.Tensor,
        return_softmax: bool):  
        
        batch_size = q.size(0)
        seq_len = q.size(1)
        head_num = q.size(2)
        head_size = q.size(3)

        attention_output = torch.zeros((batch_size, seq_len, head_num, head_size), device=q.device, dtype=torch.float16)

        attention_output = flashattn_binding.forward(
            q.contiguous(), 
            k.contiguous(), 
            v.contiguous(),
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
        
        ctx.mark_non_differentiable(attention_output)
        return attention_output


flashattn_binding_op=Flashattn_Binding.apply