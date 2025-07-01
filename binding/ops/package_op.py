import torch
from torch.autograd import Function

import block_attn_mask


__all__ = ['block_attn_mask_op']


# result.shape = {batch_size, seq_len, head_num * head_size}        
class Block_Attn_Mask(Function):
    @staticmethod
    def forward(ctx, q, k, v, 
                full_row_ptr, full_col_idx, 
                part_row_ptr, part_col_idx, part_block_mask,
                load_row_ptr, load_col_idx,
                BLOCK_M, BLOCK_N, num_warps):  
        
        block_attn_mask.forward(
            q.contiguous(), 
            k.contiguous(), 
            v.contiguous(),
            full_row_ptr.contiguous(), full_col_idx.contiguous(), 
            part_row_ptr.contiguous(), part_col_idx.contiguous(), part_block_mask.contiguous(), 
            load_row_ptr.contiguous(), load_col_idx.contiguous(),
            BLOCK_M, BLOCK_N, num_warps)

        ctx.mark_non_differentiable(q)
        return q

block_attn_mask_op =  Block_Attn_Mask.apply
