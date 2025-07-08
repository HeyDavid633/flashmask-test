# 2025.7.08 Tue.
# 
# 正确性验证，
# FlashAttn 的基准换成了自己绑定的那个
# 添加了带有自己数据结构的FA变体 binding_attn_func
# python correct_verify3.py --batch_size 1 --head_num 1 --head_size 64 --seq_len 256

import sys
import os
import argparse
import torch
from einops import rearrange, repeat
import math
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np

from torch.nn.attention.flex_attention import flex_attention  # FlexAttn

from util.masks import create_block_mask_cached, flex_causal_mask, generate_causal_mask, generate_full_mask, get_sparse_storage
from util.utils import set_dtype, seqlen_to_mask, torch_cuda_identify

from ops.package_op import block_attn_mask_func, binding_attn_func  # Our kernel


def plot_mask_as_blocks(mask, mask_name='mask_plot', seq_len=64):
    mask_data = mask.squeeze().to(torch.int).cpu()   # batch_size=1   
    mask_data_list = mask_data.tolist()
    
    plt.figure(figsize=(6, 6))
    plt.imshow(mask_data_list, cmap='gray_r', interpolation='nearest') 
    
    tick_step = int(seq_len / 8)
    ticks_x = range(0, mask_data.shape[1]+1, tick_step)  
    ticks_y = range(0, mask_data.shape[0]+1, tick_step)  
    plt.xticks(ticks_x) 
    plt.yticks(ticks_y) 
        
    grid_x = range(0, mask_data.shape[1]+1, 32)  
    grid_y = range(0, mask_data.shape[0]+1, 32)
    for x in grid_x:
        plt.axvline(x=x, color='lightgray', linestyle='-', linewidth=0.5)
    for y in grid_y:
        plt.axhline(y=y, color='lightgray', linestyle='-', linewidth=0.5)
    
    plt.savefig(f'{mask_name}.jpg', format='jpg', bbox_inches='tight', pad_inches=0)
    
def check_tensor(other_output, torch_output):
    max_diff = (other_output - torch_output).abs().max().item()
    mean_diff = (other_output - torch_output).abs().mean().item()
    return max_diff, mean_diff
    
def check_tensor_careful(cuda_output, torch_output):
    if cuda_output.shape != torch_output.shape:
        print(f"Error: The shapes of the tensors do not match! cuda_output shape: {cuda_output.shape}, torch_output shape: {torch_output.shape}")
        return
    
    dims = cuda_output.shape
    total_elements = torch.numel(cuda_output)  # 总元素数
    
    mismatch_count = 0  # 不匹配的元素数量
    print("\n","-"*20, "CHECK TENSOR", "-"*20)
    print(f" Tensor dimensions: {dims}")
    print(f" Total elements: {total_elements} \n")
    
    # 用于打印不匹配的元素信息
    for idx in range(total_elements):
        idx_tuple = torch.unravel_index(torch.tensor(idx), cuda_output.shape) 
        if abs(cuda_output[idx_tuple] - torch_output[idx_tuple]) > 0.01 or torch.isnan(cuda_output[idx_tuple]):
            mismatch_count += 1
            idx_tuple = tuple(i.item() for i in idx_tuple)
        
            # if mismatch_count % 128 == 0:
            #     print(f"miss match!!! {idx}:{idx_tuple},  torch_output: {torch_output[idx_tuple]:.5f}, cuda_output: {cuda_output[idx_tuple]:.5f}")
            # print(f"miss match!!! {idx}:{idx_tuple},  torch_output: {torch_output[idx_tuple]:.5f}, cuda_output: {cuda_output[idx_tuple]:.5f}")

    match_rate =  mismatch_count / total_elements * 100
    # print(f"\n Overall miss-match rate:  {mismatch_count} / {total_elements} = {match_rate:.2f}%")
    print(f"\n Overall Match Rate:  {total_elements - mismatch_count} / {total_elements} = {100.0 - match_rate:.2f}%")
    
def check_tensor_plot(cuda_output, torch_output):
    if cuda_output.shape != torch_output.shape:
        print(f"Error: The shapes of the tensors do not match! cuda_output shape: {cuda_output.shape}, torch_output shape: {torch_output.shape}")
        return
    
    dims = cuda_output.shape
    total_elements = torch.numel(cuda_output)  # 总元素数
    
    mismatch_count = 0  # 不匹配的元素数量
    print("\n", "-"*20, "CHECK TENSOR", "-"*20)
    print(f" Tensor dimensions: {dims}")
    print(f" Total elements: {total_elements} \n")
    
    # 用于存储差值的矩阵
    mismatch_matrix = np.zeros((dims[2], dims[3]))  # 假设尺寸是[1, 1, 64, 64]，所以是64x64的矩阵
    
    # 用于打印不匹配的元素信息
    for idx in range(total_elements):
        idx_tuple = torch.unravel_index(torch.tensor(idx), cuda_output.shape) 
        if abs(cuda_output[idx_tuple] - torch_output[idx_tuple]) > 0.01 or torch.isnan(cuda_output[idx_tuple]):
            mismatch_count += 1
            idx_tuple = tuple(i.item() for i in idx_tuple)
            
            # 将差值填充到对应位置
            x, y = idx_tuple[2], idx_tuple[3]  # 假设前两个维度是1
            diff = abs(cuda_output[idx_tuple] - torch_output[idx_tuple]).item()
            mismatch_matrix[x, y] = diff
            
            # print(f"miss match!!! {idx}:{idx_tuple},  torch_output: {torch_output[idx_tuple]:.5f}, cuda_output: {cuda_output[idx_tuple]:.5f}")
    
    match_rate = mismatch_count / total_elements * 100
    print(f"\n Overall miss-match rate:  {mismatch_count} / {total_elements} = {match_rate:.2f}%")
    
    # 绘制热力图
    plt.figure(figsize=(8, 6))
    plt.imshow(mismatch_matrix, cmap='hot', interpolation='nearest')
    plt.colorbar(label='Difference (abs(torch - cuda))')
    plt.title('Mismatch Heatmap')
    plt.xlabel('Width (64)')
    plt.ylabel('Height (64)')
    plt.savefig('error_heatmap.jpg')

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
    parser.add_argument('--head_num', type=int, default=2, help='Number of heads (default: 12)')
    parser.add_argument('--head_size', type=int, default=64, help='Head size (default: 64)')
    parser.add_argument('--seq_len', type=int, default=64, help='Sequence length (default: 256)')
    args = parser.parse_args() 

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
    
    print(f"q (batch_size, seqlen, nheads, head_dim) = {q.shape}\n")
    
    # torch.set_printoptions(profile="full")
    # torch.set_printoptions(profile="default")
    torch.set_printoptions(precision=4, sci_mode=False) # 禁用科学计数法，并设置4位精度   
    
    
    # PyTorch Naive  ---------------------------------------
    pt_out = attention_pytorch(q, k, v, dropout_p=dropout_p, causal=is_causal)
    
    # Binding FlashAttn2  --------------------------------
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.join(current_dir, "FlashAttn_binding"))
    from FlashAttn_binding.ops.package_op import flashattn_binding_func # Binded FA2
    FA_bind_out = flashattn_binding_func(q, k, v, dropout_p=dropout_p, causal=is_causal)
    max_diff, mean_diff = check_tensor(FA_bind_out, pt_out)
    print(f"[CHECK]  FlashAttn2\t  max_diff:{max_diff:.4f}  mean_diff:{mean_diff:.4f}" )
    
    # FlexAttn  ---------------------------------------
    compiled_flex_attention = torch.compile(flex_attention, mode="default", dynamic=False)
    block_mask = create_block_mask_cached(mask_mod, 1, 1, seqlen, seqlen, device=q.device)
    flex_output = compiled_flex_attention(q1, k1, v1, score_mod=score_mod, block_mask=block_mask)
    flex_output1 = flex_output.permute(0, 2, 1, 3)
    max_diff, mean_diff = check_tensor(flex_output1, pt_out)
    print(f"[CHECK]  FlexAttn\t  max_diff:{max_diff:.4f}  mean_diff:{mean_diff:.4f}" )

    BLOCK_M    = 16
    BLOCK_N    = 16
    num_warps  = 1
    nnz, full_row_ptr, full_col_idx, part_row_ptr, part_col_idx, part_block_mask, load_row_ptr, load_col_idx = get_sparse_storage(mask, block_size_m=BLOCK_M, block_size_n=BLOCK_N)
        
    # print(f"nnz: {nnz:.2f}%")
    # print("full_row_ptr:", full_row_ptr)
    # print("full_col_idx:", full_col_idx)
    # print("part_row_ptr:", part_row_ptr)
    # print("part_col_idx:", part_col_idx)
    # print("part_block_mask:\n", part_block_mask)
    
    # print("load_row_ptr:", load_row_ptr)
    # print("load_col_idx:", load_col_idx)
    # exit(0)

    # Binding Attn -------------------------------------
    binding_out = binding_attn_func(q, k, v,
                                full_row_ptr, full_col_idx, 
                                part_row_ptr, part_col_idx, part_block_mask, 
                                load_row_ptr, load_col_idx, 
                                dropout_p=dropout_p, causal=is_causal)    
    max_diff, mean_diff = check_tensor(binding_out, pt_out)
    print(f"[CHECK]  Binding_Attn\t  max_diff:{max_diff:.4f}  mean_diff:{mean_diff:.4f}" )

    # check_tensor_careful(binding_out, pt_out)
    
    