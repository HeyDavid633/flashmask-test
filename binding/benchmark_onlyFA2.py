# 2025.7.13 Sun.
#  
# 为了获取基准性能 
# 以我binding版的FlashAttn2为基准:
# 对于 cuasal ｜ sliding 以 is_causal = True 为准
# 对于 full ｜ bigbird ｜ longformer 以 is_causal = False 为准


import sys
import os
import argparse
import torch
import numpy as np
from ops.package_op import binding_attn_func  # Our kernel

from einops import rearrange, repeat
import math

import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel        # Orignal FlashAttn2
from torch.nn.attention.flex_attention import flex_attention  # FlexAttn

from util.masks import generate_causal_mask, generate_sliding_mask, generate_dilated_mask,generate_longformer_mask, generate_bigbird_mask, generate_full_mask
from util.masks import create_block_mask_cached, flex_causal_mask, flex_longformer_mask, flex_sliding_window_mask, flex_bigbird_mask 

from ops.package_op import binding_attn_func  # Binded FA2
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

def block_to_bitmap(block):
    """将8x8块转换为uint64位图"""
    assert block.shape == (8, 8), "块尺寸必须是8x8"
    bitmap = np.uint64(0)
    for i in range(8):
        for j in range(8):
            if block[i, j] != 0:  # 非零元素
                bitmap |= np.uint64(1) << np.uint64(i * 8 + j)
    return bitmap

def bitmap_to_matrix(bitmap):
    """
    安全可靠的位图转换函数
    处理所有情况：
    - CUDA张量 -> CPU原生int
    - 零值处理
    - 大整数位运算
    """
    matrix = torch.zeros((8, 8), dtype=torch.uint8)
    
    # 类型统一处理
    if isinstance(bitmap, torch.Tensor):
        bitmap = int(bitmap.item())  # 确保转为Python原生int
    bitmap = int(bitmap)  # 二次确保
    
    # 特殊处理0值
    if bitmap == 0:
        return matrix
    
    # 安全位运算
    for pos in range(64):
        if bitmap & (1 << pos):  # Python原生int运算
            i, j = divmod(pos, 8)
            matrix[i, j] = 1
    return matrix

def print_tile_structure(inner_bitmaps, outer_shape=(32,32)):
    """
    打印分层存储结构
    :param inner_bitmaps: uint64张量 包含所有InnerTile 
    :param outer_shape: OuterTile尺寸
    """
    outer_m, outer_n = outer_shape
    inner_per_outer = (outer_m // 8) * (outer_n // 8)  # 每个OuterTile包含的InnerTile数
    
    print(f"\n 分层存储结构 每个OuterTile包含{inner_per_outer}个InnerTile:")
    for outer_idx in range(len(inner_bitmaps) // inner_per_outer):
        start = outer_idx * inner_per_outer
        end = start + inner_per_outer
        print(f"OuterTile {outer_idx} 的InnerTile位图:")
        
        for inner_idx, bitmap in enumerate(inner_bitmaps[start:end]):
            print(f"OuterTile {outer_idx}; InnerTile {inner_idx}: {hex(bitmap.item())}")
            
            # 可视化矩阵
            matrix = bitmap_to_matrix(bitmap)
            for row in matrix:
                print(' '.join(['■' if x == 1 else '□' for x in row.tolist()]))
        print()

def get_InnerTile_bitmap(outer_tile):
    """
    将 OuterTile 中的part块转换为多个 InnerTile 的位图数组
    :param outer_tile: outer_tile_size x outer_tile_size 的OuterTile矩阵
    :return: 16个uint64位图组成的列表（按列优先顺序）
    """
    bitmaps = []
    outer_tile_size = outer_tile.shape[0]
    
    # 列优先遍历InnerTile
    for j in range(0, outer_tile_size, 8):
        for i in range(0, outer_tile_size, 8):
            inner_tile = outer_tile[i:i+8, j:j+8]
            bitmap = 0
            # 行优先编码InnerTile
            for bi in range(8):
                for bj in range(8):
                    if inner_tile[bi, bj] != 0:
                        bitmap |= 1 << (bi * 8 + bj)
            bitmaps.append(bitmap)
    
    return bitmaps

def get_OuterTile_storage(Mask, block_size_m=32, block_size_n=32):
    """
    外层分块存储结构处理 OuterTile
    :param Mask: 输入掩码张量 (batch_size, seqlen, seqlen)
    :param block_size_m: OuterTile行尺寸
    :param block_size_n: OuterTile列尺寸
    :return: 稀疏存储结构 + InnerTile位图列表
    """
    batch_size, n, _ = Mask.shape
    total_elements = n * n
    nnz = torch.count_nonzero(Mask) / total_elements * 100  
    
    # 初始化存储结构
    full_row_ptr = [0]
    full_col_idx = []
    part_row_ptr = [0]
    part_col_idx = []
    load_row_ptr = [0]
    load_col_idx = []
    all_inner_bitmaps = []  # 存储所有InnerTile位图
    
    full_block_count = 0
    part_block_count = 0
    load_block_count = 0
    
    for b in range(batch_size):
        for i in range(0, n, block_size_m):
            for j in range(0, n, block_size_n):
                outer_tile = Mask[b, i:i+block_size_m, j:j+block_size_n]
                
                if torch.all(outer_tile == 1):  # 全1块
                    full_col_idx.append(j // block_size_n)
                    full_block_count += 1
                    
                    # 即使是全1块，也记录其InnerTile位图
                    # inner_bitmaps = get_InnerTile_bitmap(outer_tile.cpu().numpy())
                    # all_inner_bitmaps.extend(inner_bitmaps)
                    
                elif torch.all(outer_tile == 0):
                    continue
                
                else:  # 部分填充块
                    part_col_idx.append(j // block_size_n)
                    part_block_count += 1
                    
                    # 获取该OuterTile的所有InnerTile位图
                    inner_bitmaps = get_InnerTile_bitmap(outer_tile.cpu().numpy())
                    all_inner_bitmaps.extend(inner_bitmaps)
                
                # 无论哪种块，都需要记录到load结构
                load_col_idx.append(j // block_size_n)
                load_block_count += 1
            
            # 更新行指针
            full_row_ptr.append(full_block_count)
            part_row_ptr.append(part_block_count)
            load_row_ptr.append(load_block_count)
    
    # 转换为张量
    device = Mask.device
    full_row_ptr = torch.tensor(full_row_ptr, dtype=torch.int32, device=device)
    full_col_idx = torch.tensor(full_col_idx, dtype=torch.int32, device=device)
    part_row_ptr = torch.tensor(part_row_ptr, dtype=torch.int32, device=device)
    part_col_idx = torch.tensor(part_col_idx, dtype=torch.int32, device=device)
    load_row_ptr = torch.tensor(load_row_ptr, dtype=torch.int32, device=device)
    load_col_idx = torch.tensor(load_col_idx, dtype=torch.int32, device=device)
    
    # 转换位图为uint64张量
    inner_bitmaps_tensor = torch.tensor(
        [int(x) for x in all_inner_bitmaps], 
        dtype=torch.uint64, 
        device=device
    )
    
    return nnz, full_row_ptr, full_col_idx, part_row_ptr, part_col_idx, load_row_ptr, load_col_idx, inner_bitmaps_tensor



if __name__ == "__main__":
    torch.manual_seed(0)
    torch.cuda.empty_cache()
    running_device = torch_cuda_identify(print_info = False)
    torch._dynamo.config.cache_size_limit = 64

    parser = argparse.ArgumentParser(description="Give the parameters for the attention test (with Mask)")
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size (default: 1)')
    parser.add_argument('--head_num', type=int, default=12, help='Number of heads (default: 12)')
    parser.add_argument('--head_size', type=int, default=64, help='Head size (default: 64)')
    parser.add_argument('--seq_len', type=int, default=256, help='Sequence length (default: 256)')
    parser.add_argument('--mask_id', type=int, default=0, help='Mask type: 0-Casual | 1-Sliding | 2-Longformer | 3-BigBird (default: 0)')
    args = parser.parse_args() 


    warmup_iters = 10
    running_iters = 100

    batch_size = args.batch_size
    nheads     = args.head_num
    headdim    = args.head_size # hidden_dim = nheads * headdim
    seqlen     = args.seq_len
    mask_id    = args.mask_id
    dropout_p = 0.0
    dtype = torch.float16
    is_causal = False
    
    fill_rate = 0.1  # BigBird fill rate
    BLOCK_M    = 64  # 务必要是 64
    BLOCK_N    = 64
    
    avg_seq_len = seqlen
    low, high = (2 * avg_seq_len - seqlen, seqlen + 1)
    input_lens = torch.randint(low=low, high=high, size=(batch_size,))
    seqlen_mask = seqlen_to_mask(input_lens, seqlen)
    attr_mask   = set_dtype(torch.tile(seqlen_mask, dims=(seqlen,)).reshape(batch_size, seqlen, seqlen).cuda(), "fp16")
    
    mask_mod = None
    score_mod = None
    
    if(mask_id == 0):
        is_causal = True
        mask_name = 'Causal_Mask'
        mask_mod = flex_causal_mask
        mask = generate_causal_mask(attr_mask).cuda()
    elif(mask_id == 1):
        mask_name = 'Full_Mask'
        mask = generate_full_mask(attr_mask).cuda()
        
    
    print("Testing Mask Name:", mask_name)

    nnz, full_row_ptr, full_col_idx, part_row_ptr, part_col_idx, load_row_ptr, load_col_idx, inner_bitmaps = get_OuterTile_storage(mask, BLOCK_M, BLOCK_N)

    q = torch.randn(batch_size, seqlen, nheads, headdim, device=running_device, dtype=dtype, requires_grad=False)
    k = torch.randn(batch_size, seqlen, nheads, headdim, device=running_device, dtype=dtype, requires_grad=False)
    v = torch.randn(batch_size, seqlen, nheads, headdim, device=running_device, dtype=dtype, requires_grad=False)
    q1 = q.permute(0, 2, 1, 3)
    k1 = k.permute(0, 2, 1, 3)
    v1 = v.permute(0, 2, 1, 3)
    
    
    # PyTorch Naive  ---------------------------------------
    # Binding Attn -------------------------------------
    for i in range(warmup_iters + running_iters):
        if i == warmup_iters:    
            t_start = time_stamp_cudasync()
        binding_out = binding_attn_func(q, k, v,
                                full_row_ptr, full_col_idx, 
                                part_row_ptr, part_col_idx, inner_bitmaps, 
                                load_row_ptr, load_col_idx, 
                                dropout_p=dropout_p, causal=is_causal)     
    t_end = time_stamp_cudasync()
    ourkernel_time = (t_end - t_start) * 1000 / running_iters
    # print(" bs:{} | seq:{} |  Bind Kernel : {:.3f} ms / iter |  Speedup/FA2: {:.3f}".format(batch_size, seqlen, ourkernel_time, flashAttn2_time/ourkernel_time)) 
    print(" bs:{} | seq:{} |  FlashAttn2: {:.3f} ms / iter".format(batch_size, seqlen, ourkernel_time)) 
    

        
        