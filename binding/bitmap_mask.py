# 7.09 Wed.
# 用以展示 bitmap mask 的转换数据结构

import math
import torch

import argparse
import torch
from util.utils import set_dtype, seqlen_to_mask, torch_cuda_identify
from util.masks import generate_causal_mask, generate_sliding_mask, generate_dilated_mask,generate_longformer_mask, generate_bigbird_mask, get_sparse_storage, plot_mask_as_blocks, generate_full_mask
import matplotlib.pyplot as plt
import numpy as np

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

# 原来的存储结构
def get_sparse_storage(Mask, block_size_m, block_size_n):
    n = Mask.shape[-1]
    
    full_row_ptr = [] # 初始化存储列表
    full_col_idx = []
    part_row_ptr = []
    part_col_idx = []
    part_block_mask = []
    load_row_ptr = []
    load_col_idx = []
    
    total_elements = n * n
    nnz = torch.count_nonzero(Mask) / total_elements * 100  
    
    full_block_count = 0
    part_block_count = 0  
    load_block_count = 0
    full_row_ptr.append(full_block_count)
    part_row_ptr.append(part_block_count)
    load_row_ptr.append(load_block_count)
    for i in range(0, n, block_size_m):
        for j in range(0, n, block_size_n):
            block = Mask[0, i:i+block_size_m, j:j+block_size_n]

            if torch.all(block == 1):  # full块
                full_col_idx.append(j // block_size_n)
                full_block_count += 1
                
                load_col_idx.append(j // block_size_n)
                load_block_count += 1
                
            elif torch.all(block == 0):
                continue
            
            else:  
                block = -10000.0 * (1.0 - block.to(torch.float16))  # 将0变为-10000.0，1变为0
                part_col_idx.append(j // block_size_n)
                part_block_mask.append(block)  
                part_block_count += 1
                
                load_col_idx.append(j // block_size_n)
                load_block_count += 1
        
        full_row_ptr.append(full_block_count)
        part_row_ptr.append(part_block_count)
        load_row_ptr.append(load_block_count)
        
    full_row_ptr = torch.tensor(full_row_ptr, dtype=torch.int32, device=Mask.device)
    full_col_idx = torch.tensor(full_col_idx, dtype=torch.int32, device=Mask.device)
    part_row_ptr = torch.tensor(part_row_ptr, dtype=torch.int32, device=Mask.device)
    part_col_idx = torch.tensor(part_col_idx, dtype=torch.int32, device=Mask.device)
    load_row_ptr = torch.tensor(load_row_ptr, dtype=torch.int32, device=Mask.device)
    load_col_idx = torch.tensor(load_col_idx, dtype=torch.int32, device=Mask.device)
    
    # 如果 part_block_mask 为空，填充一个全1的矩阵
    if len(part_block_mask) == 0:
        part_block_mask = torch.ones(1, block_size_m, block_size_n, dtype=torch.float16, device=Mask.device)
    else:
        part_block_mask = torch.stack(part_block_mask, dim=0) # (num_part_blocks, block_size, block_size)
    
    return nnz, full_row_ptr, full_col_idx, part_row_ptr, part_col_idx, part_block_mask, load_row_ptr, load_col_idx

def block_to_bitmap(block):
    """将8x8块转换为uint64位图"""
    assert block.shape == (8, 8), "块尺寸必须是8x8"
    bitmap = np.uint64(0)
    for i in range(8):
        for j in range(8):
            if block[i, j] != 0:  # 非零元素
                bitmap |= np.uint64(1) << np.uint64(i * 8 + j)
    return bitmap

def bitmap_to_block(bitmap):
    """将uint64位图转换为8x8块"""
    block = torch.zeros((8, 8), dtype=torch.float16)
    for pos in range(64):
        if bitmap & (np.uint64(1) << np.uint64(pos)):
            i, j = pos // 8, pos % 8
            block[i, j] = 1.0
    return block

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
        # mask_mod = flex_causal_mask
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
    
    print(f"q (batch_size, seqlen, nheads, head_dim) = {q.shape}")
    print(f"mask (batch_size, seqlen, seqlen) = {mask.shape}\n")
    
    # print(mask_name, "=", mask.shape)
    # print(mask)
    # plot_mask_as_blocks(mask, mask_name + f'_{seqlen}', seqlen)
    
    BLOCK_M    = 16
    BLOCK_N    = 16
    num_warps  = 1
    
    # nnz, full_row_ptr, full_col_idx, part_row_ptr, part_col_idx, part_block_mask, load_row_ptr, load_col_idx = get_sparse_storage(mask, block_size_m=BLOCK_M, block_size_n=BLOCK_N)
    
    nnz, full_row_ptr, full_col_idx, part_row_ptr, part_col_idx, load_row_ptr, load_col_idx, inner_bitmaps = get_OuterTile_storage(mask, BLOCK_M, BLOCK_N)
    
    print(f"nnz: {nnz:.2f}%")
    print("full_row_ptr:", full_row_ptr)
    print("full_col_idx:", full_col_idx)
    print("part_row_ptr:", part_row_ptr)
    print("part_col_idx:", part_col_idx)
    
    print("part_bitmap_mask:\n", inner_bitmaps)
    print_tile_structure(inner_bitmaps.cpu(), outer_shape = (BLOCK_M, BLOCK_N))  # 确保转移到CPU
    
    print("load_row_ptr:", load_row_ptr)
    print("load_col_idx:", load_col_idx)
    # exit(0)
    
