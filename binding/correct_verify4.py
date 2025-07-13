# 2025.7.09 Wed.
# 
# 在 bitmap_mask.py 以后真实地传入给 cuda 这个 unit64的数组
# 需要在这里验证 自己的 mask 加的是否正确，由于分块尺寸 64 * 64；
# 所以 bitmap 的存储应该是 64 个 unint_64 的数据
# 
# python correct_verify4.py --batch_size 1 --head_num 1 --head_size 64 --seq_len 128
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
    running_device = torch_cuda_identify(print_info = True)
    torch._dynamo.config.cache_size_limit = 64

    parser = argparse.ArgumentParser(description="Give the parameters for the attention test (with Mask)")
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size (default: 1)')
    parser.add_argument('--head_num', type=int, default=1, help='Number of heads (default: 12)')
    parser.add_argument('--head_size', type=int, default=64, help='Head size (default: 64)')
    parser.add_argument('--seq_len', type=int, default=128, help='Sequence length (default: 256)')
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
    
    print(f"q (batch_size, seqlen, nheads, head_dim) = {q.shape}")
    print(f"mask (batch_size, seqlen, seqlen) = {mask.shape}\n")
    
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
    # compiled_flex_attention = torch.compile(flex_attention, mode="default", dynamic=False)
    # block_mask = create_block_mask_cached(mask_mod, 1, 1, seqlen, seqlen, device=q.device)
    # flex_output = compiled_flex_attention(q1, k1, v1, score_mod=score_mod, block_mask=block_mask)
    # flex_output1 = flex_output.permute(0, 2, 1, 3)
    # max_diff, mean_diff = check_tensor(flex_output1, pt_out)
    # print(f"[CHECK]  FlexAttn\t  max_diff:{max_diff:.4f}  mean_diff:{mean_diff:.4f}" )

    BLOCK_M    = 64  # 务必要是 64
    BLOCK_N    = 64
    num_warps  = 1
    
    nnz, full_row_ptr, full_col_idx, part_row_ptr, part_col_idx, load_row_ptr, load_col_idx, inner_bitmaps = get_OuterTile_storage(mask, BLOCK_M, BLOCK_N)
    
    # print(f"nnz: {nnz:.2f}%")
    # print("full_row_ptr:", full_row_ptr)
    # print("full_col_idx:", full_col_idx)
    # print("part_row_ptr:", part_row_ptr)
    # print("part_col_idx:", part_col_idx)
    
    # print("part_bitmap_mask:\n", inner_bitmaps)
    # print_tile_structure(inner_bitmaps.cpu(), outer_shape = (BLOCK_M, BLOCK_N))  # 确保转移到CPU
    
    # print("load_row_ptr:", load_row_ptr)
    # print("load_col_idx:", load_col_idx)
    # exit(0)

    # Binding Attn -------------------------------------
    binding_out = binding_attn_func(q, k, v,
                                full_row_ptr, full_col_idx, 
                                part_row_ptr, part_col_idx, inner_bitmaps, 
                                load_row_ptr, load_col_idx, 
                                dropout_p=dropout_p, causal=is_causal)    
    
    torch.cuda.synchronize()
    max_diff, mean_diff = check_tensor(binding_out, pt_out)
    
    print(f"[CHECK]  Binding_Attn\t  max_diff:{max_diff:.4f}  mean_diff:{mean_diff:.4f}" )

    # check_tensor_careful(binding_out, pt_out)
    
    