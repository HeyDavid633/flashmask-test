# some basic function 

import torch 
import math
import timeit
import re

def torch_cuda_identify(print_info = True):        
    if torch.cuda.is_available():
        if print_info:
            print(' PyTorch version:', torch.__version__)
            print(' CUDA version \t:', torch.version.cuda)
            print(' GPU cuda:({}) \t: {}'.format(torch.cuda.current_device(), torch.cuda.get_device_name()),'\n', "-" * 50)
        return torch.device('cuda:{}'.format(torch.cuda.current_device()))
    else:
        print('cuda is not avaliable !')
        return torch.device('cpu')
    
def time_stamp_cudasync():
    torch.cuda.synchronize()
    return timeit.default_timer()   
    
def set_dtype(ts: torch.Tensor, dtype: str):
    if dtype == "fp32":
        return ts.float()
    elif dtype == "fp16":
        return ts.half()
    raise RuntimeError(f"Unsupported dtype {dtype}")

def transpose_for_scores(x, n_heads, head_size):
    # (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)
    
    # 取x的除最后一个维度外的所有维度 加完了以后 = (batch_size, seq_len, n_heads, head_size)
    new_x_shape = x.size()[:-1] + (n_heads, head_size)
    x = x.view(new_x_shape)
    # x的维度变化 (batch_size, seq_len, hidden_dim) --- (batch_size, head_num, seq_len, head_size)
    # 自动的拆开了 最后一个维度 hidden_dim
    return x.permute(0, 2, 1, 3)

def transpose_for_scores1(x):
    new_x_shape = x.size()[:-1] + (12, 64)
    x = x.view(new_x_shape)
    return x.permute(0, 2, 1, 3)

def seqlen_to_mask(lengths, max_len):
    batch_size = lengths.numel()
    mask = (torch.arange(0, max_len, device=lengths.device)
            .type_as(lengths)
            .repeat(batch_size, 1)
            .lt(lengths.unsqueeze(1)))
    return mask


def read_config_file(file_path):
    """
    从给定路径的文本文件中解析配置信息。
    :param file_path: 配置文件的路径。
    :return: 包含所有配置信息的字典。
    """
    config_data = {}
    with open(file_path, 'r') as f:
        for line in f.readlines():
            # 使用正则表达式匹配每一行的数据
            match = re.match(r'num_warp(\d+)\s*\|\s*m(\d+)n(\d+)\s*\|\s*bs:(\d+)\s*\|\s*seq:(\d+)\s*\|\s*([\d.]+)\s*ms/iter\s*\|\s*Speedup/FA2:\s*([\d.]+)', line)
            if match:
                num_warps = int(match.group(1))
                block_m = int(match.group(2))
                block_n = int(match.group(3))
                bs = int(match.group(4))
                seq_len = int(match.group(5))
                
                if bs not in config_data:
                    config_data[bs] = {}
                if seq_len not in config_data[bs]:
                    config_data[bs][seq_len] = []
                    
                config_data[bs][seq_len].append((block_m, block_n, num_warps))
    
    return config_data


def get_best_config(config_data, batch_size, seq_len):
    """
    根据给定的 batch_size 和 seq_len 获取最佳配置。
    :param config_data: 解析后的配置数据。
    :param batch_size: 目标 batch size。
    :param seq_len: 目标序列长度。
    :return: (BLOCK_M, BLOCK_N, num_warps) 的元组。
    """
    if batch_size in config_data and seq_len in config_data[batch_size]:
        configs = config_data[batch_size][seq_len]
        # 假设列表中的第一个配置就是最佳配置（因为示例数据中每个bs和seq_len组合只有一个配置）
        return configs[0]
    else:
        raise ValueError(f"没有找到 batch_size={batch_size}, seq_len={seq_len} 的配置")



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

