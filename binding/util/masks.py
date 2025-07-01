# 
#  SC25-fusion Mask
#  Casual | Sliding | Longformer | BigBird
# 
import math
import torch
import matplotlib.pyplot as plt

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

def sequence_mask(lengths, max_len=None, is_2d=True):
    batch_size = lengths.numel()
    max_len = max_len or lengths.max()
    mask = (torch.arange(0, max_len, device=lengths.device)
            .type_as(lengths)
            .repeat(batch_size, 1)
            .lt(lengths.unsqueeze(1)))
    if is_2d:
        return mask
    else:
        mask = mask.view(-1, 1, 1, max_len)
        m2 = mask.transpose(2, 3)
        return mask * m2

def seqlen_to_mask(lengths, max_len):
    batch_size = lengths.numel()
    mask = (torch.arange(0, max_len, device=lengths.device)
            .type_as(lengths)
            .repeat(batch_size, 1)
            .lt(lengths.unsqueeze(1)))
    return mask

def generate_triangle_mask(attr_mask):
    # gernerate lower triangle mask
    seq_len = attr_mask.shape[1]
    triangle_mask = torch.tril(torch.ones(seq_len, seq_len))
    triangle_mask = triangle_mask.unsqueeze(0).repeat(attr_mask.shape[0], 1, 1)
    return triangle_mask


def generate_strided_mask(attr_mask):
    # gernerate stride mask
    stride_step = int(math.sqrt(attr_mask.shape[1]))
    seq_len = attr_mask.shape[1]
    strided_mask = torch.zeros_like(attr_mask)
    
    for batch in range(strided_mask.shape[0]):
        for i in range(seq_len):
            for j in range(i+1):
                if((i - j) % stride_step == 0):
                    strided_mask[batch, i, j] = 1.0  
                if(j > i - stride_step):
                    strided_mask[batch, i, j] = 1.0  
    return strided_mask
    
   
def generate_fixed_mask(attr_mask):
    # gernerate stride mask
    fixed_step = int(math.sqrt(attr_mask.shape[1]))
    seq_len = attr_mask.shape[1]
    fixed_mask = torch.zeros_like(attr_mask)
    
    for batch in range(fixed_mask.shape[0]):
        for i in range(seq_len):
            for j in range(i+1):
                if(j % fixed_step == fixed_step-1):
                    fixed_mask[batch, i, j] = 1.0  
                if(j > i + (j % fixed_step) - fixed_step):
                    fixed_mask[batch, i, j] = 1.0  
    return fixed_mask


# 对于atomic系列，实际上应确保不越界，即参数应该小于seq_len，但此处没有注意
# atomic mask(a) global attention
def atomic_a_global(attr_mask, globalwidth = 1):
    batch_size = attr_mask.shape[0]
    seq_len = attr_mask.shape[1]
    global_mask = torch.zeros_like(attr_mask)
    
    for batch in range(batch_size):
        global_mask[batch, :globalwidth, :] = 1  
        global_mask[batch, :, :globalwidth] = 1  
    
    # 刷成下三角的 
    for i in range(seq_len-1):
        global_mask[batch, i, i+1:] = 0
            
    return global_mask


# atomic mask(b) band attention
def atomic_b_band(attr_mask, bandwidth = 1):
    batch_size = attr_mask.shape[0]
    seq_len = attr_mask.shape[1]
    band_mask = torch.zeros_like(attr_mask)

    for batch in range(batch_size):
        for i in range(seq_len):
            start = max(0, i - bandwidth)  # 确保起点不越界
            end = min(seq_len, i + bandwidth + 1)  # 确保终点不越界
            band_mask[batch, i, start:end] = 1
    
    for i in range(seq_len-1):
        band_mask[batch, i, i+1:] = 0        

    return band_mask


# atomic mask(c) dilated attention
def atomic_c_dilated(attr_mask, bandwidth = 1, dilation_rate = 1):
    batch_size = attr_mask.shape[0]
    seq_len = attr_mask.shape[1]
    dilated_mask = torch.zeros_like(attr_mask)

    for batch in range(batch_size):
        for i in range(seq_len):
            start = i - bandwidth - dilation_rate  # 需确保起点不越界
            end = min(seq_len, i + bandwidth + dilation_rate + 1)  # 确保终点不越界
            for row_idx in range(start, end, dilation_rate + 1):
                if(row_idx > -1):
                    dilated_mask[batch, i, row_idx] = 1
                    
    for i in range(seq_len-1):
        dilated_mask[:, i, i+1:] = 0
            
    return dilated_mask


# atomic mask(d) block local attention
def atomic_d_block(attr_mask, block_size = 2):
    
    batch_size = attr_mask.shape[0]
    seq_len = attr_mask.shape[1]
    block_mask = torch.zeros_like(attr_mask)

    for batch in range(batch_size):
        num_blocks = seq_len // block_size 
        for i in range(num_blocks):
            start = i * block_size
            end = start + block_size
            block_mask[batch, start:end, start:end] = 1
            
    for i in range(seq_len-1):
        block_mask[:, i, i+1:] = 0

    return block_mask

def generate_full_mask(attr_mask):
    full_mask = torch.ones_like(attr_mask)
    return full_mask


# SC25-fusion Mask ---------------------------------------
def generate_causal_mask(attr_mask):
    seq_len = attr_mask.shape[1]
    casual_mask = torch.tril(torch.ones(seq_len, seq_len))
    casual_mask = casual_mask.unsqueeze(0).repeat(attr_mask.shape[0], 1, 1)
    return casual_mask


def generate_sliding_mask(attr_mask, bandwidth=1, is_cauasl=False):
    batch_size = attr_mask.shape[0]
    seq_len = attr_mask.shape[1]
    sliding_mask = torch.zeros_like(attr_mask)

    for batch in range(batch_size):
        for i in range(seq_len):
            start = max(0, i - bandwidth)  
            end = min(seq_len, i + bandwidth + 1)  
            sliding_mask[batch, i, start:end] = 1
    
    if(is_cauasl == True):
        for i in range(seq_len-1):
            sliding_mask[batch, i, i+1:] = 0        

    return sliding_mask


def generate_dilated_mask(attr_mask, bandwidth=1, dilation_rate=1, is_cauasl=False):
    batch_size = attr_mask.shape[0]
    seq_len = attr_mask.shape[1]
    dilated_mask = torch.zeros_like(attr_mask)

    for batch in range(batch_size):
        for i in range(seq_len):
            start = i - bandwidth - dilation_rate 
            end = min(seq_len, i + bandwidth + dilation_rate + 1) 
            for row_idx in range(start, end, dilation_rate + 1):
                if(row_idx > -1):
                    dilated_mask[batch, i, row_idx] = 1
    
    if(is_cauasl == True):
        for i in range(seq_len-1):
            dilated_mask[batch, i, i+1:] = 0     
            
    return dilated_mask


def generate_longformer_mask(attr_mask, globalwidth=1, bandwidth=1, is_cauasl=False):
    batch_size = attr_mask.shape[0]
    seq_len = attr_mask.shape[1]
    longformer_mask = torch.zeros_like(attr_mask)
    
    for batch in range(batch_size):
        longformer_mask[batch, :globalwidth, :] = 1  
        longformer_mask[batch, :, :globalwidth] = 1  
        
    for batch in range(batch_size):
        start = int(seq_len / 2)
        end = int(seq_len / 2 + globalwidth)
        
        longformer_mask[batch, start:end, :] = 1  
        longformer_mask[batch, :, start:end] = 1  

    for batch in range(batch_size):
        for i in range(seq_len):
            start = max(0, i - bandwidth)  
            end = min(seq_len, i + bandwidth + 1)  
            longformer_mask[batch, i, start:end] = 1
    
    if(is_cauasl == True):
        for i in range(seq_len-1):
            longformer_mask[batch, i, i+1:] = 0        

    return longformer_mask


import random

def generate_bigbird_mask(attr_mask, globalwidth=1, bandwidth=1, fill_rate=0.2, is_cauasl=False):
    batch_size = attr_mask.shape[0]
    seq_len = attr_mask.shape[1]
    bigbird_mask = torch.zeros_like(attr_mask)
    
    for batch in range(batch_size):
        bigbird_mask[batch, :globalwidth, :] = 1  
        bigbird_mask[batch, :, :globalwidth] = 1  
        
    for batch in range(batch_size):
        for i in range(seq_len):
            start = max(0, i - bandwidth)  
            end = min(seq_len, i + bandwidth + 1)  
            bigbird_mask[batch, i, start:end] = 1
    
    num_ones_in_block = int(1024 * fill_rate)
    random.seed(0)
    
    for batch in range (batch_size):
        for i in range (int(seq_len / 32) - 1):
            start_x = i * 32
            if(i == 0):
                start_y = start_x + 32 * random.choice([1, 0])
            elif(i == int(seq_len / 32) - 2):
                start_y = start_x + 32 * random.choice([-1, 0])
            else:
                start_y = start_x + 32 * random.choice([-1, 0, 1])
            for j in range(num_ones_in_block):
                random_x = random.randint(0, 32)
                random_y = random.randint(0, 32)
                bigbird_mask[batch, start_x + random_x, start_y + random_y] = 1
    
    if(is_cauasl == True):
        for i in range(seq_len-1):
            bigbird_mask[batch, i, i+1:] = 0        

    return bigbird_mask


def get_sparse_storage(Mask, block_size_m, block_size_n):
    n = Mask.shape[-1]
    
    # 初始化存储列表
    full_row_ptr = []
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


from functools import lru_cache
from torch.nn.attention.flex_attention import create_block_mask

@lru_cache
def create_block_mask_cached(score_mod, B, H, M, N, device="cuda"):
    block_mask = create_block_mask(score_mod, B, H, M, N, device=device)
    return block_mask

def flex_causal_mask(b, h, q_idx, kv_idx):
    return q_idx >= kv_idx

def flex_sliding_window_mask(b, h, q_idx, kv_idx):
    causal_mask = q_idx >= kv_idx
    windowed_mask = (
        q_idx - kv_idx <= 256
    )  
    # return causal_mask & windowed_mask
    return windowed_mask

def flex_longformer_mask(b, h, q_idx, kv_idx):
    windowed_mask = (
        q_idx - kv_idx <= 256
    )  
    causal_mask = q_idx >= kv_idx

    global_band1 = (q_idx < 32) | (kv_idx < 32)
    global_band2 = ((96 <= q_idx) & (q_idx < 128)) | ((96 <= kv_idx) & (kv_idx < 128))
    global_mask = global_band1 | global_band2
    
    longformer_mask = windowed_mask | global_mask
    return longformer_mask

def flex_bigbird_mask(b, h, q_idx, kv_idx):
    windowed_mask = (
        q_idx - kv_idx <= 256
    )  
    causal_mask = q_idx >= kv_idx
    
    global_band1 = (q_idx < 32) | (kv_idx < 32)
    global_band2 = ((96 <= q_idx) & (q_idx < 128)) | ((96 <= kv_idx) & (kv_idx < 128))
    global_mask = global_band1 | global_band2
    
    random_cond1 = (q_idx % 3 == 0) & (kv_idx / 5 == 0)
    random_cond2 = (q_idx % 4 == 0) & (kv_idx / 7 == 0)
    random_mask = random_cond1 | random_cond2
    
    bigbird_mask = windowed_mask | global_mask | random_mask
    return bigbird_mask


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