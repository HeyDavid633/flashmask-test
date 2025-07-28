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

