# 2025.07.07 Mon.
# 
# 验证生成的mask是否形状正确
# 需要验证 bitmap 的编解码过程
# Casual | Sliding | Longformer | BigBird
# 
# python test_mask.py --batch_size 1 --head_num 1 --head_size 64 --seq_len 256 --mask_id 0
import argparse
import torch
from util.utils import set_dtype, seqlen_to_mask
from util.masks import generate_causal_mask, generate_sliding_mask, generate_dilated_mask,generate_longformer_mask, generate_bigbird_mask, get_sparse_storage, plot_mask_as_blocks, generate_full_mask
from ops.package_op import block_attn_mask_func
from ops.package_op import binding_attn_func 
import matplotlib.pyplot as plt
import math
import numpy as np




def torch_attn_std(q, k, v, mask=None):
    scores = torch.matmul(q, k.transpose(-2, -1)) / (q.shape[3] ** .5)
    if mask != None:
        scores -= 10000.0 * (1.0 - mask.unsqueeze(1))
    probs = torch.nn.functional.softmax(scores, dim=-1)
    h = torch.matmul(probs, v)
    return h

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
    
def check_tensor(cuda_output, torch_output):
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


if __name__ == "__main__":
    torch.random.manual_seed(0)
    
    parser = argparse.ArgumentParser(description="Give the parameters for the attention test (with Mask)")
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size (default: 1)')
    parser.add_argument('--head_num', type=int, default=1, help='Number of heads (default: 1)')
    parser.add_argument('--head_size', type=int, default=64, help='Head size (default: 64)')
    parser.add_argument('--seq_len', type=int, default=64, help='Sequence length (default: 64)')
    parser.add_argument('--mask_id', type=int, default=0, help='Mask type: 0-Casual | 1-Sliding | 2-Longformer | 3-BigBird (default: 0)')
    parser.add_argument('--block_m', type=int, default=16, help='Block Size of M (default:32)')
    parser.add_argument('--block_n', type=int, default=16, help='Block Size of N (default:32)')
    parser.add_argument('--num_warps', type=int, default=1, help='Warp Num to launch (default:4)')
    args = parser.parse_args() 
    
    batch_size = args.batch_size
    head_num   = args.head_num
    head_size  = args.head_size
    seq_len    = args.seq_len
    mask_id    = args.mask_id
    BLOCK_M    = args.block_m
    BLOCK_N    = args.block_n
    num_warps  = args.num_warps
    dropout_p  = 0.0
     
    if(num_warps > (BLOCK_M/16) * (BLOCK_N/16)):
        print(f"num_warps: {num_warps}, (BLOCK_M/16) * (BLOCK_N/16): {int(BLOCK_M / 16) * int(BLOCK_N / 16)}")
        print("Error! Here should be: num_warps <= (BLOCK_M/16) * (BLOCK_N/16) !")
        exit(0)
    
    data_type  = torch.float16
    running_device = "cuda"
    sqrt_seq_len = int(math.sqrt(seq_len))
    fill_rate    = 0.1
    

    avg_seq_len = seq_len
    low, high = (2 * avg_seq_len - seq_len, seq_len + 1)
    input_lens = torch.randint(low=low, high=high, size=(batch_size,))
    seqlen_mask = seqlen_to_mask(input_lens, seq_len)
    attr_mask   = set_dtype(torch.tile(seqlen_mask, dims=(seq_len,)).reshape(batch_size, seq_len, seq_len).cuda(), "fp16")
    
    
    if(mask_id == 0):
        is_causal = True
        mask_name = 'Causal_Mask'
        mask = generate_causal_mask(attr_mask).cuda()
    elif(mask_id == 1):
        mask_name = 'Sliding_Mask'
        mask = generate_sliding_mask(attr_mask, bandwidth=BLOCK_M, is_cauasl=True).cuda()
    elif(mask_id == 2):
        mask_name = 'Longformer_Mask'
        mask = generate_longformer_mask(attr_mask, globalwidth=BLOCK_M, bandwidth=BLOCK_M, is_cauasl=False).cuda()
    elif(mask_id == 3):
        mask_name = 'BigBird_Mask'
        mask = generate_bigbird_mask(attr_mask, globalwidth=BLOCK_M, bandwidth=BLOCK_M, fill_rate=fill_rate, is_cauasl=False).cuda()
    elif(mask_id == 4):
        mask_name = 'dilated_Mask'
        mask = generate_dilated_mask(attr_mask, bandwidth=BLOCK_M - 1, dilation_rate=1, is_cauasl=True).cuda()
    elif(mask_id == 5):
        mask_name = 'Test_full_Mask'
        mask = generate_full_mask(attr_mask).cuda()
        
    
    print(mask_name, "=", mask.shape)
    # print(mask)
    # plot_mask_as_blocks(mask, mask_name + f'_{seq_len}', seq_len)
    # exit(0)
    
   
    nnz, full_row_ptr, full_col_idx, part_row_ptr, part_col_idx, part_block_mask, load_row_ptr, load_col_idx = get_sparse_storage(mask, BLOCK_M, BLOCK_N)
    
    # print(f"nnz: {nnz:.2f}%")
    # print("full_row_ptr:", full_row_ptr)
    # print("full_col_idx:", full_col_idx)
    # print("part_row_ptr:", part_row_ptr)
    # print("part_col_idx:", part_col_idx)
    # print("part_block_mask:\n", part_block_mask)
    
    # print("load_row_ptr:", load_row_ptr)
    # print("load_col_idx:", load_col_idx)
    # exit(0)
    
    # torch.set_printoptions(profile="full")
    # torch.set_printoptions(profile="default")
    # torch.set_printoptions(precision=4, sci_mode=False) # 禁用科学计数法，并设置4位精度   
    
    # query = torch.ones(batch_size, head_num, seq_len, head_size, device=running_device, dtype=data_type)

    # for i in range(seq_len):
    #     query[0, 0, i, :] = (1 + i)   
    # # print(query)
    # # exit(0)
    
    # key = torch.ones(batch_size, head_num, seq_len, head_size, device=running_device, dtype=data_type)
    
    # for i in range(32, seq_len):
    #     key[0, 0, i, :] = 1.001
    #     key[0, 0, i, :] = (1 + 0.01 * i)   
    # # print(key)
    # # exit(0)
    
    # value = torch.ones(batch_size, head_num, seq_len, head_size, device=running_device, dtype=data_type)

    
    
    # query = torch.ones(batch_size, head_num, seq_len, head_size, device=running_device, dtype=data_type)
    # key = torch.ones(batch_size, head_num, seq_len, head_size, device=running_device, dtype=data_type)
    # value = torch.ones(batch_size, head_num, seq_len, head_size, device=running_device, dtype=data_type)

    query = torch.randn(batch_size, head_num, seq_len, head_size, device=running_device, dtype=data_type, requires_grad=False)
    key = torch.randn(batch_size, head_num, seq_len, head_size, device=running_device, dtype=data_type, requires_grad=False)
    value = torch.randn(batch_size, head_num, seq_len, head_size, device=running_device, dtype=data_type, requires_grad=False)
    
    query1 = query.clone()
    
    q = query.permute(0, 2, 1, 3)
    k = key.permute(0, 2, 1, 3)
    v = value.permute(0, 2, 1, 3)

    print(" query, key, value shape =", query.shape)
    
    result = block_attn_mask_func(query, key, value,
                                full_row_ptr, full_col_idx, 
                                part_row_ptr, part_col_idx, part_block_mask, 
                                load_row_ptr, load_col_idx,
                                BLOCK_M, BLOCK_N, num_warps)
    cuda_output = result
    
    # result = binding_attn_func(q, k, v,
    #                             full_row_ptr, full_col_idx, 
    #                             part_row_ptr, part_col_idx, part_block_mask, 
    #                             load_row_ptr, load_col_idx, 
    #                             dropout_p=dropout_p, causal=is_causal)
    # cuda_output = result.permute(0, 2, 1, 3)

    torch_output = torch_attn_std(query1, key, value, mask=mask)
    
 
    # print("cuda_output: ", cuda_output)
    # print("torch_output: ", torch_output)
    
    # print(f"\n Output max diff: {(cuda_output - torch_output).abs().max().item()}")
    # print(f" Output mean diff: {(cuda_output - torch_output).abs().mean().item()}")
    
    check_tensor(cuda_output, torch_output)
    # check_tensor_plot(cuda_output, torch_output)
    
    
    # with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
    #     FA2_output = torch.nn.functional.scaled_dot_product_attention(query,key,value, is_causal=False)
                        
    # # h = FA2_output.permute(0, 2, 1, 3).contiguous() 
    # # new_context_layer_shape = h.size()[:-2] + (query.shape[1]*query.shape[3], )
    # # hidden_states0 = h.view(new_context_layer_shape) 
    
    # # print("FA2_output shape: ", FA2_output.shape)
    
    # # check_tensor(FA2_output, torch_output)
    

    
    
    
    