# 2025.11.18 For shepherding
# 对 MHA 的边界做尝试；
#  主要执行的应该是E2E 实验；完全在 Bert-base 上；
# 主要参考MHA是否打碎的代码为：https://github.com/HeyDavid633/fusion-SC25/blob/main/fx-fetch-demo/fused_cuda_attn.py
# 
# 收集了数据以后，用  Speedup Not break MHA vs broken MHA 乘以各个模型即可
#
import sys
import os

import argparse
import torch
import torch._dynamo
import math
import torch.nn.functional as F
from einops import rearrange, repeat
from util.utils import set_dtype, seqlen_to_mask, torch_cuda_identify, time_stamp_cudasync, transpose_for_scores
import random

def torch_attention_std(q, k, v):
    scores = torch.matmul(q, k.transpose(-2, -1)) / (q.shape[3] ** .5)
    probs = torch.nn.functional.softmax(scores, dim=-1)
    h = torch.matmul(probs, v).permute(0, 2, 1, 3).contiguous()
    return h

# 把 (q.shape[3] ** .5) 带来的一堆操作在此前做完； 
# 需要 call_method 的操作 key_transpose 在之前做完，使得Attn抓取更简单 - 这个提出来抓不到FA了
def torch_attention_std2(q, k_t, v, scale):
    scores = torch.matmul(q, k_t) / scale
    probs = torch.nn.functional.softmax(scores, dim=-1)
    h = torch.matmul(probs, v).permute(0, 2, 1, 3).contiguous()
    return h


def bert_fwd():
    with torch.no_grad():
        # hidden_states = from_tensor
        hidden_states = input
        for layer in range(layer_num):
            input_tensor = hidden_states
            qkv = torch.matmul(hidden_states, qkv_kernel[layer]) + qkv_bias[layer]

            q, k, v = qkv.chunk(3, dim=-1)
            q = transpose_for_scores(q, head_num, head_size)
            k = transpose_for_scores(k, head_num, head_size)
            v = transpose_for_scores(v, head_num, head_size)
            
            h = torch_attention_std(q, k, v)  # for FA capture test
            
            new_context_layer_shape = h.size()[:-2] + (hidden_dim, )
            hidden_states = h.view(new_context_layer_shape)

            hidden_states = torch.matmul(hidden_states, attr_output_kernel[layer]) + attr_output_bias[layer]

            hidden_states = hidden_states + input_tensor
            hidden_states = F.layer_norm(hidden_states, (hidden_dim, ),
                                         weight=attr_output_layernorm_gamma[layer], bias=attr_output_layernorm_beta[layer])
            residual = hidden_states

            hidden_states = torch.matmul(hidden_states, inter_kernel[layer]) + inter_bias[layer]
            hidden_states = F.gelu(hidden_states)
            hidden_states = torch.matmul(hidden_states, output_kernel[layer]) + output_bias[layer]

            hidden_states = hidden_states + residual
            hidden_states = F.layer_norm(hidden_states, (hidden_dim, ),
                                         weight=output_layernorm_gamma[layer], bias=output_layernorm_beta[layer])

            transformer_output[layer] = hidden_states
            
def bert_breakmha_fwd():
    with torch.no_grad():
        # hidden_states = from_tensor
        hidden_states = input
        for layer in range(layer_num):
            input_tensor = hidden_states
            qkv = torch.matmul(hidden_states, qkv_kernel[layer]) + qkv_bias[layer]

            q, k, v = qkv.chunk(3, dim=-1)
            q = transpose_for_scores(q, head_num, head_size)
            k = transpose_for_scores(k, head_num, head_size)
            v = transpose_for_scores(v, head_num, head_size)
            
            scale_num = head_size ** 0.5
            k_t = k.transpose(-2, -1).clone()

            h = torch_attention_std2(q, k_t, v, scale = scale_num)  # for FA capture test
            
            new_context_layer_shape = h.size()[:-2] + (hidden_dim, )
            hidden_states = h.view(new_context_layer_shape)

            hidden_states = torch.matmul(hidden_states, attr_output_kernel[layer]) + attr_output_bias[layer]

            hidden_states = hidden_states + input_tensor
            hidden_states = F.layer_norm(hidden_states, (hidden_dim, ),
                                         weight=attr_output_layernorm_gamma[layer], bias=attr_output_layernorm_beta[layer])
            residual = hidden_states

            hidden_states = torch.matmul(hidden_states, inter_kernel[layer]) + inter_bias[layer]
            hidden_states = F.gelu(hidden_states)
            hidden_states = torch.matmul(hidden_states, output_kernel[layer]) + output_bias[layer]

            hidden_states = hidden_states + residual
            hidden_states = F.layer_norm(hidden_states, (hidden_dim, ),
                                         weight=output_layernorm_gamma[layer], bias=output_layernorm_beta[layer])

            transformer_output[layer] = hidden_states


if __name__ == "__main__":
    torch.manual_seed(0)
    torch.cuda.empty_cache()
    device = torch_cuda_identify(print_info = True)
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    head_size = 64
    head_num = 12
    hidden_dim = head_size * head_num
    layer_num = 1

    data_type  = torch.float16
    dtype = "fp16"
    running_device = "cuda"
    
    qkv_kernel                  = [set_dtype(torch.zeros(hidden_dim, hidden_dim * 3).uniform_(-0.4, 0.4).cuda(), dtype) for _ in range(layer_num)]
    qkv_bias                    = [set_dtype(torch.zeros(hidden_dim * 3).uniform_(-0.4, 0.4).cuda(), dtype) for _ in range(layer_num)]
    attr_output_kernel          = [set_dtype(torch.zeros(hidden_dim, hidden_dim).uniform_(-0.4, 0.4).cuda(), dtype) for _ in range(layer_num)]
    attr_output_bias            = [set_dtype(torch.zeros(hidden_dim).uniform_(-0.4, 0.4).cuda(), dtype) for _ in range(layer_num)]
    attr_output_layernorm_gamma = [set_dtype(torch.zeros(hidden_dim).uniform_(-0.4, 0.4).cuda(), dtype) for _ in range(layer_num)]
    attr_output_layernorm_beta  = [set_dtype(torch.zeros(hidden_dim).uniform_(-0.4, 0.4).cuda(), dtype) for _ in range(layer_num)]
    inter_kernel                = [set_dtype(torch.zeros(hidden_dim, hidden_dim * 4).uniform_(-0.4, 0.4).cuda(), dtype) for _ in range(layer_num)]
    inter_bias                  = [set_dtype(torch.zeros(hidden_dim * 4).uniform_(-0.4, 0.4).cuda(), dtype) for _ in range(layer_num)]
    output_kernel               = [set_dtype(torch.zeros(hidden_dim * 4, hidden_dim).uniform_(-0.4, 0.4).cuda(), dtype) for _ in range(layer_num)]
    output_bias                 = [set_dtype(torch.zeros(hidden_dim).uniform_(-0.4, 0.4).cuda(), dtype) for _ in range(layer_num)]
    output_layernorm_gamma      = [set_dtype(torch.zeros(hidden_dim).uniform_(-0.4, 0.4).cuda(), dtype) for _ in range(layer_num)]
    output_layernorm_beta       = [set_dtype(torch.zeros(hidden_dim).uniform_(-0.4, 0.4).cuda(), dtype) for _ in range(layer_num)]
    transformer_output = [None for _ in range(layer_num)]

    warmup_iters = 10
    running_iters = 20 
    
    test_cases = [
        (1, 128),
        (8, 512), 
        (16, 2048)
    ]
    
    for batch_size, seq_len in test_cases:
        
        input = torch.empty(batch_size, seq_len, hidden_dim, dtype=torch.float16).cuda()
        torch.nn.init.normal_(input, -0.02, 0.02)
        torch_compiled_model_std = torch.compile(bert_fwd, mode='default', backend='inductor')
        
        # PyTorch Naive  ---------------------------------------
        for i in range(warmup_iters + running_iters):
            if i == warmup_iters:
                t0_start = time_stamp_cudasync()

            bert_fwd()
            torch_output = transformer_output[-1]

        t0_end = time_stamp_cudasync()
        base_time = (t0_end - t0_start) * 1000 / running_iters
        print("e2e | bs:{} | seq:{}  |  Torch Native  \t\t : {:.3f} ms / iter".format(batch_size, seq_len, base_time))
        
        # PyTorch Compile Not break  ---------------------------------------
        torch_compiled_model_std = torch.compile(bert_fwd, mode='default', backend='inductor')
        for i in range(warmup_iters + running_iters):
            if i == warmup_iters:
                t1_start = time_stamp_cudasync()

            torch_compiled_model_std()
            torch_compiled_output = transformer_output[-1]

        t1_end = time_stamp_cudasync()
        torch_compiled_time = (t1_end - t1_start) * 1000 / running_iters
        print("e2e | bs:{} | seq:{}  |  Not break MHA Compile\t : {:.3f} ms / iter".format(batch_size, seq_len, torch_compiled_time))
        
        # PyTorch Compile break MHA  ---------------------------------------
        torch_compiled_model_breakmha = torch.compile(bert_breakmha_fwd, mode='default', backend='inductor')
        for i in range(warmup_iters + running_iters):
            if i == warmup_iters:
                t2_start = time_stamp_cudasync()

            torch_compiled_model_breakmha()
            torch_compiled_breakmha_output = transformer_output[-1]
            
        t2_end = time_stamp_cudasync()
        torch_compiled_breakmha_time = (t2_end - t2_start) * 1000 / running_iters
        print("e2e | bs:{} | seq:{}  |  Torch break MHA Compile : {:.3f} ms / ite".format(batch_size, seq_len, torch_compiled_breakmha_time))
        
        print("Speedup Not break MHA vs broken MHA : {:.2f}\n".format(torch_compiled_breakmha_time / torch_compiled_time))
        
        


