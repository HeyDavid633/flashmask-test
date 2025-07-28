# 2024.10.19 Sat.
#
# python fwd-compile-print.py
# 
# 打印 更多融合debug信息和依赖图svg
# TORCH_LOGS=fusion TORCH_COMPILE_DEBUG=1 INDUCTOR_ORIG_FX_SVG=1 INDUCTOR_POST_FUSION_SVG=1 python fwd-compile-print-withmask.py

import torch
import numpy as np
import random
import torch.nn.functional as F
import torch._dynamo
from util.utils import transpose_for_scores, seqlen_to_mask, set_dtype, torch_cuda_identify
from util.masks import generate_triangle_mask, generate_strided_mask, generate_fixed_mask

def fwd_bert_std():
    hidden_states = input_from_tensor
    # for layer in range(layer_num):
    layer = 0
    input_tensor = hidden_states

    qkv = torch.matmul(hidden_states, qkv_kernel[layer]) + qkv_bias[layer]
    q, k, v = qkv.chunk(3, dim=-1)
    q = transpose_for_scores(q, head_num, head_size)
    k = transpose_for_scores(k, head_num, head_size)
    v = transpose_for_scores(v, head_num, head_size)

    # ------------------------------------------------------------- Attention start
    # (B, H, S, W) @ (B, H, W, S) -> (B, H, S, S) -softmax-> (B, H, S, S)
    scores = torch.matmul(q, k.transpose(-2, -1)) / (head_size ** .5)
    scores -= 10000.0 * (1.0 - mask.unsqueeze(1))  
    probs = F.softmax(scores, dim=-1)
    # (B, H, S, S) @ (B, H, S, W) -> (B, H, S, W) -trans-> (B, S, H, W)
    h = torch.matmul(probs, v)
    # ------------------------------------------------------------ Attention End 
    
    h = h.permute(0, 2, 1, 3).contiguous()
    # -merge-> (B, S, D)
    new_context_layer_shape = h.size()[:-2] + (hidden_dim, )
    hidden_states = h.view(new_context_layer_shape)
    
    #attention output projection GEMM
    hidden_states = torch.matmul(hidden_states, attr_output_kernel[layer]) + attr_output_bias[layer]
    hidden_states = hidden_states + input_tensor  # 残差连接
    
    # layer_Norm
    hidden_states = F.layer_norm(hidden_states, (hidden_dim, ), weight=attr_output_layernorm_gamma[layer], bias=attr_output_layernorm_beta[layer])
    
    residual = hidden_states       # 为残差连接做好准备
    #FFN GEMM 1 + add bias 
    hidden_states = torch.matmul(hidden_states, inter_kernel[layer]) + inter_bias[layer] 
    hidden_states = F.gelu(hidden_states)  #激活函数
    #FFN GEMM 2 + add bias
    hidden_states = torch.matmul(hidden_states, output_kernel[layer]) + output_bias[layer]  
    hidden_states = hidden_states + residual  #残差连接

    # layer_Norm
    hidden_states = F.layer_norm(hidden_states, (hidden_dim, ),  weight=output_layernorm_gamma[layer], bias=output_layernorm_beta[layer])  
    transformer_output[layer] = hidden_states
        
        

    
if __name__ == '__main__':
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)
    
    torch_cuda_identify()
    
    batch_size = 1
    head_size = 64 # head_dim aka. head_size
    seq_len = 128
    head_num = 12
    layer_num = 1
    hidden_dim = head_num * head_size

    dtype = "fp16"
    
    # 4类Atom mask叠加组合成5种Mask
    avg_seq_len = seq_len
    low, high = (2 * avg_seq_len - seq_len, seq_len + 1)
    input_lens = torch.randint(low=low, high=high, size=(batch_size,))
    seqlen_mask = seqlen_to_mask(input_lens, seq_len)
    attr_mask   = set_dtype(torch.tile(seqlen_mask, dims=(seq_len,)).reshape(batch_size, seq_len, seq_len).cuda(), dtype)
    
    lower_triangle_mask = generate_triangle_mask(attr_mask).cuda()
    strided_mask = generate_strided_mask(attr_mask).cuda()
    fixed_mask = generate_fixed_mask(attr_mask).cuda()
    
    mask = lower_triangle_mask
    
    
    input_from_tensor           = set_dtype(torch.empty(batch_size, seq_len, hidden_dim).uniform_(-0.4, 0.4).cuda(), dtype)
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
    
    
    # TorchInductor 跟踪: 显示每个 TorchInductor 阶段所花费的时间 + 输出代码和图可视化
    torch._inductor.config.trace.enabled = True
    
    # torch.compile --- mode:default
    torch._dynamo.reset()
    fwd_compile_default = torch.compile(fwd_bert_std, mode="default")
    fwd_compile_default()
    
    
    # torch.compile --- mode:max-autotune
    # torch._dynamo.reset()
    # fwd_compile_autotune = torch.compile(fwd_bert_std, mode="max-autotune")
    # fwd_compile_autotune()
        


    
    
    
