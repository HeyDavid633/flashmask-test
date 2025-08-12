# 2025.08.06 Wed  
# 在Huggingface的Bert前向中替换SPDA为我们的 
#  
# 由于在 bert_replace_sdpa.py 提取不顺利，故在这里只提取单个 SDPA
import os
import argparse
import sys
import copy
import torch

import math
import torch.nn.functional as F

from einops import rearrange, repeat
from transformers import AutoModel, AutoTokenizer # Hugging Face 模型和分词器
from prettytable import PrettyTable 
from typing import List

from util.masks import generate_causal_mask, generate_sliding_mask, generate_dilated_mask,generate_longformer_mask, generate_bigbird_mask, generate_full_mask
from util.masks import create_block_mask_cached, flex_causal_mask, flex_longformer_mask, flex_sliding_window_mask, flex_bigbird_mask 

from ops.package_op import binding_attn_func  # Binded FA2
from util.utils import set_dtype, seqlen_to_mask, torch_cuda_identify, time_stamp_cudasync
from util.utils import bitmap_to_matrix, print_tile_structure, get_InnerTile_bitmap, get_OuterTile_storage

def replace_sdpa_nodes(gm: torch.fx.GraphModule) -> torch.fx.GraphModule:
    """替换GraphModule中的SDPA节点为自定义实现"""
    fx_graph = gm.graph
    
    # 扩展SDPA节点匹配模式
    sdpa_patterns = [
        torch.ops.aten.scaled_dot_product_attention.default,
        # torch.ops.aten._scaled_dot_product_efficient_attention.default,
        # torch.nn.functional.scaled_dot_product_attention
    ]
    
    # 查找所有SDPA节点
    sdpa_nodes = []
    for node in fx_graph.nodes:
        if node.op == "call_function" and node.target in sdpa_patterns:
            sdpa_nodes.append(node)
    
    if not sdpa_nodes:
        print("No SDPA nodes found in the graph")
        return gm
    

    node = sdpa_nodes[0]

    print(f">>> Found attention structure: {node.name}, {node.target}")

    q, k, v = node.args[:3]
    attn_mask = None
    dropout_p = 0.0
    is_causal = False
    
    # 提取额外参数
    if len(node.args) > 3:
        attn_mask = node.args[3]
    if len(node.args) > 4:
        dropout_p = node.args[4]
    if len(node.args) > 5:
        is_causal = bool(node.args[5])
    
    try:    
        # 创建新节点
        with fx_graph.inserting_before(node):
            new_node = fx_graph.call_function(
                binding_attn_func,
                args=(q, k, v, full_row_ptr, full_col_idx, part_row_ptr, 
                        part_col_idx, inner_bitmaps, load_row_ptr, load_col_idx,
                        dropout_p, is_causal)
            )
        
        # 替换并删除旧节点
        node.replace_all_uses_with(new_node)
        fx_graph.erase_node(node)
            
    except Exception as e:
        print(f">>> Failed to replace SDPA node {node}: {str(e)}")

    # 验证并重新编译
    fx_graph.lint()
    gm.recompile()
    print(f"Replaced {len(sdpa_nodes)} SDPA nodes with custom implementation")
    return gm



def get_model_name():
    return "distilbert-base-uncased"    # 返回 Hugging Face 模型标识符

def create_model():
    model = AutoModel.from_pretrained(get_model_name())  # 加载预训练模型
    model.eval()             # 设置为评估模式（关闭 Dropout 等训练专用层，禁用训练阶段的随机行为， 确保推理结果稳定）
    return model.to(running_device)  # 将模型移动到指定设备（GPU/CPU）

def print_model_info(model, inputs):
    """打印模型关键信息的表格"""
    table = PrettyTable()
    table.field_names = ["Key", "Value"]
    table.align["Key"] = "l"
    table.align["Value"] = "l"
    
    # 从模型配置中获取参数
    config = model.config
    table.add_row(["Model Name", get_model_name()])
    table.add_row(["Batch Size", inputs["input_ids"].shape[0]])
    table.add_row(["Sequence Length", inputs["input_ids"].shape[1]])
    table.add_row(["Hidden Dimension", config.dim])
    table.add_row(["Number of Layers", config.n_layers])
    table.add_row(["Number of Attention Heads", config.n_heads])
    table.add_row(["Intermediate Size (FFN)", config.hidden_dim])
    table.add_row(["Vocab Size", config.vocab_size])
    table.add_row(["Max Position Embeddings", config.max_position_embeddings])
    
    print("\n" + "="*50)
    print("Model Architecture Information")
    print("="*50)
    print(table)
    print("="*50 + "\n")

if __name__ == '__main__':    
    torch.manual_seed(0)
    torch.cuda.empty_cache()
    running_device = torch_cuda_identify(print_info = False)
    dtype = torch.float16

    tokenizer = AutoTokenizer.from_pretrained(get_model_name()) # 加载与模型匹配的分词器
    text = "Hello world "
    inputs = tokenizer(text, max_length=128, padding='max_length', truncation=True, return_tensors="pt")

    inputs = {k: v.to(running_device) for k, v in inputs.items()}

    model = create_model().half()
    # print_model_info(model, inputs)
    config = model.config
    batch_size, seqlen, nheads, headdim = inputs["input_ids"].shape[0], inputs["input_ids"].shape[1], config.n_heads, config.hidden_dim

    parser = argparse.ArgumentParser(description="Give the parameters for the attention test (with Mask)")
    parser.add_argument('--mask_id', type=int, default=0, help='Mask type: 0-Casual | 1-Sliding | 2-Longformer | 3-BigBird (default: 0)')
    args = parser.parse_args()
    mask_id = args.mask_id

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
    
    nnz, full_row_ptr, full_col_idx, part_row_ptr, part_col_idx, load_row_ptr, load_col_idx, inner_bitmaps = get_OuterTile_storage(mask, 64, 64)
    q = torch.randn(batch_size, seqlen, nheads, headdim, device=running_device, dtype=dtype, requires_grad=False)
    k = torch.randn(batch_size, seqlen, nheads, headdim, device=running_device, dtype=dtype, requires_grad=False)
    v = torch.randn(batch_size, seqlen, nheads, headdim, device=running_device, dtype=dtype, requires_grad=False)

    input_args = (inputs["input_ids"], inputs["attention_mask"], inputs.get("token_type_ids", None))
    gm = torch.export.export(model, input_args).module()  # 返回的是 gm.forward 即捕获后的前向传播函数
    # gm.graph.print_tabular()
    
    # 替换SDPA节点
    # print("\nBefore SDPA replacement:")
    # gm.graph.print_tabular()
    
    gm = replace_sdpa_nodes(gm)
    
    # print("\nAfter SDPA replacement:")
    gm.graph.print_tabular()
    
    # print("\nRunning FP16 inference with seqlen=128...")
    # output = gm(*input_args)
    # print("Inference finished. Output shape:", output.last_hidden_state.shape)
    
    # 保存替换前后的计算图对比
    # with open("graph_before.txt", "w") as f:
    #     gm_before = torch.export.export(model, input_args).module()
    #     f.write(str(gm_before.graph))
    
    # with open("graph_after.txt", "w") as f:
    #     f.write(str(gm.graph))
    
