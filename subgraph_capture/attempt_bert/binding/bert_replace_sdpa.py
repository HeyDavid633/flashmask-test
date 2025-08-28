# 2025.08.06 Wed  
# 在Huggingface的Bert前向中替换SPDA为我们的 
#  
# 参考 attempt_bert/extract_graph_bert.py 
# fusion-SC25/fx-fetch-demo/fused_my_kernel13.py
# flashmask-test/binding/benchmark3.py
import sys
import graph_net.torch 
import os
import argparse
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

import torch
import torch.fx as fx
from typing import List, Tuple

def replace_sdpa_nodes(gm: fx.GraphModule, mask: torch.Tensor) -> fx.GraphModule:
    """替换 GraphModule 中的 SDPA 节点为自定义实现"""
    graph = gm.graph
    nnz, full_row_ptr, full_col_idx, part_row_ptr, part_col_idx, load_row_ptr, load_col_idx, inner_bitmaps = get_OuterTile_storage(mask, 64, 64)

    # SDPA 操作符列表
    sdpa_patterns = [
        torch.ops.aten.scaled_dot_product_attention.default,
        # 可扩展其他变体
    ]

    sdpa_nodes = [
        node for node in graph.nodes
        if node.op == "call_function" and node.target in sdpa_patterns
    ]

    if not sdpa_nodes:
        print("No SDPA nodes found in the graph")
        return gm

    # 注册为模型 buffer，避免 inline 张量导致代码生成错误
    buffers_to_register = {
        "full_row_ptr": full_row_ptr.to(torch.int32),
        "full_col_idx": full_col_idx.to(torch.int32),
        "part_row_ptr": part_row_ptr.to(torch.int32),
        "part_col_idx": part_col_idx.to(torch.int32),
        "inner_bitmaps": inner_bitmaps.to(torch.uint64),
        "load_row_ptr": load_row_ptr.to(torch.int32),
        "load_col_idx": load_col_idx.to(torch.int32),
    }

    # 添加 buffers 到 GraphModule
    for name, buf in buffers_to_register.items():
        if not hasattr(gm, name):
            gm.register_buffer(name, buf)

    replaced_count = 0
    for node in sdpa_nodes:
        print(f">>> Found SDPA node: {node.name}, target={node.target}")

        q, k, v = node.args[:3]
        attn_mask = node.args[3] if len(node.args) > 3 else None
        dropout_p = node.args[4] if len(node.args) > 4 else 0.0
        is_causal = bool(node.args[5]) if len(node.args) > 5 else False

        try:
            with graph.inserting_before(node):
                # 获取已注册的 buffer 节点
                full_row_ptr_node = graph.get_attr("full_row_ptr")
                full_col_idx_node = graph.get_attr("full_col_idx")
                part_row_ptr_node = graph.get_attr("part_row_ptr")
                part_col_idx_node = graph.get_attr("part_col_idx")
                inner_bitmaps_node = graph.get_attr("inner_bitmaps")
                load_row_ptr_node = graph.get_attr("load_row_ptr")
                load_col_idx_node = graph.get_attr("load_col_idx")

                # 构造新函数调用的参数
                args = (
                    q, k, v,
                    full_row_ptr_node, full_col_idx_node,
                    part_row_ptr_node, part_col_idx_node,
                    inner_bitmaps_node,
                    load_row_ptr_node, load_col_idx_node,
                    dropout_p, is_causal
                )

                # 调用自定义函数
                new_node = graph.call_function(binding_attn_func, args=args)

            # 替换使用并删除旧节点
            node.replace_all_uses_with(new_node)
            graph.erase_node(node)
            replaced_count += 1

        except Exception as e:
            print(f"Failed to replace SDPA node {node.name}: {str(e)}")
            continue

    # 清理与验证
    graph.lint()
    gm.recompile()
    print(f"✅ Replaced {replaced_count} SDPA nodes with custom attention")
    return gm

def get_model_name():
    return "distilbert-base-uncased"    # 返回 Hugging Face 模型标识符

def create_model():
    model = AutoModel.from_pretrained(get_model_name())  # 加载预训练模型
    model.eval()            
    return model.to(running_device)

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
    
    input_args = (inputs["input_ids"], inputs["attention_mask"], inputs.get("token_type_ids", None))
    gm = torch.export.export(model, input_args).module()  # 返回的是 gm.forward 即捕获后的前向传播函数

    # gm.graph.print_tabular()
    
    # 替换SDPA节点
    # print("\nBefore SDPA replacement:")
    # gm.graph.print_tabular()
    
    gm = replace_sdpa_nodes(gm, mask)
    
    # print("\nAfter SDPA replacement:")
    # gm.graph.print_tabular()
    
    print("\nRunning FP16 inference with seqlen=128...")
    output = gm(*input_args)
    print("Inference finished. Output shape:", output.last_hidden_state.shape)
    
    # 保存替换前后的计算图对比
    with open(get_model_name()+"_graph_before.txt", "w") as f:
        gm_before = torch.export.export(model, input_args).module()
        f.write(str(gm_before.graph))
    
    with open(get_model_name()+"_graph_after.txt", "w") as f:
        f.write(str(gm.graph))
    
