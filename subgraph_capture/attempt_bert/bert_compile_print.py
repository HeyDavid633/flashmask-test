# 2025.08.06 Wed  
# Torch compile 打印 huggingface 的bert模型信息
# 
# 冷启动 TORCHINDUCTOR_FORCE_DISABLE_CACHES=1 
# 详细打印信息： TORCH_COMPILE_DEBUG=1 
# 融合信息： TORCH_LOGS=fusion INDUCTOR_ORIG_FX_SVG=1 INDUCTOR_POST_FUSION_SVG=1 
# TORCHINDUCTOR_FORCE_DISABLE_CACHES=1 TORCH_COMPILE_DEBUG=1 python bert_compile_print.py 
import sys
import copy
import torch
from transformers import AutoModel, AutoTokenizer # Hugging Face 模型和分词器
import os
from prettytable import PrettyTable 
from typing import List

def get_model_name():
    return "distilbert-base-uncased"    # 返回 Hugging Face 模型标识符

def create_model():
    model = AutoModel.from_pretrained(get_model_name())  # 加载预训练模型
    model.eval()             # 设置为评估模式（关闭 Dropout 等训练专用层，禁用训练阶段的随机行为， 确保推理结果稳定）
    return model.to(device)  # 将模型移动到指定设备（GPU/CPU）

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
    
    tokenizer = AutoTokenizer.from_pretrained(get_model_name()) # 加载与模型匹配的分词器

    text = "Hello world "
    inputs = tokenizer(text, max_length=128, padding='max_length', truncation=True, return_tensors="pt")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    model = create_model().half()
    print_model_info(model, inputs)

    compiled_model = torch.compile(model, mode='default')
    
    print("Running FP16 inference with seqlen=128...")
    output = compiled_model(**inputs)
    # print(output, output.last_hidden_state.type)
    print("Inference finished. Output shape:", output.last_hidden_state.shape)
    
