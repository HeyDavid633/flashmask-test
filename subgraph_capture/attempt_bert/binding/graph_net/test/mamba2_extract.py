# 
# https://huggingface.co/state-spaces/mamba-2.8b-hf
#
# https://arxiv.org/pdf/2405.21060
# https://github.com/state-spaces/mamba
#
# /usr/local/lib/python3.12/dist-packages/transformers/models/mamba2/configuration_mamba2.py
#
# 由于 torch._dynamo 在 tracing 模型时，遇到了 torch._dynamo.mark_static_address 这样的“禁止追踪”的函数
# Mamba2 目前不兼容 torch.compile/torchdynamo tracing
# TORCHDYNAMO_DISABLE=1 python graph_net/test/mamba2_extract.py 
# 

import sys
sys.path.append("/flashmask-test/GraphNet")

import torch
from transformers import AutoModel, AutoTokenizer # Hugging Face 模型和分词器
import graph_net.torch 
import os
from torchviz import make_dot # apt-get install graphviz
from prettytable import PrettyTable 

def get_model_name():
    return "state-spaces/mamba-2.8b-hf"   
    # return "mamba2" 

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
    
    config = model.config
    table.add_row(["Model Name", get_model_name()])
    table.add_row(["Batch Size", inputs["input_ids"].shape[0]])
    table.add_row(["Sequence Length", inputs["input_ids"].shape[1]])
    
    table.add_row(["Hidden Size", getattr(config, "hidden_size", "N/A")])
    table.add_row(["State Size", getattr(config, "state_size", "N/A")])
    table.add_row(["Number of Hidden Layers", getattr(config, "num_hidden_layers", "N/A")])
    table.add_row(["Number of Heads", getattr(config, "num_heads", "N/A")])
    table.add_row(["Head Dim", getattr(config, "head_dim", "N/A")])
    table.add_row(["Expand", getattr(config, "expand", "N/A")])
    table.add_row(["Conv Kernel", getattr(config, "conv_kernel", "N/A")])
    table.add_row(["Number of Groups", getattr(config, "n_groups", "N/A")])
    
    table.add_row(["Vocab Size", getattr(config, "vocab_size", "N/A")])
    table.add_row(["Max Sequence Length", getattr(config, "max_seq_len", "N/A")])

    print("\n" + "="*50)
    print("Model Architecture Information")
    print("="*50)
    print(table)
    print("="*50 + "\n")

if __name__ == '__main__':
    
    tokenizer = AutoTokenizer.from_pretrained(get_model_name()) # 加载与模型匹配的分词器

    text = "Hello world"
    inputs = tokenizer(text, return_tensors="pt")    # 文本编码为 PyTorch 张量
    print(f"\n Input Information:\n Text: '{text}'")
    print(f" Tokenized Input: {inputs}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    model = create_model()   # 创建并加载模型
    print_model_info(model, inputs)
    model = graph_net.torch.extract(name=get_model_name())(model)  # 提取计算图 返回的是 gm.forward，即捕获后的前向传播函数

    print("Running inference...")
    output = model(**inputs)  # 前向传播（同时记录计算图）
    print("Inference finished. Output shape:", output.last_hidden_state.shape)
    
    # 作图 可视化得呈现
    dot = make_dot(output.last_hidden_state, params=dict(model.named_parameters()))
    output_dir = os.path.join(
        os.environ.get("GRAPH_NET_EXTRACT_WORKSPACE", "./graphnet_workspace"),
        get_model_name()
    )
    os.makedirs(output_dir, exist_ok=True)
    dot.render(
        os.path.join(output_dir, get_model_name() + "_graph"),
        format="pdf"
    )
