#  2025.08.06 Wed  
#  初步提取 Bert 的子图 打印模型信息
# 
# 目前的子图 抓取方式流程： （必须肉眼观察才能抓取到）
#  1. 抓整图 模型信息打印 gm.graph.print_tabular()
#  2. 在整图信息中观察内容 然后手动调整 extract_subgraph 函数的入口参数
#    限制有：1）抓取完全是死的 无法抓一类，只能抓一个  2）如果有多个输入SDPA则抓不了 3）我并不想 Erase 掉任何东西 
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

def extract_subgraph(
    gm: torch.fx.GraphModule,
    input_names: List[str],
    output_names: List[str],
) -> torch.fx.GraphModule:

    # Assert that input and output names are valid
    assert set(input_names + output_names).issubset(
        {node.name for node in gm.graph.nodes}
    )

    # Copy to avoid modifying the original graph
    gm = copy.deepcopy(gm)

    # Set new inputs
    for node in gm.graph.nodes:
        if node.name in input_names:
            node.op, node.target, node.args, node.kwargs = 'placeholder', node.name, (), {}
    
    # Set new outputs
    output_node = gm.graph.find_nodes(op='output')[0]
    output_node.args = (tuple([node for node in gm.graph.nodes if node.name in output_names]),)

    # Eliminate dead code
    gm.graph.eliminate_dead_code()

    # Remove unused placeholders
    for node in gm.graph.find_nodes(op='placeholder'):
        if node.name not in input_names:
            gm.graph.erase_node(node)

    return torch.fx.GraphModule(gm, gm.graph)

if __name__ == '__main__':
    
    tokenizer = AutoTokenizer.from_pretrained(get_model_name()) # 加载与模型匹配的分词器

    text = "Hello world"
    inputs = tokenizer(text, return_tensors="pt")    # 文本编码为 PyTorch 张量
    print(f"\n Input Information:\n Text: '{text}'")
    print(f" Tokenized Input: {inputs}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    model = create_model() 
    print_model_info(model, inputs)
    input_args = (inputs["input_ids"], inputs["attention_mask"])
    gm = torch.export.export(model, input_args).module()  # 返回的是 gm.forward 即捕获后的前向传播函数

    # gm.graph.print_tabular()

    print("Running inference...")
    output = model(**inputs)  
    print("Inference finished. Output shape:", output.last_hidden_state.shape)

    # linear_nodes = gm.graph.find_nodes(op='call_function', target=torch.ops.aten.linear.default)
    # print("len(linear_nodes): ", len(linear_nodes))
    # transpose_nodes = gm.graph.find_nodes(op='call_function', target=torch.ops.aten.transpose.int)
    # view_nodes = gm.graph.find_nodes(op='call_function', target=torch.ops.aten.view.default)

    # sub_gm = extract_subgraph(gm, [transpose_nodes[4].name], [view_nodes[5].name])
    # sub_gm.graph.print_tabular()
  
    
#  得到的输出结果    
#   Running inference...
# Graph and tensors for 'distilbert-base-uncased' extracted successfully to: /flashmask-test/GraphNet/graphnet_workspace/distilbert-base-uncased
# Inference finished. Output shape: torch.Size([1, 4, 768])  4 是分词后的 token 数量，768 是 DistilBERT 的隐藏层维度
