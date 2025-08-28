# 
# 样例来自于
# https://medium.com/@tansiahuat/mastering-pytorch-graphmodule-how-to-extract-subgraphs-efficiently-072a8aeaa436
import torch
import copy
from typing import List
import torchvision
from torchvision import transforms
import torch.nn as nn
from torchinfo import summary 

def get_alexnet_model():
    # 加载预训练模型 (使用 ImageNet 预训练权重)
    model = torchvision.models.alexnet(weights=torchvision.models.AlexNet_Weights.DEFAULT)
    model.eval()  # 设置为评估模式
    return model

def print_model_structure(model):
    print("\n" + "="*50)
    print("Model Architecture")
    print("="*50)
    
    print("\n2. Detailed Layer Information:")
    summary(model, input_size=(1, 3, 224, 224), col_names=["input_size", "output_size", "num_params", "trainable"])
    

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
    # Normalization parameters for ImageNet
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    # Create dummy input (batch_size=1, channels=3, height=224, width=224)
    random_input = torch.rand(1, 3, 224, 224)
    normalized_input = normalize(random_input).to(torch.float16)
    

    model = get_alexnet_model()
    # print_model_structure(model)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.half()
    normalized_input = normalized_input.to(device)

    # Export the model to GraphModule
    gm = torch.export.export(model, (normalized_input,)).module()

    gm.graph.print_tabular()

    # Find the linear nodes
    linear_nodes = gm.graph.find_nodes(op='call_function', target=torch.ops.aten.linear.default)
    print(len(linear_nodes))

    # In this example, we want to extract the second linear layer as a subgraph
    # So the input should be the output of the first linear layer
    # And the output should be the output of the second linear layer
    sub_gm = extract_subgraph(gm, [linear_nodes[0].name], [linear_nodes[1].name])
    sub_gm.graph.print_tabular()