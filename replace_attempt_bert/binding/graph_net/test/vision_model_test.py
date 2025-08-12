import sys
sys.path.append("/flashmask-test/GraphNet")

import argparse
import os
import json
import torch
import torchvision
from torchvision import transforms
import os
import graph_net
from torchviz import make_dot  # apt-get install graphviz


def get_model_name():
    return "resnet18"  # 使用 torchvision 中的 ResNet-18 模型

if __name__ == '__main__':
    # Normalization parameters for ImageNet
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    # Create dummy input
    batch_size = 1
    height, width = 224, 224  # Standard ImageNet size
    num_channels = 3
    random_input = torch.rand(batch_size, num_channels, height, width)
    normalized_input = normalize(random_input)

    # Instantiate model
    model = torchvision.models.get_model(get_model_name(), weights="DEFAULT")
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    normalized_input = normalized_input.to(device)
    
    model = graph_net.torch.extract(name=get_model_name())(model)

    print("Running inference...")
    print("Input shape:", normalized_input.shape)
    output = model(normalized_input)
    print("Inference finished. Output shape:", output.shape)
    
    dot = make_dot(output, params=dict(model.named_parameters()))
    output_dir = os.path.join(
        os.environ.get("GRAPH_NET_EXTRACT_WORKSPACE", "./graphnet_workspace"),
        get_model_name()
    )
    os.makedirs(output_dir, exist_ok=True)
    dot.render(
        os.path.join(output_dir, get_model_name()+"_graph"),
        format="pdf"
    )