import sys
sys.path.append("/flashmask-test/GraphNet")

import torch
import torchvision
from torchvision import transforms
import graph_net
from torchviz import make_dot  # apt-get install graphviz
from prettytable import PrettyTable
import os

def print_model_info(model):
    """打印ViT模型关键信息的表格"""
    table = PrettyTable()
    table.field_names = ["Key", "Value"]
    table.align["Key"] = "l"
    table.align["Value"] = "l"

    # Torchvision风格模型
    table.add_row(["Model Name", model.__class__.__name__])
    table.add_row(["Hidden Size", getattr(model, "hidden_dim", "N/A")])
    table.add_row(["MLP Dim", getattr(model, "mlp_dim", "N/A")])
    table.add_row(["Number of Layers", getattr(model, "num_layers", "N/A")])
    table.add_row(["Number of Attention Heads", getattr(model, "num_heads", "N/A")])
    table.add_row(["Patch Size", getattr(model, "patch_size", "N/A")])
    table.add_row(["Image Size", getattr(model, "image_size", "N/A")])
    table.add_row(["Num Classes", getattr(model, "num_classes", "N/A")])

    print("\n" + "="*50)
    print("ViT Model Architecture Information")
    print("="*50)
    print(table)
    print("="*50 + "\n")

if __name__ == '__main__':
    # ViT-B_16 expects 224x224 images, normalization as below
    normalize = transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )

    batch_size = 1
    height, width = 224, 224
    num_channels = 3
    random_input = torch.rand(batch_size, num_channels, height, width)
    normalized_input = normalize(random_input)

    # Instantiate ViT model
    model = torchvision.models.get_model("vit_b_16", weights="DEFAULT")
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    normalized_input = normalized_input.to(device)
    
    print_model_info(model)

    model = graph_net.torch.extract(name="vit_b_16")(model)

    print("Running inference...")
    print("Input shape:", normalized_input.shape)
    output = model(normalized_input)
    print("Inference finished. Output shape:", output.shape)
    
    dot = make_dot(output, params=dict(model.named_parameters()))
    output_dir = os.path.join(
        os.environ.get("GRAPH_NET_EXTRACT_WORKSPACE", "./graphnet_workspace"),
        "vit_b_16"
    )
    os.makedirs(output_dir, exist_ok=True)
    dot.render(
        os.path.join(output_dir, "vit_b_16_graph"),
        format="pdf"
    )