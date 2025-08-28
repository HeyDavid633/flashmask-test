# 2025.8.03 
# 作为最简单的样例 首先尝试 从AlexNet上抓取子图
# 以AlexNet来举例 （1）简单实现模型，要可以看到模型结构 （2）torch compile打印 （3）打印结果对比 现在的 整图抓取效果 （4）试着在 AlexNet上子图抓取
import torch
import torchvision
from torchvision import transforms
import torch.nn as nn
from torchinfo import summary  # 需要安装 torchinfo: pip install torchinfo

# 使用 torchvision 提供的 AlexNet
def get_alexnet_model():
    # 加载预训练模型 (使用 ImageNet 预训练权重)
    model = torchvision.models.alexnet(weights=torchvision.models.AlexNet_Weights.DEFAULT)
    model.eval()  # 设置为评估模式
    return model

# 打印模型结构的函数
def print_model_structure(model):
    print("\n" + "="*50)
    print("Model Architecture")
    print("="*50)
    
    # # 方法1: 直接打印模型结构
    # print("\n1. Basic Model Structure:")
    # print(model)
    
    # 方法2: 使用 torchinfo 获取详细结构信息
    print("\n2. Detailed Layer Information:")
    summary(model, input_size=(1, 3, 224, 224), col_names=["input_size", "output_size", "num_params", "trainable"])
    
    # 方法3: 打印模块树形结构
    # print("\n3. Module Hierarchy:")
    # for name, module in model.named_modules():
    #     if not list(module.children()):  # 只打印叶子节点
    #         print(f"{name:40} | {type(module).__name__}")

if __name__ == '__main__':
    # Normalization parameters for ImageNet
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    # Create dummy input (batch_size=1, channels=3, height=224, width=224)
    random_input = torch.rand(1, 3, 224, 224)
    normalized_input = normalize(random_input)

    # 获取模型
    model = get_alexnet_model()
    
    # 打印模型结构
    print_model_structure(model)
    
    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    normalized_input = normalized_input.to(device)
    
    # Run inference
    print("\n" + "="*50)
    print("Running Inference")
    print("="*50)
    print("Input shape:", normalized_input.shape)
    
    with torch.no_grad():  # Disable gradient calculation
        output = model(normalized_input)
    
    print("\nInference finished.")
    print("Output shape:", output.shape)
    print("Output class prediction:", torch.argmax(output, dim=1).item())