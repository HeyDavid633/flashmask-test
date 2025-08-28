# 2025.8.03 
# 作为最简单的样例 首先尝试 从AlexNet上抓取子图
# 以AlexNet来举例 （1）简单实现模型，要可以看到模型结构 （2）torch compile打印 （3）打印结果对比 现在的 整图抓取效果 （4）试着在 AlexNet上子图抓取

# TORCHINDUCTOR_FORCE_DISABLE_CACHES=1 TORCH_LOGS=fusion TORCH_COMPILE_DEBUG=1 INDUCTOR_ORIG_FX_SVG=1 INDUCTOR_POST_FUSION_SVG=1 python alexnet_test.py
# 确保每次编译都是冷启动，禁用了cache
# TORCHINDUCTOR_FORCE_DISABLE_CACHES=1 
# torch_debug_print的输出信息 -- 包含多个文件
# TORCH_COMPILE_DEBUG=1 
#  输出 FX svg图；以及在node层面 融合后的svg图 注意在打印之前需要apt-get install graphviz  |  pip install pydot  
# TORCH_LOGS=fusion INDUCTOR_ORIG_FX_SVG=1 INDUCTOR_POST_FUSION_SVG=1 
# 
#

import torch
import torchvision
from torchvision import transforms
import torch.nn as nn
from torchinfo import summary 


# 使用 torchvision 提供的 AlexNet
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

    torch._dynamo.reset()
    compiled_model = torch.compile(model, mode='default')
    
    # Run inference
    print("\n" + "="*50)
    print("Running Inference")
    print("="*50)
    print("Input shape:", normalized_input.shape)
    
    with torch.no_grad():  # Disable gradient calculation
        output = compiled_model(normalized_input)
    
    print("\nInference finished.")
    print("Output shape:", output.shape)
    print("Output class prediction:", torch.argmax(output, dim=1).item())

    # if hasattr(torch, '_dynamo'):
    #     print("\nCompilation details:")
    #     print(torch._dynamo.explain(model)(normalized_input))
    
    # if hasattr(torch, '_dynamo'):
    #     print("\nCompilation (Graph break) details:")
    #     explanation = torch._dynamo.explain(compiled_model)(normalized_input)
    #     for break_reason in explanation.break_reasons:
    #         print(f"Graph break at: {break_reason}")