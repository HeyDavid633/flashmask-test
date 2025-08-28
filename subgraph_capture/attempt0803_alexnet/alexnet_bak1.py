# 2025.8.03 
# 作为最简单的样例 首先尝试 从AlexNet上抓取子图
# 以AlexNet来举例 （1）简单实现模型，要可以看到模型结构 （2）torch compile打印 （3）打印结果对比 现在的 整图抓取效果 （4）试着在 AlexNet上子图抓取
import torch
import torchvision
from torchvision import transforms
import torch.nn as nn
from torchinfo import summary

class AlexNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

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
    normalized_input = normalize(random_input)

    # Instantiate model
    model = AlexNet()
    model.eval()  # Set to evaluation mode

    print_model_structure(model)
    
    
    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    normalized_input = normalized_input.to(device)
    
    # Run inference
    print("Running inference...")
    print("Input shape:", normalized_input.shape)
    
    with torch.no_grad():  # Disable gradient calculation
        output = model(normalized_input)
    
    print("Inference finished.")
    print("Output shape:", output.shape)
    print("Output class prediction:", torch.argmax(output, dim=1).item())