# Import Library
import torch
import torch.nn as nn
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet

# Define Model
"""
EfficientNet b7을 사용했습니다.
Task에 맞는 결과를 위해 26차원으로 변환하는 Lineart Layer를 추가합니다.
Activation Functuin은 silu를 사용했습니다.
"""
class MnistEfficientNet(nn.Module):
    def __init__(self, in_channels):
        super(MnistEfficientNet, self).__init__()
        self.EffNet = EfficientNet.from_pretrained('efficientnet-b7', in_channels=in_channels)
        self.FC = nn.Linear(1000, 26)

    def forward(self, x):
        x = F.silu(self.EffNet(x))
        x = torch.sigmoid(self.FC(x))
        return x