from torch import nn
from torchinfo import summary
import torch
import torch.nn.functional as F

try:
    from .blocks import InceptionBlock
except ImportError:
    from blocks import InceptionBlock
class Inception(InceptionBlock):
    def __init__(self, output_ch: int, halve_dim: bool=False) -> None:
        output_ch //= 4
        super(Inception, self).__init__(
            conv1_ch=output_ch,
            conv2_chs=(output_ch, output_ch),
            conv3_chs=(output_ch, output_ch),
            conv4_ch=(output_ch),
            halve_dim=halve_dim
        )

class FaceNet(nn.Module):
    def __init__(self):
        super(FaceNet, self).__init__()

        self.stem = nn.Sequential(
            nn.LazyConv2d(64, kernel_size=7, stride=2, padding=3),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            Inception(192),
            nn.BatchNorm2d(192),
            nn.MaxPool2d(2)
        )

        self.body3 = nn.Sequential(
            Inception(256),                 # 3a
            Inception(320),                 # 3b
            Inception(640, halve_dim=True), #3c,
        )

        self.body4 = nn.Sequential(
            Inception(640), #4a
            Inception(640), #4b
            Inception(640), #4c
            Inception(640), #4d
            Inception(1024, halve_dim=True), #4e
        )

        self.body5 = nn.Sequential(
            Inception(1024), #5a
            Inception(1024), #5b
        )

        self.head = nn.Sequential(
            nn.AvgPool2d(7),
            nn.Flatten(),
            nn.LazyLinear(128),
        )
    
    def forward(self, x):
        x = self.stem(x)
        x = self.body3(x)
        x = self.body4(x)
        x = self.body5(x)
        x = self.head(x)
        l2 = torch.sqrt(x**2)
        return l2

class MiniFaceNet(nn.Module):
    def __init__(self):
        super(MiniFaceNet, self).__init__()

        self.stem = nn.Sequential(
            nn.LazyConv2d(64, kernel_size=7, stride=2, padding=3),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            Inception(64),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2)
        )

        self.body3 = nn.Sequential(
            Inception(128),                 # 3a
            Inception(256),                 # 3b
            Inception(320, halve_dim=True), #3c,
        )

        self.body4 = nn.Sequential(
            Inception(320), #4a
            Inception(320), #4b
            Inception(320), #4c
            Inception(320), #4d
            Inception(640, halve_dim=True), #4e
        )

        self.body5 = nn.Sequential(
            Inception(640), #5a
            Inception(640), #5b
        )

        self.head = nn.Sequential(
            nn.AvgPool2d(7),
            nn.Flatten(),
            nn.LazyLinear(128),
        )
    
    def forward(self, x):
        x = self.stem(x)
        x = self.body3(x)
        x = self.body4(x)
        x = self.body5(x)
        x = self.head(x)
        l2 = torch.sqrt(x**2)
        return l2

if __name__ == '__main__':
    model = MiniFaceNet()
    summary(model, input_size=(1, 3, 224, 224))
