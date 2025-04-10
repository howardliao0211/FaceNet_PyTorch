from torch import nn
from torchinfo import summary
import torch
import torch.nn.functional as F

try:
    from .blocks import InceptionBlock, ResNeXtBlock
except ImportError:
    from blocks import InceptionBlock, ResNeXtBlock

class FaceNet(nn.Module):
    def __init__(self):
        super(FaceNet, self).__init__()

        self.stem = nn.Sequential(
            nn.LazyConv2d(64, kernel_size=7, stride=2, padding=3),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            self.block(192, use_1x1conv=True),
            nn.BatchNorm2d(192),
            nn.MaxPool2d(2)
        )

        self.body3 = nn.Sequential(
            self.block(256, use_1x1conv=True),                 # 3a
            self.block(320, use_1x1conv=True),                 # 3b
            self.block(640, use_1x1conv=True, halve_dim=True), #3c,
        )

        self.body4 = nn.Sequential(
           self.block(640), #4a
           self.block(640), #4b
           self.block(640), #4c
           self.block(640), #4d
           self.block(1024, use_1x1conv=True, halve_dim=True), #4e
        )

        self.body5 = nn.Sequential(
            self.block(1024), #5a
            self.block(1024), #5b
        )

        self.head = nn.Sequential(
            nn.AvgPool2d(7),
            nn.Flatten(),
            nn.LazyLinear(128),
        )
    
    def block(self, channel: int, use_1x1conv: bool=False, halve_dim: bool=False) -> nn.Module:
        groups: int=32
        bot_mul: float=0.5
        
        if halve_dim:
            return ResNeXtBlock(channel, groups, bot_mul, use_1x1conv, strides=2)
        else:
            return ResNeXtBlock(channel, groups, bot_mul, use_1x1conv)

    def forward(self, x):
        x = self.stem(x)
        x = self.body3(x)
        x = self.body4(x)
        x = self.body5(x)
        x = self.head(x)
        x = x.reshape(x.shape[0], -1)
        x = F.normalize(x, p=2, dim=1)
        return x

if __name__ == '__main__':
    model = FaceNet()
    summary(model, input_size=(1, 3, 224, 224))
