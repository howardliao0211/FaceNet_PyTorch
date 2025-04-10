from torch import nn
from torchinfo import summary
import torch
import torch.nn.functional as F

class InceptionBlock(nn.Module):
    def __init__(self,
                 conv1_ch: int,
                 conv2_chs: tuple[int],
                 conv3_chs: tuple[int],
                 conv4_ch: int,
                 halve_dim: bool=False) -> None:
        """
        Initializes an InceptionBlock module with four branches, each containing
        different convolutional operations.
        """

        super(InceptionBlock, self).__init__()

        if len(conv2_chs) != 2:
            raise ValueError(f'The len(conv2_chs) should be 2 but is {len(conv2_chs)}')

        if len(conv3_chs) != 2:
            raise ValueError(f'The len(conv3_chs) should be 2 but is {len(conv3_chs)}')

        strides = 2 if halve_dim else 1

        # Branch 1
        self.conv1 = nn.LazyConv2d(conv1_ch, kernel_size=1, stride=strides)

        # Branch 2
        self.conv2_1 = nn.LazyConv2d(conv2_chs[0], kernel_size=1, stride=strides)
        self.conv2_2 = nn.LazyConv2d(conv2_chs[1], kernel_size=3, padding=1)

        # Branch 3
        self.conv3_1 = nn.LazyConv2d(conv3_chs[0], kernel_size=1, stride=strides)
        self.conv3_2 = nn.LazyConv2d(conv3_chs[1], kernel_size=5, padding=2)

        # Branch 4
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=strides, padding=1)
        self.conv4 = nn.LazyConv2d(conv4_ch, kernel_size=1)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        # Use ReLU for non-linearity.
        b1_logits = F.relu(self.conv1(X))
        b2_logits = F.relu(self.conv2_2(F.relu(self.conv2_1(X))))
        b3_logits = F.relu(self.conv3_2(F.relu(self.conv3_1(X))))
        b4_logits = F.relu(self.maxpool(F.relu(self.conv4(X))))

        # Concatenate at the channel dimension (sample, channel, height, width).
        final_logits = torch.cat([b1_logits, b2_logits, b3_logits, b4_logits], dim=1)

        return final_logits

class ResNeXtBlock(nn.Module):
    """The ResNeXt block."""
    def __init__(self, num_channels, groups, bot_mul, use_1x1conv=False,
                 strides=1):
        super().__init__()
        bot_channels = int(round(num_channels * bot_mul))
        self.conv1 = nn.LazyConv2d(bot_channels, kernel_size=1, stride=1)
        self.conv2 = nn.LazyConv2d(bot_channels, kernel_size=3,
                                   stride=strides, padding=1,
                                   groups=bot_channels//groups)
        self.conv3 = nn.LazyConv2d(num_channels, kernel_size=1, stride=1)
        self.bn1 = nn.LazyBatchNorm2d()
        self.bn2 = nn.LazyBatchNorm2d()
        self.bn3 = nn.LazyBatchNorm2d()
        if use_1x1conv:
            self.conv4 = nn.LazyConv2d(num_channels, kernel_size=1,
                                       stride=strides)
            self.bn4 = nn.LazyBatchNorm2d()
        else:
            self.conv4 = None

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = F.relu(self.bn2(self.conv2(Y)))
        Y = self.bn3(self.conv3(Y))
        if self.conv4:
            X = self.bn4(self.conv4(X))
        return F.relu(Y + X)

if __name__ == '__main__':
    block = InceptionBlock(conv1_ch=3,
                           conv2_chs=(3, 4),
                           conv3_chs=(5, 6),
                           conv4_ch=7,
                           halve_dim=True)
    X = torch.rand((1, 3, 224, 224))
    Y = block(X)
    print(f'X shape: {X.shape}. Y shape: {Y.shape}')

    summary(block, input_size=(3, 244, 244))
