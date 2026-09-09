import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    """Residual Block for ResNet"""
    def __init__(self, in_channels, out_channels, stride=1, use_bias=False):
        super(ResidualBlock, self).__init__()
        self.subconv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=use_bias)
        self.bn1 = nn.BatchNorm2d(out_channels, eps=1e-3, momentum=0.01)
        self.subconv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=use_bias)
        self.bn2 = nn.BatchNorm2d(out_channels, eps=1e-3, momentum=0.01)

        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)

        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=use_bias),
                # nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        shortcut = self.shortcut(x)
        x = self.subconv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.subconv2(x)
        x = self.bn2(x)
        x += shortcut
        return self.relu2(x)


class ResNet20(nn.Module):
    '''
    ResNet20 model (same with ATS and RTS)
    '''
    def __init__(self, in_channels, num_classes=10):
        super(ResNet20, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, stride=(1, 1), padding=1)
        # self.bn1 = nn.BatchNorm2d(num_features=16)
        self.bn1 = nn.BatchNorm2d(num_features=16, eps=1e-3, momentum=0.01)

        self.in_channels = 16

        # Residual blocks
        self.layer1 = self._make_layer(16, 3, stride=1)  # 3 blocks, output 16 channels
        self.layer2 = self._make_layer(32, 3, stride=2)  # 3 blocks, output 32 channels
        self.layer3 = self._make_layer(64, 3, stride=2)  # 3 blocks, output 64 channels

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(64, num_classes)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)


    def _make_layer(self, out_channels, num_blocks, stride):
        layers = []
        in_channels = self.in_channels
        for i in range(num_blocks):
            layers.append(ResidualBlock(in_channels, out_channels, stride if i == 0 else 1, use_bias=True))
            in_channels = out_channels
        self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        batch_size = x.size(0)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avg_pool(x)

        x = x.view(batch_size, -1)
        x = x.reshape(batch_size, -1)

        x = self.softmax(self.fc1(x))

        return x

def FM_ResNet20():
    """Return ResNet20 for FM"""

    return ResNet20(in_channels=1, num_classes=10)

def C10_ResNet20():
    """Return ResNet20 for C10"""

    return ResNet20(in_channels=3, num_classes=10)