import torch.nn as nn


class SpectrogramModel(nn.Module):
    def __init__(
        self,
    ):
        super(SpectrogramModel, self).__init__()

        conv_layers = []

        self.conv1 = nn.Conv2d(1, 4, kernel_size=20, stride=4, padding=5)
        self.relu1 = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(4)
        conv_layers += [self.conv1, self.relu1, self.bn1]

        self.conv2 = nn.Conv2d(4, 8, kernel_size=6, stride=2, padding=1)
        self.relu2 = nn.ReLU()
        self.bn2 = nn.BatchNorm2d(8)
        conv_layers += [self.conv2, self.relu2, self.bn2]

        self.conv3 = nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1)
        self.relu3 = nn.ReLU()
        self.bn3 = nn.BatchNorm2d(16)
        conv_layers += [self.conv3, self.relu3, self.bn3]

        self.conv4 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.relu4 = nn.ReLU()
        self.bn4 = nn.BatchNorm2d(32)
        conv_layers += [self.conv4, self.relu4, self.bn4]

        self.ap = nn.AdaptiveAvgPool2d(output_size=1)
        self.lin = nn.Linear(in_features=32, out_features=9)  # TODO: hardcoded nr. of output classes

        self.conv = nn.Sequential(*conv_layers)

    def forward(self, x):
        x = self.conv(x)
        x = self.ap(x)
        x = x.view(x.shape[0], -1)
        x = self.lin(x)
        return x
