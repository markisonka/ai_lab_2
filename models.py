from torch import nn


class BasicBlock2(nn.Module):
    def __init__(self, inp, oup, stride=1):
        super().__init__()
        hidden_dim = inp * 6
        self.conv = nn.Sequential(
            nn.Sequential(
                nn.Conv2d(inp, hidden_dim, 1, bias=None),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(),
            ),
            nn.Sequential(
                nn.Conv2d(
                    hidden_dim,
                    hidden_dim,
                    3,
                    stride=stride,
                    padding=1,
                    groups=hidden_dim,
                    bias=None,
                ),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(),
            ),
            nn.Conv2d(hidden_dim, oup, 1, bias=False),
            nn.BatchNorm2d(oup),
        )

    def forward(self, x):
        return self.conv(x)


class StertcBlock(BasicBlock2):
    def __init__(self):
        super().__init__(32, 16)
        self.conv = nn.Sequential(
            nn.Sequential(
                nn.Conv2d(32, 32, 3, stride=1, padding=1, groups=32, bias=None),
                nn.BatchNorm2d(32),
                nn.ReLU6(inplace=True),
            ),
            nn.Conv2d(32, 16, 1, bias=False),
            nn.BatchNorm2d(16),
        )


class ResConnect(BasicBlock2):
    def __init__(self, inp, oup):
        super().__init__(inp, oup)

    def forward(self, x):
        return x + self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self, N_class: int = 1000):
        super().__init__()

        self.features = nn.Sequential(
            nn.Sequential(
                nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=None),
                nn.BatchNorm2d(32),
                nn.ReLU6(inplace=True),
            ),
            StertcBlock(),
            BasicBlock2(16, 24, 2),
            ResConnect(24, 24),
            BasicBlock2(24, 32, 2),
            ResConnect(32, 32),
            ResConnect(32, 32),
            BasicBlock2(32, 64, 2),
            ResConnect(64, 64),
            ResConnect(64, 64),
            ResConnect(64, 64),
            BasicBlock2(64, 96),
            ResConnect(96, 96),
            ResConnect(96, 96),
            BasicBlock2(96, 160, 2),
            ResConnect(160, 160),
            ResConnect(160, 160),
            BasicBlock2(160, 320),
            nn.Sequential(
                nn.Conv2d(320, 1280, 1, stride=1, bias=None),
                nn.BatchNorm2d(1280),
                nn.ReLU6(inplace=True),
            ),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(1280, N_class),
        )

    def forward(self, x):
        x = self.features(x)
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(-1, 1280)
        x = self.classifier(x)
        return x


class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes):
        super().__init__()
        stride = 1 if inplanes == planes else 2
        self.conv1 = nn.Conv2d(
            inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = (
            None
            if inplanes == planes
            else nn.Sequential(
                nn.Conv2d(inplanes, planes, kernel_size=1, stride=2, bias=False),
                nn.BatchNorm2d(planes),
            )
        )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class resnet18(nn.Module):
    def __init__(self, N_class: int = 1000):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = nn.Sequential(BasicBlock(64, 64), BasicBlock(64, 64))
        self.layer2 = nn.Sequential(BasicBlock(64, 128), BasicBlock(128, 128))
        self.layer3 = nn.Sequential(BasicBlock(128, 256), BasicBlock(256, 256))
        self.layer4 = nn.Sequential(BasicBlock(256, 512), BasicBlock(512, 512))

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, N_class)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(-1, 512)
        x = self.fc(x)
        return x
