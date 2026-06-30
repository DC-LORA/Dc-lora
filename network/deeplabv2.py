import torch.nn as nn

affine_par = True

class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 1, stride=stride, bias=False)
        self.bn1   = nn.BatchNorm2d(planes, affine=affine_par)
        for p in self.bn1.parameters(): p.requires_grad = False
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=1, padding=dilation, dilation=dilation, bias=False)
        self.bn2   = nn.BatchNorm2d(planes, affine=affine_par)
        for p in self.bn2.parameters(): p.requires_grad = False
        self.conv3 = nn.Conv2d(planes, planes*4, 1, bias=False)
        self.bn3   = nn.BatchNorm2d(planes*4, affine=affine_par)
        for p in self.bn3.parameters(): p.requires_grad = False
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        r = x
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        if self.downsample is not None:
            r = self.downsample(r)
        return self.relu(x + r)

class ClassifierModule(nn.Module):
    def __init__(self, inplanes, dilations, paddings, num_classes):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Conv2d(inplanes, num_classes, 3, 1, p, dilation=d, bias=True)
            for d, p in zip(dilations, paddings)
        ])
        for m in self.branches: m.weight.data.normal_(0, 0.01)

    def forward(self, x):
        y = self.branches[0](x)
        for i in range(1, len(self.branches)):
            y = y + self.branches[i](x)
        return y

class ResNetMulti(nn.Module):
    def __init__(self, block, layers, num_classes):
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, 7, 2, 3, bias=False)
        self.bn1   = nn.BatchNorm2d(64, affine=affine_par)
        for p in self.bn1.parameters(): p.requires_grad = False
        self.relu  = nn.ReLU(inplace=True)
        self.pool  = nn.MaxPool2d(3, 2, 1, ceil_mode=True)

        self.layer1 = self._make_layer(block,  64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=4)

        # DLv2 classifier (ASPP-like multi-dilation)
        self.classifier = ClassifierModule(2048, [6,12,18,24], [6,12,18,24], num_classes)

        # expose attrs expected by your training code
        self.backbone = nn.Sequential(
            self.conv1, self.bn1, self.relu, self.pool,
            self.layer1, self.layer2, self.layer3, self.layer4
        )

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        down = None
        if (stride != 1) or (self.inplanes != planes*block.expansion) or (dilation in [2,4]):
            down = nn.Sequential(
                nn.Conv2d(self.inplanes, planes*block.expansion, 1, stride=stride, bias=False),
                nn.BatchNorm2d(planes*block.expansion, affine=affine_par)
            )
            for p in down[1].parameters(): p.requires_grad = False
        layers = [block(self.inplanes, planes, stride=stride, dilation=dilation, downsample=down)]
        self.inplanes = planes*block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, stride=1, dilation=dilation))
        return nn.Sequential(*layers)

    def forward(self, x):
        # keep same structure that your code expects: backbone then classifier
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = self.layer1(x); x = self.layer2(x); x = self.layer3(x); x = self.layer4(x)
        return self.classifier(x)


def get_deeplab_v2(num_classes=19, multi_level=False):
    # multi_level kept for API parity; your model returns a single logits map
    return ResNetMulti(Bottleneck, [3, 4, 23, 3], num_classes)

def deeplabv2_resnet101(num_classes=19, output_stride=16, **kwargs):
    # output_stride kept for API parity; DLv2 hard-codes dilations
    return ResNetMulti(Bottleneck, [3,4,23,3], num_classes)
