import torch
import torch.nn as nn
from torchvision.models.utils import load_state_dict_from_url

class VGG(nn.Module):

    def __init__(self, features, pretrained):
        super(VGG, self).__init__()
        self.features = features

        if not pretrained:
            self._initialize_weights()

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_layers(in_channels):
    layers = []

    cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


def vgg19(pretrained, in_channels):
    model = VGG(make_layers(in_channels), pretrained)
    if pretrained:
        state_dict = load_state_dict_from_url('https://download.pytorch.org/models/vgg19-dcbb9e9d.pth')
        model.load_state_dict(state_dict, strict=False)
    return model
