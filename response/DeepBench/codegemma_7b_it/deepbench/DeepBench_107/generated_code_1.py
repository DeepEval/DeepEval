import torch
import torchvision.models as models
import torch.nn as nn
from typing import Callable, List, Optional
from torchvision.models.resnet import ResNet, BasicBlock, Bottleneck
from torchvision.models._utils import WeightsEnum
from torchvision.ops import misc_nn_ops

class ResNetBackboneWithFPN(nn.Module):
    def __init__(self, backbone_name: str, weights: Optional[WeightsEnum], norm_layer: Callable[..., nn.Module] = misc_nn_ops.FrozenBatchNorm2d, trainable_layers: int = 3, returned_layers: Optional[List[int]] = None, extra_blocks: Optional[ExtraFPNBlock] = None,):
        super().__init__()

        self.backbone = models.resnet50(weights=weights)
        self.fpn = FPN(self.backbone.backbone_shape, num_outs=5)

        self.trainable_layers = trainable_layers

        if self.trainable_layers == 0:
            for param in self.backbone.parameters():
                param.requires_grad = False
        elif self.trainable_layers == 1:
            for param in self.backbone.conv1.parameters():
                param.requires_grad = False
            for param in self.backbone.bn1.parameters():
                param.requires_grad = False
        elif self.trainable_layers == 2:
            for param in self.backbone.conv1.parameters():
                param.requires_grad = False
            for param in self.backbone.bn1.parameters():
                param.requires_grad = False
            for param in self.backbone.layer1.parameters():
                param.requires_grad = False
        elif self.trainable_layers == 3:
            for param in self.backbone.conv1.parameters():
                param.requires_grad = False
            for param in self.backbone.bn1.parameters():
                param.requires_grad = False
            for param in self.backbone.layer1.parameters():
                param.requires_grad = False
            for param in self.backbone.layer2.parameters():
                param.requires_grad = False

    def forward(self, x):
        features = self.backbone(x)
        features = self.fpn(features)

        return features

class FPN(nn.Module):
    def __init__(self, backbone_shape, num_outs):
        super().__init__()

        self.backbone_shape = backbone_shape
        self.num_outs = num_outs

        self.lateral_convs = nn.ModuleList()
        self.output_convs = nn.ModuleList()

        for i in range(4):
            self.lateral_convs.append(nn.Conv2d(backbone_shape[i], backbone_shape[i], kernel_size=1, stride=1, padding=0))
            self.output_convs.append(nn.Conv2d(backbone_shape[i], backbone_shape[i], kernel_size=3, stride=1, padding=1))

    def forward(self, x):
        features = []

        for i in range(4):
            lateral_feat = self.lateral_convs[i](x[i])
            if i > 0:
                lateral_feat += F.interpolate(features[i - 1], size=lateral_feat.size()[2:], mode='bilinear', align_corners=False)
            output_feat = self.output_convs[i](lateral_feat)
            features.append(output_feat)

        return features

if __name__ == "__main__":
    backbone = ResNetBackboneWithFPN(backbone_name="resnet50", weights=None, trainable_layers=3)
    input_tensor = torch.randn(1, 3, 224, 224)
    output = backbone(input_tensor)
    print(output)