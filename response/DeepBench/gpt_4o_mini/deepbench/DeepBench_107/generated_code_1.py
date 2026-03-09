import torch
import torch.nn as nn
from typing import Optional, List, Callable
from torchvision.models import resnet50
from torchvision.ops import FeaturePyramidNetwork

class WeightsEnum:
    pass  # Placeholder for weights enumeration class

class ExtraFPNBlock(nn.Module):
    def __init__(self):
        super().__init__()
        # Add additional blocks here if needed

class BackboneWithFPN(nn.Module):
    def __init__(self, backbone, fpn):
        super().__init__()
        self.backbone = backbone
        self.fpn = fpn

    def forward(self, x):
        features = self.backbone(x)
        return self.fpn(features)

def resnet_fpn_backbone(*, backbone_name: str, weights: Optional[WeightsEnum], 
                         norm_layer: Callable[..., nn.Module] = nn.BatchNorm2d,
                         trainable_layers: int = 3, returned_layers: Optional[List[int]] = None, 
                         extra_blocks: Optional[ExtraFPNBlock] = None) -> BackboneWithFPN:
    
    backbone = resnet50(pretrained=True)  # Load a ResNet50 backbone
    layers = list(backbone.children())
    
    if trainable_layers < 0:
        trainable_layers = len(layers)
    
    for i, layer in enumerate(layers):
        for param in layer.parameters():
            param.requires_grad = (i >= len(layers) - trainable_layers)

    # Assuming the output features from ResNet are from specific layers
    feature_layers = [layers[i] for i in (3, 4, 5)]  # Example output layers
    fpn = FeaturePyramidNetwork(in_channels_list=[256, 512, 1024], out_channels=256)

    return BackboneWithFPN(nn.Sequential(*feature_layers), fpn)

if __name__ == "__main__":
    sample_input = torch.rand(1, 3, 224, 224)  # Batch size of 1, 3 channels, 224x224 image
    backbone_fpn = resnet_fpn_backbone(backbone_name='resnet50', weights=None)
    output = backbone_fpn(sample_input)
    print(output)