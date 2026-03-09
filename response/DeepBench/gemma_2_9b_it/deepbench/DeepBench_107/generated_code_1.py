from typing import Optional

import torch
from torchvision.models import resnet50, resnet101
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.backbone_utils import BackboneWithFPN
from torch import nn

WeightsEnum = torch.hub.Weights

def resnet_fpn_backbone( *, backbone_name: str, weights: Optional[WeightsEnum], norm_layer: Callable[..., nn.Module] = misc_nn_ops.FrozenBatchNorm2d, trainable_layers: int = 3, returned_layers: Optional[List[int]] = None, extra_blocks: Optional[ExtraFPNBlock] = None, ) -> BackboneWithFPN:
    if backbone_name == 'resnet50':
        backbone = resnet50(weights=weights, norm_layer=norm_layer)
    elif backbone_name == 'resnet101':
        backbone = resnet101(weights=weights, norm_layer=norm_layer)
    else:
        raise ValueError(f"Unsupported backbone name: {backbone_name}")

    return resnet_fpn_backbone(
        backbone,
        trainable_layers=trainable_layers,
        returned_layers=returned_layers,
        extra_blocks=extra_blocks,
    )


if __name__ == "__main__":
    input_tensor = torch.randn(1, 3, 800, 600)
    backbone = resnet_fpn_backbone(backbone_name='resnet50', weights=WeightsEnum.DEFAULT)
    output = backbone(input_tensor)
    print(output)