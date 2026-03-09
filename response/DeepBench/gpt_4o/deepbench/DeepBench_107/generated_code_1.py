import torch
from torch import nn
from torchvision.models.detection.backbone_utils import BackboneWithFPN
from torchvision.models.resnet import resnet50, resnet101, resnet152
from torchvision.ops import misc as misc_nn_ops
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.rpn import RegionProposalNetwork, RPNHead
from torchvision.models.detection.roi_heads import RoIHeads
from torchvision.models.detection.faster_rcnn import FasterRCNN
from torchvision.ops.feature_pyramid_network import LastLevelMaxPool
from typing import Optional, List, Callable

class WeightsEnum:
    pass

class ExtraFPNBlock:
    pass

def resnet_fpn_backbone(
    *,
    backbone_name: str,
    weights: Optional[WeightsEnum],
    norm_layer: Callable[..., nn.Module] = misc_nn_ops.FrozenBatchNorm2d,
    trainable_layers: int = 3,
    returned_layers: Optional[List[int]] = None,
    extra_blocks: Optional[ExtraFPNBlock] = None,
) -> BackboneWithFPN:
    if backbone_name == 'resnet50':
        backbone = resnet50(norm_layer=norm_layer, pretrained=weights is not None)
    elif backbone_name == 'resnet101':
        backbone = resnet101(norm_layer=norm_layer, pretrained=weights is not None)
    elif backbone_name == 'resnet152':
        backbone = resnet152(norm_layer=norm_layer, pretrained=weights is not None)
    else:
        raise ValueError(f"Unsupported backbone_name {backbone_name}")

    layers_to_train = ['layer4', 'layer3', 'layer2', 'layer1']
    if trainable_layers < 0 or trainable_layers > 4:
        raise ValueError("trainable_layers should be in the range [0, 4]")
    
    # Freeze layers not to train
    for layer in layers_to_train[trainable_layers:]:
        for param in getattr(backbone, layer).parameters():
            param.requires_grad = False

    if returned_layers is None:
        returned_layers = [1, 2, 3, 4]
    
    assert all(layer in [1, 2, 3, 4] for layer in returned_layers)
    return_layers = {f'layer{k}': str(v) for v, k in enumerate(returned_layers)}

    in_channels_stage2 = backbone.inplanes // 8
    in_channels_list = [in_channels_stage2 * 2 ** (i - 1) for i in returned_layers]
    out_channels = 256
    if extra_blocks is None:
        extra_blocks = LastLevelMaxPool()

    return BackboneWithFPN(backbone, return_layers, in_channels_list, out_channels, extra_blocks=extra_blocks)

if __name__ == "__main__":
    # Minimal example
    backbone = resnet_fpn_backbone(backbone_name='resnet50', weights=None)
    print(backbone)