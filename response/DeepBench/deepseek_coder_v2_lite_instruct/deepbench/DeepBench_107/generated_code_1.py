import torch
import torch.nn as nn
from torchvision.models.resnet import resnet18, resnet34, resnet50, resnet101, resnet152
from torchvision.models.resnet import Bottleneck as Bottleneck
from torchvision.models.resnet import BasicBlock
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.ops import FeaturePyramidNetwork, misc_nn_ops
from typing import Callable, List, Optional, Union

def resnet_fpn_backbone(
    *,
    backbone_name: str,
    weights: Optional[str],
    norm_layer: Callable[..., nn.Module] = misc_nn_ops.FrozenBatchNorm2d,
    trainable_layers: int = 3,
    returned_layers: Optional[List[int]] = None,
    extra_blocks: Optional[nn.Module] = None,
) -> nn.Module:
    backbone = globals()[backbone_name](weights=weights, norm_layer=norm_layer)
    return_layers = {'layer4': 'out'}
    if returned_layers is None:
        returned_layers = [4]
    if extra_blocks is None:
        extra_blocks = FeaturePyramidNetwork([256, 512, 1024, 2048], 256, norm_layer)
    in_channels_list = [backbone.layer4[0].conv1.out_channels]
    out_channels = 256

    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

    for param in backbone.parameters():
        param.requires_grad = False

    return_layers = {k: v for k, v in return_layers.items() if int(k) <= trainable_layers}
    in_channels_list = [backbone.out_channels[k - 1] for k in return_layers.keys()]
    fpn = FeaturePyramidNetwork(in_channels_list, out_channels, extra_blocks)

    model = nn.Sequential(backbone, fpn)

    return model

if __name__ == "__main__":
    # Create sample input values
    backbone_name = 'resnet50'
    weights = 'DEFAULT'
    trainable_layers = 3
    returned_layers = [4]

    # Call the function
    model = resnet_fpn_backbone(
        backbone_name=backbone_name,
        weights=weights,
        trainable_layers=trainable_layers,
        returned_layers=returned_layers
    )

    # Print the results
    print(model)