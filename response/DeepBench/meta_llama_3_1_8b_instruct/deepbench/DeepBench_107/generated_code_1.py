import torch
import torch.nn as nn
from torchvision.models import resnet
from torchvision.models.detection.backbone import build_backbone
from torchvision.models.detection.faster_rcnn import FasterRCNN
from torchvision.models.detection.fpn import LastLevelMaxPool, FPN
from torchvision.models.detection.backbone_utils import BackboneWithFPN
from torchvision.models.detection.fcos import FCOS
from typing import Optional, List, Callable
from enum import Enum
from torchvision.models._utils import IntermediateLayerGetter

class WeightsEnum(Enum):
    imagenet1k = 'imagenet1k'

class ExtraFPNBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ExtraFPNBlock, self).__init__()
        self.block = nn.Conv2d(in_channels, out_channels, kernel_size=1)

class misc_nn_ops:
    FrozenBatchNorm2d = nn.BatchNorm2d

def resnet_fpn_backbone(
    *, 
    backbone_name: str, 
    weights: Optional[WeightsEnum], 
    norm_layer: Callable[..., nn.Module] = misc_nn_ops.FrozenBatchNorm2d, 
    trainable_layers: int = 3, 
    returned_layers: Optional[List[int]] = None, 
    extra_blocks: Optional[ExtraFPNBlock] = None, 
) -> BackboneWithFPN:
    # Create the backbone
    backbone = getattr(resnet, backbone_name)(pretrained=weights is not None, 
                                               norm_layer=norm_layer)
    
    # Freeze some layers
    for name, parameter in backbone.named_parameters():
        if 'layer3' not in name and 'layer4' not in name:
            parameter.requires_grad_(False)
        else:
            if trainable_layers == 0:
                parameter.requires_grad_(False)
            elif int(name.split('.')[1]) < trainable_layers:
                parameter.requires_grad_(True)
            else:
                parameter.requires_grad_(False)
                
    # Create the FPN
    in_channels_stage2 = 256
    in_channels_list = [in_channels_stage2]
    for i in range(1, len(backbone.named_modules()[1:-1])):
        ifi = i + 1
        m = backbone.named_modules()[1:-1][i][1]
        if isinstance(m, nn.Conv2d):
            out_channels = m.out_channels
            in_channels_list.append(out_channels)
    
    if returned_layers is None:
        returned_layers = list(range(len(backbone.layer4)))
    
    return_layers = {f'layer{i}': str(i) for i in returned_layers}
    
    fpn = FPN(
        in_channels_list=in_channels_list,
        out_channels=256,
        norm_layer=norm_layer,
        top_blocks=LastLevelMaxPool() if extra_blocks is None else None,
    )
    
    # Combine the backbone and the FPN
    backbone_with_fpn = IntermediateLayerGetter(backbone, return_layers=return_layers)
    model = FPN(backbone_with_fpn, fpn)
    
    return model

if __name__ == "__main__":
    weights = WeightsEnum.imagenet1k
    backbone_name ='resnet50'
    backbone = resnet_fpn_backbone(backbone_name=backbone_name, weights=weights)
    print(backbone)