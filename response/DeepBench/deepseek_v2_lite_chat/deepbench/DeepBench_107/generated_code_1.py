import torch
import torch.nn as nn
from typing import Optional, List

# Define the ResNet architectures
def resnet(arch: str, block: nn.Module, layers: List[int], num_classes: int = 1000, zero_init_residual: bool = False) -> nn.Module:
    model = nn.Sequential(
        *[block(num_chs, num_blocks, stride=stride) for num_chs, num_blocks, stride in zip([256, 512, 1024, 2048], [3, 4, 6, 3], [1, 2, 2, 2])]
    )
    return model

# Define the FPN layer
class FPN(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(FPN, self).__init__()
        self.top_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bottom_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        top = self.top_layer(x)
        bottom = self.bottom_layer(x)
        return top, bottom

def resnet_fpn_backbone(backbone_name: str, weights: Optional[WeightsEnum] = None, norm_layer: Callable[..., nn.Module] = nn.BatchNorm2d, trainable_layers: int = 3, returned_layers: Optional[List[int]] = None, extra_blocks: Optional[ExtraFPNBlock] = None):
    if backbone_name == "resnet18":
        num_chs = 64
        num_blocks = [2, 2, 2, 2]
        model = resnet(backbone_name, BasicBlock, num_blocks, num_classes=1000, zero_init_residual=True)
    elif backbone_name == "resnet34":
        num_chs = 64
        num_blocks = [3, 4, 6, 3]
        model = resnet(backbone_name, BasicBlock, num_blocks, num_classes=1000, zero_init_residual=True)
    elif backbone_name == "resnet50":
        num_chs = 256
        num_blocks = [3, 4, 6, 3]
        model = resnet(backbone_name, Bottleneck, num_blocks, num_classes=1000, zero_init_residual=True)
    elif backbone_name == "resnet101":
        num_chs = 256
        num_blocks = [3, 4, 23, 3]
        model = resnet(backbone_name, Bottleneck, num_blocks, num_classes=1000, zero_init_residual=True)
    elif backbone_name == "resnet152":
        num_chs = 256
        num_blocks = [3, 8, 36, 3]
        model = resnet(backbone_name, Bottleneck, num_blocks, num_classes=1000, zero_init_residual=True)
    else:
        raise ValueError("Unsupported backbone name")
    
    if weights:
        if isinstance(weights, str) and weights.lower() == "imagenet":
            model.load_state_dict(torch.hub.load_state_dict_from_url(url="https://download.pytorch.org/models/resnet{}_{}f.pth".format(str(num_chs), str(trainable_layers)), model_dir=None, progress=True)["model"])
        else:
            model.load_state_dict(torch.load(weights, map_location=torch.device('cpu')))
        model = model.eval()
    
    fpn = FPN(in_channels=num_chs, out_channels=256)
    
    # Freeze the specified number of layers
    num_to_freeze = trainable_layers
    for name, param in model.named_parameters():
        if 'layer' in name and 'bias' not in name and 'norm' not in name:
            param.requires_grad = False
        if num_to_freeze > 0 and name.startswith('layer.'):
            num_to_freeze -= 1
    
    model = nn.Sequential(model, fpn)
    
    if returned_layers is not None:
        layers = []
        for i in range(len(model)):
            if i in returned_layers:
                layers.append(model[i])
        model = nn.Sequential(*layers)
    
    if extra_blocks is not None:
        model = extra_blocks(model, out_channels=256)
    
    return model

# Define the extra blocks if needed
class ExtraFPNBlock(nn.Module):
    def __init__(self, model: nn.Module, out_channels: int):
        super(ExtraFPNBlock, self).__init__()
        self.model = model
        self.out_channels = out_channels

    def forward(self, x):
        return self.model(x)

if __name__ == "__main__":
    backbone = resnet_fpn_backbone("resnet50", trainable_layers=2, norm_layer=nn.BatchNorm2d)
    backbone.eval()
    input_tensor = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        output = backbone(input_tensor)
        print(output.shape)  # Expected output shape: (1, 256, 224, 224)