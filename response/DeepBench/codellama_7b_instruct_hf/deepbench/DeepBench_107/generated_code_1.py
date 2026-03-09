import torch
import torch.nn as nn
import torchvision

def resnet_fpn_backbone(
    backbone_name: str,
    weights: Optional[WeightsEnum] = None,
    norm_layer: Callable[..., nn.Module] = misc_nn_ops.FrozenBatchNorm2d,
    trainable_layers: int = 3,
    returned_layers: Optional[List[int]] = None,
    extra_blocks: Optional[ExtraFPNBlock] = None,
) -> BackboneWithFPN:
    """
    Constructs a specified ResNet backbone with FPN on top. Freezes the specified number of layers in the backbone.
    """
    # Load the ResNet backbone
    backbone = getattr(torchvision.models, backbone_name)(weights=weights)
    # Replace the last layer with a FPN layer
    last_layer = backbone.last_linear
    fpn_layer = torch.nn.Sequential(
        nn.Conv2d(last_layer.out_channels, 256, kernel_size=1),
        norm_layer(256),
        nn.ReLU(inplace=True),
    )
    backbone.last_linear = fpn_layer
    # Freeze the specified number of layers in the backbone
    for layer in backbone.children()[:trainable_layers]:
        for param in layer.parameters():
            param.requires_grad = False
    # Create a dictionary of the returned layers
    returned_layers = {
        str(idx): layer for idx, layer in enumerate(backbone.children()[:returned_layers])
    }
    # Create the FPN module
    fpn = FPNModule(
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        norm_layer=norm_layer,
        activation=nn.ReLU(inplace=True),
    )
    # Add the FPN module to the backbone
    backbone.fpn = fpn
    # Return the backbone and the FPN module
    return backbone, fpn

if __name__ == "__main__":
    # Load a ResNet backbone with ImageNet weights
    backbone, fpn = resnet_fpn_backbone(backbone_name="resnet50", weights="imagenet")
    # Create a sample input tensor
    input_tensor = torch.randn(1, 3, 224, 224)
    # Forward pass through the backbone and FPN
    features = backbone(input_tensor)
    fpn_features = fpn(features)
    # Print the output shapes
    print(features.shape)  # (1, 256, 112, 112)
    print(fpn_features.shape)  # (1, 256, 56, 56)