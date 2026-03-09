import torch
from torch import nn
from timm.models.swin_transformer_v2 import SwinTransformerV2, swin_transformer_v2_base

def swin_v2_b(pretrained=None, **kwargs):
    model = swin_transformer_v2_base(pretrained=False, **kwargs)
    if pretrained:
        state_dict = torch.load(pretrained, map_location='cpu')
        model.load_state_dict(state_dict)
    return model

if __name__ == "__main__":
    # Create a sample input tensor with the shape (batch_size, channels, height, width)
    sample_input = torch.rand((1, 3, 224, 224))

    # Initialize the Swin Transformer v2 Base model
    model = swin_v2_b()

    # Forward pass through the model
    output = model(sample_input)

    # Print the output shape
    print("Output shape:", output.shape)