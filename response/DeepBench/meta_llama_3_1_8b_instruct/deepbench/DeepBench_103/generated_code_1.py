import torch
import torch.nn as nn
import torchvision
from torchvision.models import swin_transformer

def swin_v2_b(pretrained_weights=None, **kwargs):
    model = swin_transformer.v2_b(weights=pretrained_weights)
    if pretrained_weights:
        model.load_state_dict(torch.load(pretrained_weights))
    return model

if __name__ == "__main__":
    pretrained_weights ='swin_v2_b.pth'  # replace with your pre-trained weights file
    model = swin_v2_b(pretrained_weights)
    print(model)
    input_tensor = torch.randn(1, 3, 224, 224)
    output = model(input_tensor)
    print(output.shape)