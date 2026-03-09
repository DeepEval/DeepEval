import torch
import torch.nn as nn
from timm import create_model

def swin_v2_b(pretrained=False, **kwargs): 
    model = create_model('swinv2_base', pretrained=pretrained, **kwargs)
    return model

if __name__ == "__main__":
    sample_input = torch.randn(1, 3, 224, 224)
    model = swin_v2_b(pretrained=True)
    output = model(sample_input)
    print(output)