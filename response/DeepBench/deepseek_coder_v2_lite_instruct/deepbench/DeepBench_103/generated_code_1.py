import torch
from torch import nn
from transformers import SwinTransformerV2Model, SwinTransformerV2Config

def swin_v2_b(weights=None, **kwargs):
    config = SwinTransformerV2Config.from_pretrained('microsoft/swin-v2-base')
    model = SwinTransformerV2Model(config)
    
    if weights:
        model.load_state_dict(torch.load(weights), strict=False)
    
    return model

if __name__ == "__main__":
    model = swin_v2_b()
    inputs = torch.randn(1, 3, 224, 224)
    outputs = model(inputs)
    print(outputs)