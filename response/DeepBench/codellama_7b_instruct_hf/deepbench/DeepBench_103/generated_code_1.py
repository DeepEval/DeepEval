import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision import models

def swin_v2_b(pretrained=False, progress=True, **kwargs):
    model = models.swin_transformer_v2_b(pretrained=pretrained, progress=progress, **kwargs)
    return model

if __name__ == "__main__":
    # create sample input values
    sample_input = torch.randn(1, 3, 224, 224)

    # call the function and print the results
    model = swin_v2_b()
    output = model(sample_input)
    print(output)