import torch
from timm.models import create_model

def vit_small_patch16_224(pretrained=False, **kwargs):
    model = create_model('vit_small_patch16_224', pretrained=pretrained, **kwargs)
    return model

if __name__ == "__main__":
    # Create a sample input tensor
    sample_input = torch.randn(1, 3, 224, 224)
    
    # Call the function
    model = vit_small_patch16_224(pretrained=True)
    
    # Print the model architecture and output shape
    print(model)
    
    # Run the model on the sample input
    output = model(sample_input)
    
    # Print the output shape
    print(output.shape)