import torch
import timm

def vit_small_patch16_224(pretrained_weights=None, **kwargs):
    model = timm.create_model('vit_small_patch16_224', pretrained=False, **kwargs)
    
    if pretrained_weights:
        state_dict = torch.load(pretrained_weights)
        model.load_state_dict(state_dict, strict=False)
    
    return model

if __name__ == "__main__":
    # Create a sample input tensor with batch size 1, 3 color channels, 224x224 image size
    sample_input = torch.rand(1, 3, 224, 224)
    
    # Create a ViT model without pre-trained weights
    model = vit_small_patch16_224()
    
    # Pass the sample input through the model
    output = model(sample_input)
    
    # Print the output shape
    print(output.shape)