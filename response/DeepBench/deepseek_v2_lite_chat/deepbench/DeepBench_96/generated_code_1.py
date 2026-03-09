import torch
import timm

def vit_small_patch16_224(pretrained=True, **kwargs):
    model = timm.create_model('vit_small_patch16_224', pretrained=pretrained, **kwargs)
    if pretrained:
        state_dict = model.state_dict()
        # Adjust input channels based on the weights
        if state_dict['patch_embed.proj.weight'].shape[1] != 768:  # Assuming '768' as the expected output channels
            state_dict['patch_embed.proj.weight'] = state_dict['patch_embed.proj.weight'][:,:768]
    return model

if __name__ == "__main__":
    # Create a ViT model with small patch size 16, optionally using pre-trained weights
    model = vit_small_patch16_224(pretrained=False)
    
    # Sample input values (assuming input images are of shape [1, 3, 224, 224])
    input_images = torch.randn(1, 3, 224, 224)
    
    # Pass the sample input through the model and print the output
    output = model(input_images)
    print(output)