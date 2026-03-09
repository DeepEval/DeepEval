import torch
import timm

def vit_small_patch16_224(pretrained=False, weights=None, **kwargs):
    model = timm.create_model('vit_small_patch16_224', pretrained=pretrained, **kwargs)

    if weights is not None:
        if 'vit_small_patch16_224_in21k' in weights:
            in_chans = 3
        else:
            in_chans = 1
        model.patch_embed.proj.in_channels = in_chans

        checkpoint = torch.load(weights)
        model.load_state_dict(checkpoint['model'])

    return model

if __name__ == "__main__":
    # Create sample input values
    input_data = torch.randn(1, 3, 224, 224)

    # Call the function with pre-trained weights
    model = vit_small_patch16_224(weights='vit_small_patch16_224_in21k.pth')

    # Forward pass
    output = model(input_data)

    # Print the output
    print(output.shape)