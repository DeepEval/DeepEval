import torch
import timm

def vit_small_patch16_224(weights=None, *args, **kwargs): 
    if weights is not None:
        model = timm.create_model('vit_small_patch16_224', pretrained=False, *args, **kwargs)
        input_channels = torch.load(weights)['state_dict']['patch_embed.proj.weight'].size(1)
        model.patch_embed.proj = timm.layers.Conv2d(input_channels, model.patch_embed.proj.out_channels, kernel_size=(16, 16), stride=(16, 16), bias=False)
        model.load_state_dict(torch.load(weights)['state_dict'], strict=False)
    else:
        model = timm.create_model('vit_small_patch16_224', pretrained=True, *args, **kwargs)

    return model

if __name__ == "__main__":
    model = vit_small_patch16_224()
    sample_input = torch.randn(1, 3, 224, 224)
    output = model(sample_input)
    print(output.shape)