import timm
import torch

def vit_small_patch16_224(pretrained_weights=None, **kwargs):
    if pretrained_weights is not None:
        model = timm.create_model('vit_small_patch16_224', pretrained=False)
        if pretrained_weights.endswith('.pth') or pretrained_weights.endswith('.pt'):
            state_dict = torch.load(pretrained_weights)
            model.load_state_dict(state_dict)
        else:
            raise ValueError('Unsupported file format for weights')
    else:
        model = timm.create_model('vit_small_patch16_224', pretrained=True)
    model = model(**kwargs)
    return model

if __name__ == "__main__":
    # Create sample input values
    pretrained_weights = 'vit_small_patch16_224.pth'
    kwargs = {'num_classes': 10}

    # Call the function and print the results
    model = vit_small_patch16_224(pretrained_weights, **kwargs)
    print(model)