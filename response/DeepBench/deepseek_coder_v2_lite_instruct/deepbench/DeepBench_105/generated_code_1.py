import torch
import timm

def resnet50(weights=None, **kwargs):
    model = timm.create_model('resnet50', pretrained=weights is not None, **kwargs)
    if weights:
        if weights.endswith('.pth'):
            state_dict = torch.load(weights, map_location='cpu')
        else:
            state_dict = weights
        model.load_state_dict(state_dict, strict=False)
        if 'input_channels' in state_dict:
            model.default_cfg['input_channels'] = state_dict['input_channels']
    return model

if __name__ == "__main__":
    # Example usage
    model = resnet50(weights='resnet50_weights.pth')
    print(model)
    # For validation, you can also run a forward pass with sample input
    dummy_input = torch.randn(1, 3, 224, 224)
    output = model(dummy_input)
    print(output.shape)  # Should print something like torch.Size([1, 1000]) for classification tasks