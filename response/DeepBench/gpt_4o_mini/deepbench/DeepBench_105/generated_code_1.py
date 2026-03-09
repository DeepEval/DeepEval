import torch
import timm

def resnet50(weights=None, **kwargs):
    if weights:
        state_dict = torch.load(weights, map_location='cpu')
        input_channels = state_dict['conv1.weight'].shape[1]
    else:
        input_channels = 3  # Default for RGB images

    model = timm.create_model('resnet50', pretrained=False, **kwargs)
    
    if weights:
        model.load_state_dict(state_dict, strict=False)
        required_keys = ['conv1.weight', 'fc.weight', 'fc.bias']
        for key in required_keys:
            if key not in state_dict:
                raise ValueError(f'Missing required key in state_dict: {key}')

    return model

if __name__ == "__main__":
    # Create a sample model without pre-trained weights
    model = resnet50()
    print(model)

    # Create a sample input and test the model
    sample_input = torch.randn(1, 3, 224, 224)  # Batch size of 1, RGB image of size 224x224
    output = model(sample_input)
    print(output.shape)  # Should output torch.Size([1, 1000]) for the 1000 ImageNet classes