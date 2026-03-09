import timm
import torch
import torchvision

def resnet50(weights=None, **kwargs):
    if weights is not None:
        # Set input channels based on the weights
        weights = torch.load(weights)
        num_channels = list(weights['state_dict'].values())[0].shape[1]
        kwargs['num_classes'] = num_channels

    # Create a ResNet-50 model using the timm library
    model = timm.create_model('resnet50', **kwargs)

    if weights is not None:
        # Load the state dictionary from the weights
        state_dict = torch.load(weights)['state_dict']
        
        # Ensure that certain keys are present
        required_keys = ['conv1.weight', 'bn1.weight', 'bn1.bias', 'bn1.running_mean', 'bn1.running_var']
        for key in required_keys:
            if key not in state_dict:
                state_dict[key] = model.state_dict()[key]

        model.load_state_dict(state_dict, strict=False)

    return model

if __name__ == "__main__":
    # Create sample input values
    weights = 'path_to_your_pretrained_model.pth'
    model = resnet50(weights=weights)

    # Print the results
    print(model)
    print(model.state_dict().keys())