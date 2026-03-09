import torch
import torchvision

def resnet50(pretrained_weights=None, **kwargs):
    """
    Creates a ResNet-50 model using the Timm library.

    Args:
        pretrained_weights (str, optional): Path to pre-trained model weights. If provided, the input channels will be set based on the weights.
        kwargs (optional): Additional arguments and keyword arguments for the Timm library.

    Returns:
        torch.nn.Module: The ResNet-50 model.
    """
    if pretrained_weights is not None:
        # Set the input channels based on the pre-trained weights
        input_channels = 3
        if 'grayscale' in pretrained_weights:
            input_channels = 1
        model = timm.create_model('resnet50', num_classes=1000, input_channels=input_channels, **kwargs)

        # Load the state dictionary from the pre-trained weights
        state_dict = torch.load(pretrained_weights, map_location='cpu')
        model.load_state_dict(state_dict)

    else:
        # Create a new ResNet-50 model
        model = timm.create_model('resnet50', num_classes=1000, **kwargs)

    # Ensure that certain keys are present in the state dictionary
    for key in ['fc.weight', 'fc.bias']:
        if key not in model.state_dict().keys():
            raise ValueError(f'Missing key {key} in the state dictionary.')

    return model

if __name__ == "__main__":
    # Load pre-trained weights for the ResNet-50 model
    pretrained_weights = 'resnet50_weights.pth'

    # Create a new ResNet-50 model with the pre-trained weights
    model = resnet50(pretrained_weights=pretrained_weights)

    # Print the model's architecture
    print(model)