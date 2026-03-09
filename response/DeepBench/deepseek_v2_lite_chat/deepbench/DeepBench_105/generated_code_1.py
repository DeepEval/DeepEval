import torch
from timm.models import resnet50

def resnet50(weights=None, **kwargs):
    """
    Creates a ResNet-50 model.

    Args:
        weights (str or None): Optional pretrained weights path
        **kwargs: Additional arguments for the ResNet50 model

    Returns:
        torch.nn.Module: The ResNet-50 model
    """
    # Create the model
    model = resnet50(pretrained=weights is not None, **kwargs)
    
    # Load weights if provided
    if weights is not None:
        # Check if the state_dict has the required keys
        required_keys = set(model.state_dict().keys())
        if weights.get_legacy_keys():
            # Ensure that the keys are present in the new format
            for k in required_keys:
                if k not in weights.keys():
                    raise ValueError(f"Missing key in pretrained weights: {k}")
        else:
            # Ensure that the keys are present in the new format
            for k in required_keys:
                if k not in weights.keys():
                    raise ValueError(f"Missing key in pretrained weights: {k}")
        model.load_state_dict(weights, strict=False)
    
    return model

if __name__ == "__main__":
    # Sample input values
    input_shape = (3, 224, 224)
    num_classes = 1000  # Example number of classes for classification
    
    # Call the function
    model = resnet50(weights='imagenet', num_classes=num_classes)
    
    # Print the model structure
    print(model)
    
    # Example usage with dummy input
    input_data = torch.randn(1, *input_shape)
    output = model(input_data)
    print("Output shape:", output.shape)