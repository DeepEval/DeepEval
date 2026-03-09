import timm

def resnet50(weights=None, *args, **kwargs):
    if weights is not None:
        kwargs['in_chans'] = 3  # Set input channels based on weights
        kwargs['num_classes'] = 1000  # Set number of classes based on weights

    model = timm.create_model('resnet50', pretrained=False, *args, **kwargs)

    if weights is not None:
        state_dict = torch.load(weights)
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}  # Remove module prefix
        model.load_state_dict(state_dict)
        assert set(state_dict.keys()).issubset(model.state_dict().keys()), 'Missing keys in state_dict'

    return model

if __name__ == "__main__":
    # Create sample input values
    input_data = torch.randn(1, 3, 224, 224)

    # Create ResNet-50 model with weights
    model = resnet50(weights='path/to/weights.pth')

    # Run the model and print the output
    output = model(input_data)
    print(output)