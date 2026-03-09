from transformers import AutoModelForImageClassification, AutoTokenizer

def swin_v2_b(pretrained_weights=None, *args, **kwargs):
    # Check if a pretrained model is provided
    if pretrained_weights:
        # Load the model with the given weights
        model = AutoModelForImageClassification.from_pretrained(pretrained_weights)
    else:
        # Load the default Swin Transformer v2 model
        model = AutoModelForImageClassification.from_config('microsoft/swin-tiny-patch4-window7-224')
    
    # Ensure the model is on the GPU if available
    model = model.to('cuda' if torch.cuda.is_available() else 'cpu')
    
    return model

if __name__ == "__main__":
    # Example usage
    model = swin_v2_b(pretrained_weights='microsoft/swin-tiny-patch4-window7-224')
    print("Model loaded:", model)

    # Dummy input data, replace with actual data
    inputs = torch.randn(2, 3, 224, 224)  # batch size 2, 3 channels, 224x224 input
    outputs = model(inputs)
    print("Outputs:", outputs)