import torch
import timm

def resnet50(weights=None, *args, **kwargs):
    model = timm.create_model('resnet50', pretrained=False, *args, **kwargs)
    
    if weights:
        state_dict = torch.load(weights)
        model.load_state_dict(state_dict)
        # Ensure certain keys are present
        required_keys = ['layer1.0.conv1.weight', 'layer4.2.bn3.bias']
        for key in required_keys:
            if key not in state_dict:
                raise ValueError(f"Key {key} not found in the provided weights.")
                
    return model

if __name__ == "__main__":
    # Sample input to test the function
    model = resnet50()
    print(model)

    # You can uncomment the following lines to test with a sample input tensor
    # sample_input = torch.randn(1, 3, 224, 224)
    # output = model(sample_input)
    # print(output)