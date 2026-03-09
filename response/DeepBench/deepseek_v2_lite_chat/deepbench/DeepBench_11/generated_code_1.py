import torch

def xavier_normal_(tensor):
    fan_in, fan_out = torch.nn.init.calculate_fan_in_and_fan_out(tensor)
    gain = torch.nn.init.calculate_gain('xavier_normal')
    std = gain * (1. / torch.sqrt(fan_in + fan_out))
    with torch.no_grad():
        return tensor.normal_(mean=0.0, std=std)

if __name__ == "__main__":
    import torch

    # Example usage
    # Create a tensor of size 5 with values sampled from a Xavier normal distribution
    tensor = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True)
    
    # Call the Xavier normal initialization function
    initialized_tensor = xavier_normal_(tensor)
    
    # Print the initialized tensor
    print("Initialized Tensor:", initialized_tensor)