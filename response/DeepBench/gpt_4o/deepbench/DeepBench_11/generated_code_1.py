import torch
from torch import Tensor

def xavier_normal_(tensor: Tensor, gain: float = 1.0, generator: torch.Generator = None) -> Tensor:
    if not tensor.is_floating_point():
        raise ValueError("Only floating point tensors are supported")
    
    fan_in, fan_out = torch.nn.init._calculate_fan_in_and_fan_out(tensor)
    std = gain * (2.0 / float(fan_in + fan_out)) ** 0.5
    with torch.no_grad():
        return tensor.normal_(0, std, generator=generator)

if __name__ == "__main__":
    # Create a tensor with a specific shape
    tensor = torch.empty(3, 5)
    
    # Initialize the tensor using xavier_normal_ function
    xavier_normal_(tensor)
    
    # Print the initialized tensor
    print("Initialized Tensor:")
    print(tensor)