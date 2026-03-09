import torch

def kaiming_uniform_(tensor, a=-1, mode='fan_in', nonlinearity='relu'):
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    scale = a * torch.sqrt(3.0 / fan_in if mode == 'fan_in' else 3.0 / fan_out)
    uniform = torch.randn(tensor.shape) * scale
    if nonlinearity == 'leaky_relu':
        leaky_value = 0.2
        uniform = torch.where(uniform > 0, uniform, leaky_value)
    tensor.data.copy_(uniform)

def _calculate_fan_in_and_fan_out(tensor):
    dimensions = tensor.dim()
    if dimensions < 2:
        raise ValueError("Fan in and fan out can only be computed for tensors with 2 or more dimensions")

    size = tensor.size(1)
    fan_in = tensor.size(0)
    fan_out = tensor.size(0)

    if fan_in == 0 or fan_out == 0:
        raise ValueError("Fan in and fan out cannot be computed due to zero size")

    return fan_in, fan_out

if __name__ == "__main__":
    # Example usage
    tensor_size = (100, 100)
    tensor = torch.randn(*tensor_size)
    kaiming_uniform_(tensor, a=-1, mode='fan_in', nonlinearity='relu')
    print("Kaiming Uniform Initialized Tensor:")
    print(tensor)