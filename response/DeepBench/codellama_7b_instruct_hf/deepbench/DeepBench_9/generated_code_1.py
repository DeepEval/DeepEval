import torch

def kaiming_uniform_(tensor, a, fan_in, fan_out, nonlinearity, generator=None):
    # Calculate the bounds for the uniform distribution
    if nonlinearity == 'relu':
        fan = fan_in
    elif nonlinearity == 'leaky_relu':
        fan = fan_out
    else:
        raise ValueError(f'Unsupported nonlinearity: {nonlinearity}')

    gain = torch.nn.init.calculate_gain(nonlinearity, a)
    bound = 1 / gain / torch.sqrt(fan)

    # Sample from the uniform distribution
    if generator is None:
        tensor.uniform_(-bound, bound)
    else:
        tensor.uniform_(-bound, bound, generator)

    return tensor

if __name__ == "__main__":
    # Example usage
    tensor = torch.empty(2, 3, 4)
    kaiming_uniform_(tensor, 0, 3, 4, 'relu')
    print(tensor)