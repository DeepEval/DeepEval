import torch
import torch.distributions as distributions
import numpy as np

def kaiming_uniform_(tensor, a=0, mode='fan_in', nonlinearity='relu', generator=None):
    fan_in, fan_out = tensor.size()
    if mode == 'fan_in':
        fan = fan_in
    elif mode == 'fan_out':
        fan = fan_out
    else:
        raise ValueError('mode must be "fan_in" or "fan_out"')
    if nonlinearity =='relu':
        gain = np.sqrt(2)
    elif nonlinearity == 'leaky_relu':
        gain = 1
    else:
        raise ValueError('nonlinearity must be "relu" or "leaky_relu"')
    std = gain / np.sqrt(fan)
    bound = std * np.sqrt(3)
    if generator is None:
        generator = torch.manual_seed(0)
    distribution = distributions.Uniform(-bound, bound)
    values = distribution.sample(tensor.shape)
    tensor.copy_(values)

if __name__ == "__main__":
    # Create a sample 3-dimensional tensor
    tensor = torch.randn(3, 3)
    # Create a sample generator
    generator = torch.manual_seed(42)
    # Call the function with sample input values
    kaiming_uniform_(tensor, a=0, mode='fan_in', nonlinearity='relu', generator=generator)
    print(tensor)