import torch
import math

def kaiming_uniform_(tensor, a=0, mode='fan_in', nonlinearity='leaky_relu', generator=None):
    fan = torch.nn.init._calculate_correct_fan(tensor, mode)
    gain = torch.nn.init.calculate_gain(nonlinearity, a)
    std = gain / math.sqrt(fan)
    bound = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation
    with torch.no_grad():
        return tensor.uniform_(-bound, bound, generator=generator)

if __name__ == "__main__":
    tensor = torch.empty(3, 5)
    result = kaiming_uniform_(tensor, a=0.01, mode='fan_in', nonlinearity='leaky_relu')
    print(result)