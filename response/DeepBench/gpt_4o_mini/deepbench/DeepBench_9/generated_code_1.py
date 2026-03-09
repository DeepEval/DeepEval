import torch
import math

def kaiming_uniform_(tensor, a=0, mode='fan_in', nonlinearity='relu', generator=None):
    fan = torch.nn.init._calculate_fan_in_and_fan_out(tensor)
    
    if mode == 'fan_in':
        fan = fan[0]
    elif mode == 'fan_out':
        fan = fan[1]
    else:
        raise ValueError("Invalid mode. Choose either 'fan_in' or 'fan_out'.")

    gain = torch.nn.init.calculate_gain(nonlinearity, a)
    bound = math.sqrt(3.0) * gain / math.sqrt(fan)
    
    return tensor.uniform_(-bound, bound, generator)

if __name__ == "__main__":
    tensor = torch.empty(3, 5)
    kaiming_uniform_(tensor, a=0.01, mode='fan_in', nonlinearity='relu')
    print(tensor)