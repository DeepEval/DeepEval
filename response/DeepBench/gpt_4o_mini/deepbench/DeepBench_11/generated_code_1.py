import torch

def xavier_normal_(tensor, gain=1.0, generator=None):
    fan_in, fan_out = torch.nn.init._calculate_fan_in_and_fan_out(tensor)
    std = gain * (2.0 / (fan_in + fan_out)) ** 0.5
    return tensor.normal_(0, std, generator=generator)

if __name__ == "__main__":
    tensor = torch.empty(3, 5)
    xavier_normal_(tensor)
    print(tensor)