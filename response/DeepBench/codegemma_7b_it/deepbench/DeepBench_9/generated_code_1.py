import torch
import math
def kaiming_uniform_(tensor, a, mode, nonlinearity, generator=None):
    fan = None
    if mode == "fan_in":
        fan = tensor.size(0)
    elif mode == "fan_out":
        fan = tensor.size(1)
    elif mode == "fan_avg":
        fan = tensor.size(0) / 2
    else:
        raise ValueError("Invalid mode.")

    gain = 5.0 / 3
    if nonlinearity == "leaky_relu":
        gain /= 2

    std = gain * math.sqrt(2.0 / fan)
    bound = math.sqrt(3.0) * std

    if generator is None:
        generator = torch.Generator()

    return torch.nn.init.uniform_(tensor, -bound, bound, generator=generator)

if __name__ == "__main__":
    # Create sample input values
    tensor = torch.randn(3, 4)

    # Call the function
    kaiming_uniform_(tensor, a=0.1, mode="fan_in", nonlinearity="relu")

    # Print the results
    print(tensor)