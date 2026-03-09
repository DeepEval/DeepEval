import torch

def kaiming_uniform_(tensor, a=0, mode='fan_in', nonlinearity='relu', generator=None):
    if generator is not None:
        gen = generator
    else:
        gen = torch.manual_seed(0)

    if mode == 'fan_in':
        fan = torch.nn.init._calculate_fan_in_and_fan_out(tensor)[0]
    elif mode == 'fan_out':
        fan = torch.nn.init._calculate_fan_in_and_fan_out(tensor)[1]
    else:
        raise ValueError("Mode should be 'fan_in' or 'fan_out'")

    gain = torch.nn.init.calculate_gain(nonlinearity, a)
    std = gain / torch.sqrt(fan)
    bound = torch.sqrt(3.0) * std

    with torch.no_grad():
        return tensor.uniform_(-bound, bound, generator=gen)

if __name__ == "__main__":
    # Create a sample tensor
    tensor = torch.randn(3, 5)

    # Call the function with sample arguments
    kaiming_uniform_(tensor, a=0, mode='fan_in', nonlinearity='relu')

    # Print the results
    print(tensor)