import torch

def silu(input_tensor, inplace=False):
    sigmoid = torch.sigmoid(input_tensor)
    if inplace:
        input_tensor.mul_(sigmoid)
        return input_tensor
    else:
        return input_tensor * sigmoid

if __name__ == "__main__":
    sample_input = torch.tensor([-3.0, -1.0, 0.0, 1.0, 3.0])
    output = silu(sample_input)
    print(output)