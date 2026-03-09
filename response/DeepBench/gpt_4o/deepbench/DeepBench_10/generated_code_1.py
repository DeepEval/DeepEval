import torch
import torch.nn.functional as F

def silu(input, inplace=False):
    if inplace:
        input.mul_(torch.sigmoid(input))
        return input
    else:
        return input * torch.sigmoid(input)

if __name__ == "__main__":
    # Create a sample input tensor
    sample_input = torch.tensor([-1.0, 0.0, 1.0, 2.0, 3.0])

    # Call the silu function with inplace=False
    output = silu(sample_input, inplace=False)
    print("Output with inplace=False:", output)

    # Call the silu function with inplace=True
    inplace_input = sample_input.clone()  # Clone to preserve original sample_input
    silu(inplace_input, inplace=True)
    print("Output with inplace=True:", inplace_input)