import torch

def silu(input, inplace=False):
    if inplace:
        input.sigmoid_()
        input *= input
        return input
    else:
        return input * torch.sigmoid(input)

if __name__ == "__main__":
    # Example usage:
    input = torch.randn(2, 3)
    output = silu(input)
    print(output)