import torch

def xavier_normal_(tensor, gain=1.0, generator=None):
    if generator is None:
        torch.nn.init.xavier_normal_(tensor, gain=gain)
    else:
        torch.nn.init.xavier_normal_(tensor, gain=gain, generator=generator)

if __name__ == "__main__":
    # Create a random tensor
    tensor = torch.randn(3, 3)
    print("Before initialization:")
    print(tensor)

    # Call the function to initialize the tensor
    xavier_normal_(tensor)

    # Print the tensor after initialization
    print("After initialization:")
    print(tensor)