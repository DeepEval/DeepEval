import torch

def KORNIA_CHECK_LAF(laf, raises=False):
    # Check if the shape of the tensor is valid
    if laf.shape != torch.Size([2, 3]):
        if raises:
            raise ValueError("Invalid shape. Expected (B,N,2,3), got {}".format(laf.shape))
        else:
            return False
    return True

if __name__ == "__main__":
    # Create a sample tensor with a shape that matches the required format
    laf = torch.rand(2, 3)
    
    # Check if the tensor has a valid shape
    result = KORNIA_CHECK_LAF(laf, raises=True)
    
    # Print the result
    print("The tensor has a valid shape:", result)