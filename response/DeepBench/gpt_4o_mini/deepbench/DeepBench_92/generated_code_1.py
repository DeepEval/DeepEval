import torch

def KORNIA_CHECK_LAF(laf, raises=True):
    if laf.shape != (laf.shape[0], laf.shape[1], 2, 3):
        if raises:
            raise Exception("Invalid shape for Local Affine Frame tensor. Expected shape is (B, N, 2, 3).")
        return False
    return True

if __name__ == "__main__":
    # Create a valid LAF tensor with shape (B, N, 2, 3)
    valid_laf = torch.randn(2, 5, 2, 3)
    
    # Create an invalid LAF tensor with shape (B, N, 2, 4)
    invalid_laf = torch.randn(2, 5, 2, 4)
    
    # Check valid LAF tensor
    try:
        result = KORNIA_CHECK_LAF(valid_laf, raises=True)
        print("Valid LAF shape check passed:", result)
    except Exception as e:
        print("Error:", e)

    # Check invalid LAF tensor
    try:
        result = KORNIA_CHECK_LAF(invalid_laf, raises=True)
        print("Invalid LAF shape check passed:", result)
    except Exception as e:
        print("Error:", e)