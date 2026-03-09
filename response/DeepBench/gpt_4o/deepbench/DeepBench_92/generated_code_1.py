import torch

def KORNIA_CHECK_LAF(laf, raises):
    if laf.shape[-3:] != (2, 3) or len(laf.shape) != 4:
        if raises:
            raise Exception("Invalid LAF shape. Expected shape is (B, N, 2, 3).")
        return False
    return True

if __name__ == "__main__":
    # Create a valid LAF tensor with shape (B, N, 2, 3)
    valid_laf = torch.rand(5, 10, 2, 3)
    
    # Create an invalid LAF tensor with incorrect shape
    invalid_laf = torch.rand(5, 10, 3, 2)
    
    # Check valid LAF without raising an exception
    print(KORNIA_CHECK_LAF(valid_laf, raises=False))  # Expected output: True
    
    # Check invalid LAF without raising an exception
    print(KORNIA_CHECK_LAF(invalid_laf, raises=False))  # Expected output: False
    
    # Check valid LAF with raising an exception
    print(KORNIA_CHECK_LAF(valid_laf, raises=True))  # Expected output: True
    
    # Check invalid LAF with raising an exception (this will raise an Exception)
    try:
        print(KORNIA_CHECK_LAF(invalid_laf, raises=True))
    except Exception as e:
        print(e)  # Expected output: Exception message