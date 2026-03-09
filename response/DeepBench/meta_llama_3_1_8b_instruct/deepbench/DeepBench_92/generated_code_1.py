import torch

def KORNIA_CHECK_LAF(laf, raises):
    if laf.shape!= (1, 3, 2, 3):
        if raises:
            raise Exception("Invalid shape for Local Affine Frame (laf) tensor")
        else:
            return False
    return True

if __name__ == "__main__":
    # Create a sample input with valid shape
    laf_valid = torch.randn(1, 3, 2, 3)
    print(KORNIA_CHECK_LAF(laf_valid, raises=True))  # Should print: True

    # Create a sample input with invalid shape
    laf_invalid = torch.randn(1, 2, 2, 3)
    print(KORNIA_CHECK_LAF(laf_invalid, raises=True))  # Should raise an exception

    # Create a sample input with invalid shape and raises set to False
    laf_invalid2 = torch.randn(1, 2, 2, 3)
    print(KORNIA_CHECK_LAF(laf_invalid2, raises=False))  # Should print: False