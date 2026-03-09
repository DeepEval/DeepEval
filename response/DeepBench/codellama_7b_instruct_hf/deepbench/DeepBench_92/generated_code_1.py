import torch

def KORNIA_CHECK_LAF(laf, raises=True):
    if not (laf.ndim == 4 and laf.shape[-2:] == (2, 3)):
        if raises:
            raise ValueError("Invalid shape for laf tensor")
        else:
            return False
    return True

if __name__ == "__main__":
    # Testing the function with valid input
    laf = torch.randn(2, 3, 2, 3)
    print(KORNIA_CHECK_LAF(laf))  # Should print True

    # Testing the function with invalid input
    laf = torch.randn(2, 3, 2, 4)
    print(KORNIA_CHECK_LAF(laf))  # Should print False

    # Testing the function with raises=False
    laf = torch.randn(2, 3, 2, 3)
    print(KORNIA_CHECK_LAF(laf, raises=False))  # Should print True