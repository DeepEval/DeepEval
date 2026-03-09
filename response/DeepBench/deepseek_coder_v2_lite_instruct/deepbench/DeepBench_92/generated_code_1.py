import torch

def KORNIA_CHECK_LAF(laf, raises=True):
    if laf.shape != (B, N, 2, 3):
        if raises:
            raise ValueError("LAF tensor shape must be (B, N, 2, 3)")
        else:
            return False
    return True

if __name__ == "__main__":
    # Sample input values
    B = 2
    N = 3
    laf = torch.randn(B, N, 2, 3)

    # Call the function and print the results
    result = KORNIA_CHECK_LAF(laf)
    print("Shape check passed:", result)

    # Test with an invalid shape
    laf_invalid = torch.randn(B, N, 2, 2)  # Invalid shape (2, 2) instead of (2, 3)
    try:
        result_invalid = KORNIA_CHECK_LAF(laf_invalid, raises=True)
    except ValueError as e:
        print(e)