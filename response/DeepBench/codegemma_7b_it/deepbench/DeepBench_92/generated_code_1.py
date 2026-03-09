import torch

def KORNIA_CHECK_LAF(laf, raises=True):
    """Checks the shape of a Local Affine Frame (laf) tensor.

    Args:
        laf: The laf tensor.
        raises: Whether to raise an exception upon failure.

    Returns:
        True if the shape is valid, False otherwise.

    Raises:
        Exception if raises is True and the shape is invalid.
    """

    if not isinstance(laf, torch.Tensor):
        raise TypeError("Input `laf` must be a tensor.")

    if not torch.is_floating_point(laf):
        raise TypeError("Input `laf` must be a floating-point tensor.")

    if len(laf.shape) != 4 or laf.shape[1] != 2 or laf.shape[2] != 3:
        if raises:
            raise ValueError("Invalid shape for `laf`: expected (B,N,2,3) but got {}".format(laf.shape))
        return False

    return True

if __name__ == "__main__":
    # Example usage:
    laf = torch.randn(4, 2, 3)
    print(KORNIA_CHECK_LAF(laf))  # True

    laf = torch.randn(4, 3, 2, 3)
    print(KORNIA_CHECK_LAF(laf))  # False