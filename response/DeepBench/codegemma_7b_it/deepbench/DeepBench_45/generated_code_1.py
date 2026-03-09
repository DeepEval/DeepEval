import torch

def change_basis_real_to_complex(k, dtype=None, device=None):
    """
    Construct a transformation matrix to change the basis from real to complex spherical harmonics.

    Parameters
    ----------
    k : int
        The representation index, which determines the order of the representation.
    dtype : torch.dtype, optional
        The data type for the output tensor. If not provided, the function will infer it.
        Default is None.
    device : torch.device, optional
        The device where the output tensor will be placed. If not provided, the function
        will use the default device. Default is None.

    Returns
    -------
    torch.Tensor
        A transformation matrix Q that changes the basis from real to complex spherical harmonics.
    """

    if dtype is None:
        dtype = torch.float32

    if device is None:
        device = torch.device("cpu")

    # Construct the transformation matrix Q
    Q = torch.zeros((2 ** (2 * k), 2 ** (2 * k)), dtype=dtype, device=device)

    # Fill in the matrix elements according to the mathematical expressions
    for ell in range(0, k + 1):
        for m in range(-ell, ell + 1):
            for ell_ in range(0, k + 1):
                for m_ in range(-ell_, ell_ + 1):
                    if ell == ell_ and m == m_:
                        Q[ell + ell_ * (2 ** (2 * k)), ell_ + m_ * (2 ** (2 * k))] = 0.5 * torch.sqrt((2 * ell_ + 1) / (4 * torch.pi))

    return Q

if __name__ == "__main__":
    # Create sample input values
    k = 2
    dtype = torch.float32
    device = torch.device("cpu")

    # Call the function
    Q = change_basis_real_to_complex(k, dtype, device)

    # Print the results
    print(Q)