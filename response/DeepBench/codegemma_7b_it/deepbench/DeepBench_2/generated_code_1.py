import torch

def apply_similarity(X, R, T, s):
    """Applies a similarity transformation to a cloud.

    Args:
        X: A cloud of shape (minibatch, num_points, d).
        R: A batch of orthonormal matrices of shape (minibatch, d, d).
        T: A batch of translations of shape (minibatch, d).
        s: A batch of scaling factors of shape (minibatch,).

    Returns:
        The transformed cloud of shape (minibatch, num_points, d).
    """

    # Apply the rotation and translation
    X_transformed = torch.einsum('bij,bjd->bid', R, X) + T.unsqueeze(1)

    # Apply the scaling factor
    X_transformed *= s.unsqueeze(1).unsqueeze(2)

    return X_transformed

if __name__ == "__main__":
    # Create sample input values
    minibatch = 3
    d = 5
    num_points = 10
    X = torch.randn(minibatch, num_points, d)
    R = torch.randn(minibatch, d, d)
    R = R @ R.transpose(1, 2)
    T = torch.randn(minibatch, d)
    s = torch.rand(minibatch)

    # Apply the similarity transformation
    X_transformed = apply_similarity(X, R, T, s)

    # Print the results
    print(X_transformed.shape)