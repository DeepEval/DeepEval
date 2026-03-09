import torch

def apply_similarity_transformation(X, R, T, s):
    # X: (minibatch, num_points, d)
    # R: (minibatch, d, d)
    # T: (minibatch, d)
    # s: (minibatch,)

    # Expand s to match the shape of R for element-wise multiplication
    s = s.view(-1, 1, 1)

    # Apply rotation and scaling
    X_transformed = torch.bmm(X, R)

    # Apply scaling
    X_transformed *= s

    # Apply translation
    X_transformed += T.unsqueeze(1)

    return X_transformed

if __name__ == "__main__":
    # Define a minimal runnable example
    minibatch = 2
    num_points = 3
    d = 2
    X = torch.randn(minibatch, num_points, d)
    R = torch.randn(minibatch, d, d)
    T = torch.randn(minibatch, d)
    s = torch.randn(minibatch)

    # Ensure R is orthonormal
    with torch.no_grad():
        R = R / torch.norm(R, dim=(1, 2), keepdim=True)

    # Apply the function
    X_transformed = apply_similarity_transformation(X, R, T, s)

    # Print the results
    print("Original X:\n", X)
    print("Rotation R:\n", R)
    print("Translation T:\n", T)
    print("Scaling s:\n", s)
    print("Transformed X:\n", X_transformed)