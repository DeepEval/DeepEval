import torch

def run_7point(points1, points2):
    assert points1.shape == points2.shape, "Input point sets must have the same shape."
    assert points1.shape[1] == 7, "Exactly 7 points must be provided."

    B, N, _ = points1.shape

    # Normalize points
    points1 = torch.nn.functional.normalize(points1)
    points2 = torch.nn.functional.normalize(points2)

    # Construct linear system
    A = torch.zeros((3 * B, 9))
    for i in range(B):
        for j in range(N):
            x1, y1 = points1[i, j]
            x2, y2 = points2[i, j]
            A[3 * i + j, :] = torch.tensor([x1 * x2, x1 * y2, x1, y1 * x2, y1 * y2, y1, x2, y2, 1])

    # Solve linear system using SVD
    U, S, Vh = torch.linalg.svd(A)
    F1 = Vh.t()[:, -1].view(3, 3)
    F2 = Vh.t()[:, -2].view(3, 3)

    # Form cubic polynomial and solve for roots
    w, _ = torch.linalg.eig(F1)
    roots1 = torch.sort(w)[0]
    w, _ = torch.linalg.eig(F2)
    roots2 = torch.sort(w)[0]

    # Compute fundamental matrices
    F1 = (roots1[2] / roots1[1]) * F1
    F2 = (roots2[2] / roots2[1]) * F2

    # Handle edge cases
    try:
        torch.linalg.matrix_rank(F1)
    except LinAlgError:
        F1 = torch.eye(3)
    try:
        torch.linalg.matrix_rank(F2)
    except LinAlgError:
        F2 = torch.eye(3)

    # Concatenate solutions
    if torch.linalg.matrix_rank(F1) == 3 and torch.linalg.matrix_rank(F2) == 3:
        return torch.stack([F1, F2, torch.eye(3)])
    elif torch.linalg.matrix_rank(F1) == 3:
        return torch.stack([F1, torch.eye(3)])
    elif torch.linalg.matrix_rank(F2) == 3:
        return torch.stack([F2, torch.eye(3)])
    else:
        return torch.stack([torch.eye(3)])

if __name__ == "__main__":
    # Create sample input values
    points1 = torch.tensor([[[0.5, 0.5], [0.5, 0.75], [0.5, 0.25], [0.75, 0.75], [0.75, 0.25], [0.25, 0.75], [0.25, 0.25]]])
    points2 = torch.tensor([[[0.5, 0.4], [0.5, 0.6], [0.5, 0.1], [0.75, 0.6], [0.75, 0.1], [0.25, 0.6], [0.25, 0.1]]])

    # Call the function and print the results
    fundamentals = run_7point(points1, points2)
    print(fundamentals)