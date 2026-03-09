import torch

def compute_fundamental_matrix(points1, points2, weights):
    B, N, _ = points1.shape
    A = []

    for i in range(N):
        x1, y1 = points1[:, i]
        x2, y2 = points2[:, i]
        A.append([x2 * x1, x2 * y1, x2, y2 * x1, y2 * y1, y2, x1, y1, 1])

    A = torch.tensor(A).reshape(N, 9)
    A = A * weights.unsqueeze(1)

    _, _, Vt = torch.linalg.svd(A)
    F_flat = Vt[-1]
    F = F_flat.reshape(3, 3)

    # Enforce rank 2 constraint
    U, S, Vt = torch.linalg.svd(F)
    S[-1] = 0
    F = U @ torch.diag(S) @ Vt

    return F

if __name__ == "__main__":
    # Example input values
    points1 = torch.tensor([[[1, 2], [3, 4], [5, 6], [7, 8]]])
    points2 = torch.tensor([[[1, 2], [3, 4], [5, 6], [7, 8]]])
    weights = torch.tensor([[1, 1, 1, 1]])

    # Call the function
    F = compute_fundamental_matrix(points1, points2, weights)

    # Print the results
    print(F)