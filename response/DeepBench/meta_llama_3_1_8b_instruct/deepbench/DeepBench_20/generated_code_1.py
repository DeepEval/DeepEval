import numpy as np
import torch

def masked_gather(points, idx):
    # Check if points and idx have matching batch dimensions
    if points.shape[0]!= idx.shape[0]:
        raise ValueError("Batch dimensions of points and idx do not match")

    # Replace -1 indices with 0
    idx = torch.where(idx == -1, 0, idx)

    # Gather points based on indices
    gathered_points = torch.gather(points, 1, idx.unsqueeze(-1).expand(-1, -1, points.shape[2]))

    # Set gathered values corresponding to original -1 indices to 0.0
    gathered_points = torch.where(idx == -1, torch.zeros_like(gathered_points), gathered_points)

    return gathered_points

if __name__ == "__main__":
    # Create sample input values
    points = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], [[7.0, 8.0], [-1.0, 10.0], [11.0, 12.0]]])
    idx = torch.tensor([[0, 1, 2], [0, 2, 1]])

    # Call the function and print the results
    print(masked_gather(points, idx))