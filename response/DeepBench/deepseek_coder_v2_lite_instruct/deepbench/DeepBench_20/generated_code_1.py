import torch

def masked_gather(points, idx):
    if points.shape[:-1] != idx.shape[:-1]:
        raise ValueError("points and idx must have matching batch dimensions")
    
    # Replace -1 indices with 0
    mask = idx == -1
    idx = idx.clone()
    idx[mask] = 0
    
    # Gather points
    gathered_points = torch.gather(points, -1, idx)
    
    # Set gathered values corresponding to original -1 indices to 0.0
    gathered_points[mask] = 0.0
    
    return gathered_points

if __name__ == "__main__":
    points = torch.tensor([
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
        [[10.0, 11.0, 12.0], [13.0, 14.0, 15.0], [16.0, 17.0, 18.0]]
    ])
    
    idx = torch.tensor([
        [0, 1, -1],
        [1, 2, -1]
    ])
    
    result = masked_gather(points, idx)
    print(result)