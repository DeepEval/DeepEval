import torch

def masked_gather(points, idx):
    if points.shape[0] != idx.shape[0]:
        raise ValueError("Batch dimensions of points and idx must match.")
    
    # Replace -1 with 0 in idx
    idx_replaced = torch.where(idx == -1, torch.tensor(0, dtype=idx.dtype), idx)
    
    # Gather points using the modified indices
    gathered_points = torch.gather(points, 1, idx_replaced.unsqueeze(-1).expand(-1, -1, points.size(-1)))
    
    # Create a mask for the indices that were originally -1
    mask = (idx == -1).unsqueeze(-1).expand(-1, -1, points.size(-1))
    
    # Zero out the gathered points where the original indices were -1
    gathered_points = torch.where(mask, torch.tensor(0.0, dtype=points.dtype), gathered_points)
    
    return gathered_points

if __name__ == "__main__":
    # Example usage
    points = torch.tensor([
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
        [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]
    ])
    
    idx = torch.tensor([
        [2, 0, -1],
        [1, -1, 0]
    ])
    
    result = masked_gather(points, idx)
    print(result)