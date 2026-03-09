import torch

def masked_gather(points, idx):
    if points.shape[0] != idx.shape[0]:
        raise ValueError("Batch dimensions of points and idx must match.")
    
    idx = torch.where(idx == -1, torch.tensor(0, dtype=idx.dtype, device=idx.device), idx)
    gathered_points = points[torch.arange(points.shape[0]).unsqueeze(-1), idx]
    
    mask = (idx == 0)
    gathered_points[mask] = 0.0
    
    return gathered_points

if __name__ == "__main__":
    points = torch.tensor([
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
        [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]
    ])
    
    idx = torch.tensor([[0, 2, -1], [1, -1, -1]])  # Batch size is 2, with padding (-1)
    
    result = masked_gather(points, idx)
    print(result)