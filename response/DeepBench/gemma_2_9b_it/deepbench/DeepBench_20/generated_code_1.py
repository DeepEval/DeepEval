import torch

def masked_gather(points, idx):
    if points.dim() != idx.dim() or points.size(0) != idx.size(0):
      raise ValueError("Points and idx must have matching batch dimensions")
    idx = idx.clone()
    idx[idx == -1] = 0
    gathered_points = torch.gather(points, dim=1, index=idx.unsqueeze(-1).repeat(1, 1, points.size(-1)))
    gathered_points[idx == 0] = 0.0
    return gathered_points

if __name__ == "__main__":
    points = torch.randn(4, 3)
    idx = torch.tensor([[0, 1, -1], [2, -1, 0], [1, 0, 2], [-1, 2, 1]])
    gathered_points = masked_gather(points, idx)
    print(gathered_points)