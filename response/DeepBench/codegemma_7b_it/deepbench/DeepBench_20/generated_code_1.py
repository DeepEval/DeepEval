import torch

def masked_gather(points, idx):
    assert points.shape[:-1] == idx.shape, "idx and points shape do not match"
    idx = torch.where(idx == -1, 0, idx)
    gathered_points = torch.gather(points, -1, idx.long())
    gathered_points.masked_fill_(idx == 0, 0.0)
    return gathered_points

if __name__ == "__main__":
    points = torch.randn(4, 10, 7)
    idx = torch.randint(0, 10, (4, 8))
    idx[:, 2] = -1
    idx[:, 4] = -1
    output = masked_gather(points, idx)
    print(output.shape)