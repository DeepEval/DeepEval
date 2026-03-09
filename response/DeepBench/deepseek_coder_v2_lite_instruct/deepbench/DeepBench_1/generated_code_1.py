import torch

def chamfer_distance(x, y, x_lengths=None, y_lengths=None, x_normals=None, y_normals=None, weights=None, batch_reduction='mean', point_reduction='mean', norm=2, single_directional=False, abs_cosine=False):
    if x.size(0) == 0 or y.size(0) == 0:
        return torch.tensor(0.0), torch.tensor(0.0)

    if x_normals is not None and y_normals is not None:
        x_normals = x_normals.unsqueeze(0).repeat(y.size(0), 1, 1)
        y_normals = y_normals.unsqueeze(1).repeat(x.size(0), 1, 1)

    x_expanded = x.unsqueeze(1).repeat(1, y.size(1), 1)
    y_expanded = y.unsqueeze(0).repeat(x.size(1), 1, 1)

    dist = torch.norm(x_expanded - y_expanded, p=norm, dim=-1)

    if weights is not None:
        dist = dist * weights.unsqueeze(0).unsqueeze(0).repeat(x.size(0), y.size(0), 1)

    if x_lengths is not None and y_lengths is not None:
        dist = dist.sum(dim=-1) / (x_lengths.unsqueeze(1).repeat(1, y.size(0)) + 1e-8)
        dist = dist.sum(dim=-1) / (y_lengths.unsqueeze(0).repeat(x.size(0), 1) + 1e-8)
    else:
        dist = dist.sum(dim=-1) / (x.size(1) * y.size(1))

    if single_directional:
        dist = torch.min(dist, dist.transpose(0, 1))

    if batch_reduction == 'mean':
        dist = dist.mean()
    elif batch_reduction == 'sum':
        dist = dist.sum()

    if x_normals is not None and y_normals is not None:
        cosine_dist = 1 - (x_normals * y_normals).sum(dim=-1)
        if weights is not None:
            cosine_dist = cosine_dist * weights.unsqueeze(0).unsqueeze(0).repeat(x.size(0), y.size(0), 1)
        if point_reduction == 'mean':
            cosine_dist = cosine_dist.mean(dim=-1)
        elif point_reduction == 'sum':
            cosine_dist = cosine_dist.sum(dim=-1)
        if batch_reduction == 'mean':
            cosine_dist = cosine_dist.mean()
        elif batch_reduction == 'sum':
            cosine_dist = cosine_dist.sum()
        if abs_cosine:
            cosine_dist = cosine_dist.abs()

    return dist, cosine_dist

if __name__ == "__main__":
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    y = torch.tensor([[1.1, 2.1, 3.1], [4.1, 5.1, 6.1]])
    dist, cosine_dist = chamfer_distance(x, y)
    print(f"Chamfer Distance: {dist}")
    print(f"Cosine Distance of Normals: {cosine_dist}")