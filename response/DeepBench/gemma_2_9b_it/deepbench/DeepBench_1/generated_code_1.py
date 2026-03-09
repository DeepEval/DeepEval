import torch
from torch.nn import functional as F

def chamfer_distance(x, y, x_lengths=None, y_lengths=None, x_normals=None, y_normals=None, weights=None, batch_reduction='mean', point_reduction='mean', norm='l2', single_directional=False, abs_cosine=False):
    x_distances = torch.cdist(x, y, p=2)
    y_distances = torch.cdist(y, x, p=2)

    if point_reduction == 'mean':
        x_distances = x_distances.mean(dim=1)
        y_distances = y_distances.mean(dim=1)
    elif point_reduction == 'min':
        x_distances = x_distances.min(dim=1)[0]
        y_distances = y_distances.min(dim=1)[0]

    if batch_reduction == 'mean':
        chamfer_dist = (x_distances.mean() + y_distances.mean()) / 2
    elif batch_reduction == 'sum':
        chamfer_dist = (x_distances.sum() + y_distances.sum()) / 2

    if x_normals is not None and y_normals is not None:
        if abs_cosine:
            cosine_dist = 1 - torch.abs(torch.cosine_similarity(x_normals, y_normals, dim=1))
        else:
            cosine_dist = 1 - torch.cosine_similarity(x_normals, y_normals, dim=1)

        if point_reduction == 'mean':
            cosine_dist = cosine_dist.mean(dim=1)
        elif point_reduction == 'min':
            cosine_dist = cosine_dist.min(dim=1)[0]

        if batch_reduction == 'mean':
            cosine_dist = cosine_dist.mean()
        elif batch_reduction == 'sum':
            cosine_dist = cosine_dist.sum()

        return chamfer_dist, cosine_dist
    else:
        return chamfer_dist, None

if __name__ == "__main__":
    x = torch.randn(10, 3)
    y = torch.randn(15, 3)
    chamfer_dist, cosine_dist = chamfer_distance(x, y)
    print(f"Chamfer Distance: {chamfer_dist}")
    print(f"Cosine Distance: {cosine_dist}")