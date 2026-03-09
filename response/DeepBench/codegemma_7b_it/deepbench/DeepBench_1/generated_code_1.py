import torch

def chamfer_distance(x, y, x_lengths=None, y_lengths=None, x_normals=None, y_normals=None, weights=None,
                 batch_reduction='mean', point_reduction='mean', norm=2, single_directional=False, abs_cosine=False):
    """Chamfer distance between two point clouds x and y.

    Args:
        x: (B, N, D) array, point cloud in N dimensions
        y: (B, M, D) array, point cloud in N dimensions
        x_lengths: (B,) array, lengths of point clouds in x
        y_lengths: (B,) array, lengths of point clouds in y
        x_normals: (B, N, D) array, normals of point clouds in x
        y_normals: (B, M, D) array, normals of point clouds in y
        weights: (B, N) or (B, M) array, weights for chamfer distance
        batch_reduction: {sum, mean}, reduction operation for the batch dimension
        point_reduction: {sum, mean, min, max}, reduction operation for the point dimension
        norm: int, p-norm used to calculate chamfer distance
        single_directional: bool, if True, calculate chamfer distance in one direction only
        abs_cosine: bool, if True, calculate chamfer distance using absolute values of cosine similarity

    Returns:
        (float, float) or (torch.Tensor, torch.Tensor), distance between two point clouds in x and y, chamfer distance of normals between x and y
    """

    if x_lengths is not None:
        x_lengths = x_lengths.float()
    if y_lengths is not None:
        y_lengths = y_lengths.float()

    # (B, N, D) -> (B, N, 1)
    x = x.unsqueeze(-1)
    # (B, M, D) -> (B, 1, M)
    y = y.unsqueeze(1)

    # (B, N, M)
    diff = x - y

    if x_normals is not None and y_normals is not None:
        # (B, N, D) -> (B, N, 1)
        x_normals = x_normals.unsqueeze(-1)
        # (B, M, D) -> (B, 1, M)
        y_normals = y_normals.unsqueeze(1)
        # (B, N, M)
        x_y_normals = torch.bmm(x_normals, y_normals.transpose(-2, -1))

        if abs_cosine:
            x_y_normals = torch.abs(x_y_normals)
        cosine_dist = torch.acos(x_y_normals)

        if x_lengths is not None and y_lengths is not None:
            x_lengths = x_lengths.view(-1, 1, 1)
            y_lengths = y_lengths.view(-1, 1, 1)
            cosine_dist = cosine_dist / torch.sqrt(x_lengths * y_lengths)

        chamfer_dist = torch.cat([x_lengths + cosine_dist - 2 * x_y_normals, y_lengths + cosine_dist - 2 * x_y_normals], dim=-1)
    else:
        # (B, N, M)
        if x_lengths is not None and y_lengths is not None:
            dist = x_lengths.view(-1, 1, 1) + y_lengths.view(-1, 1, 1) - 2 * torch.bmm(x, y.transpose(-2, -1))
        else:
            dist = x.view(-1, 1, 1) + y.view(-1, 1, 1) - 2 * torch.bmm(x, y.transpose(-2, -1))

        chamfer_dist = dist.view(-1, 2)

    if weights is not None:
        chamfer_dist = chamfer_dist * weights.unsqueeze(-1)

    if point_reduction == 'sum':
        chamfer_dist = chamfer_dist.sum(dim=-1)
    elif point_reduction == 'mean':
        chamfer_dist = chamfer_dist.mean(dim=-1)
    elif point_reduction == 'min':
        chamfer_dist, _ = chamfer_dist.min(dim=-1)
    elif point_reduction == 'max':
        chamfer_dist, _ = chamfer_dist.max(dim=-1)

    if single_directional:
        chamfer_dist = chamfer_dist.min(dim=-1)[0]

    if batch_reduction == 'sum':
        chamfer_dist = chamfer_dist.sum()
    elif batch_reduction == 'mean':
        chamfer_dist = chamfer_dist.mean()

    return chamfer_dist

if __name__ == "__main__":
    # Sample input values
    x = torch.randn(2, 5, 3)
    y = torch.randn(2, 7, 3)

    # Calculate chamfer distance
    chamfer_dist = chamfer_distance(x, y)

    # Print results
    print(chamfer_dist)