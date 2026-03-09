import torch
import torch.nn.functional as F

def chamfer_distance(x, y, x_lengths=None, y_lengths=None, x_normals=None, y_normals=None,
                    weights=(1.0, 1.0), batch_reduction='mean', point_reduction='mean', 
                    norm='euclidean', single_directional=False, abs_cosine=False):
    
    if x_lengths is not None:
        x = [x[i][:x_lengths[i]] for i in range(len(x_lengths))]
    if y_lengths is not None:
        y = [y[i][:y_lengths[i]] for i in range(len(y_lengths))]
    
    x = torch.cat(x, dim=0) if isinstance(x, list) else x
    y = torch.cat(y, dim=0) if isinstance(y, list) else y

    dist_x_to_y = torch.cdist(x, y)
    dist_y_to_x = torch.cdist(y, x)
    
    chamfer_dist_x = torch.mean(torch.min(dist_x_to_y, dim=1)[0])
    chamfer_dist_y = torch.mean(torch.min(dist_y_to_x, dim=1)[0])
    
    chamfer_distance = (weights[0] * chamfer_dist_x + weights[1] * chamfer_dist_y) / sum(weights)

    if x_normals is not None and y_normals is not None:
        x_normals = torch.cat(x_normals, dim=0) if isinstance(x_normals, list) else x_normals
        y_normals = torch.cat(y_normals, dim=0) if isinstance(y_normals, list) else y_normals
        
        if abs_cosine:
            cos_similarity = F.cosine_similarity(x_normals, y_normals, dim=-1)
            cosine_distance = 1 - cos_similarity.mean()
        else:
            cosine_distance = 1 - (F.normalize(x_normals, dim=-1) * F.normalize(y_normals, dim=-1)).sum(dim=-1).mean()

        return chamfer_distance.item(), cosine_distance.item()
    
    return chamfer_distance.item(), None

if __name__ == "__main__":
    x = torch.rand(10, 3)  # Point cloud x with 10 points
    y = torch.rand(12, 3)  # Point cloud y with 12 points
    x_normals = torch.rand(10, 3)  # Normals for point cloud x
    y_normals = torch.rand(12, 3)  # Normals for point cloud y
    
    chamfer_dist, cosine_dist = chamfer_distance(
        x, y,
        x_normals=x_normals,
        y_normals=y_normals,
        weights=(1.0, 1.0),
        batch_reduction='mean',
        point_reduction='mean',
        abs_cosine=True
    )
    
    print(f"Chamfer Distance: {chamfer_dist}")
    print(f"Cosine Distance: {cosine_dist}")