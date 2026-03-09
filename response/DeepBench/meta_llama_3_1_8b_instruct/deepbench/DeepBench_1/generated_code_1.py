import numpy as np
import torch
import torch.nn.functional as F

def chamfer_distance(x, y, x_lengths=None, y_lengths=None, x_normals=None, y_normals=None, weights=None, 
                     batch_reduction='mean', point_reduction='mean', norm=True, single_directional=False, 
                     abs_cosine=False):
    # Calculate the Chamfer distance
    dist_x_to_y, dist_y_to_x = F.cosine_similarity(x.unsqueeze(2), y.unsqueeze(1), dim=1, eps=1e-8)
    if single_directional:
        dist = dist_x_to_y
    else:
        dist = torch.cat((dist_x_to_y, dist_y_to_x), dim=2)
    
    # Apply weights and reduce distance
    if weights is not None:
        dist = weights * dist
    dist = torch.sum(dist, dim=2)
    
    if x_lengths is not None and y_lengths is not None:
        mask = torch.zeros_like(dist)
        mask[np.arange(len(x_lengths)), np.arange(len(x_lengths))] = 1
        mask[np.arange(len(y_lengths)), np.arange(len(y_lengths))] = 1
        dist = dist * mask
        if batch_reduction =='mean':
            dist = torch.sum(dist, dim=1) / (x_lengths.unsqueeze(1) + y_lengths.unsqueeze(1) - 2 * torch.eye(len(x_lengths)))
        elif batch_reduction =='sum':
            dist = torch.sum(dist, dim=1)
        else:
            raise ValueError("Invalid batch_reduction value")
    else:
        if batch_reduction =='mean':
            dist = torch.mean(dist, dim=1, keepdim=True)
        elif batch_reduction =='sum':
            dist = torch.sum(dist, dim=1, keepdim=True)
    
    # Reduce distance along points
    if point_reduction =='mean':
        dist = torch.mean(dist, dim=1)
    elif point_reduction =='sum':
        dist = torch.sum(dist, dim=1)
    
    # Apply L1 norm
    if norm:
        dist = torch.abs(dist)
    
    # Calculate cosine distance of normals
    if x_normals is not None and y_normals is not None:
        normals_dist = F.cosine_similarity(x_normals, y_normals, dim=1, eps=1e-8)
    else:
        normals_dist = torch.zeros_like(dist)
    
    # Reduce cosine distance of normals
    if x_lengths is not None and y_lengths is not None:
        normals_dist = torch.sum(normals_dist, dim=1)
    else:
        normals_dist = torch.mean(normals_dist, dim=1, keepdim=True)
    
    # Apply absolute value and cosine of cosine distance
    if abs_cosine:
        normals_dist = torch.abs(normals_dist)
    
    return dist, normals_dist

if __name__ == "__main__":
    # Generate sample point clouds
    np.random.seed(0)
    x = np.random.rand(10, 3)
    y = np.random.rand(10, 3)
    
    # Convert to tensors
    x = torch.tensor(x, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)
    
    # Create sample normals
    x_normals = torch.tensor(np.random.rand(10, 3), dtype=torch.float32)
    y_normals = torch.tensor(np.random.rand(10, 3), dtype=torch.float32)
    
    # Create sample weights
    weights = torch.tensor(np.random.rand(10, 10), dtype=torch.float32)
    
    # Set optional parameters
    x_lengths = torch.tensor([10], dtype=torch.int32)
    y_lengths = torch.tensor([10], dtype=torch.int32)
    
    # Calculate Chamfer distance
    dist, normals_dist = chamfer_distance(x, y, x_lengths, y_lengths, x_normals, y_normals, weights, 
                                          batch_reduction='mean', point_reduction='mean', norm=True, 
                                          single_directional=False, abs_cosine=False)
    
    # Print results
    print("Chamfer Distance:", dist.item())
    print("Normals Distance:", normals_dist.item())