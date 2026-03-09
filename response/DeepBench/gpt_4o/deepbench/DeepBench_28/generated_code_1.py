import torch
import torch.nn.functional as F

def depth_to_normals(depth, camera_matrix, normalize_points):
    if not isinstance(depth, torch.Tensor) or not isinstance(camera_matrix, torch.Tensor):
        raise TypeError("Both depth and camera_matrix must be Tensors.")
    
    if depth.dim() != 4 or depth.size(1) != 1:
        raise ValueError("depth must have the shape (B, 1, H, W).")
    
    if camera_matrix.dim() != 3 or camera_matrix.size(1) != 3 or camera_matrix.size(2) != 3:
        raise ValueError("camera_matrix must have the shape (B, 3, 3).")
    
    B, _, H, W = depth.shape
    
    u, v = torch.meshgrid(torch.arange(W), torch.arange(H), indexing='xy')
    u = u.to(depth.device).float()
    v = v.to(depth.device).float()
    
    fx = camera_matrix[:, 0, 0].unsqueeze(1).unsqueeze(2)
    fy = camera_matrix[:, 1, 1].unsqueeze(1).unsqueeze(2)
    cx = camera_matrix[:, 0, 2].unsqueeze(1).unsqueeze(2)
    cy = camera_matrix[:, 1, 2].unsqueeze(1).unsqueeze(2)
    
    x = (u - cx) * depth / fx
    y = (v - cy) * depth / fy
    z = depth
    
    if normalize_points:
        norm = torch.sqrt(x**2 + y**2 + z**2)
        x = x / norm
        y = y / norm
        z = z / norm
    
    points = torch.stack((x, y, z), dim=1)
    
    dzdx = (z[:, :, :, 2:] - z[:, :, :, :-2]) / 2
    dzdy = (z[:, :, 2:, :] - z[:, :, :-2, :]) / 2

    dx = torch.zeros_like(z)
    dy = torch.zeros_like(z)
    
    dx[:, :, :, 1:-1] = dzdx
    dy[:, :, 1:-1, :] = dzdy
    
    normals = torch.cat((dx, dy, torch.ones_like(z)), dim=1)
    normals = F.normalize(normals, p=2, dim=1)
    
    return normals

if __name__ == "__main__":
    B, H, W = 1, 5, 5
    depth = torch.rand(B, 1, H, W)
    camera_matrix = torch.tensor([[[1.0, 0.0, 2.5], [0.0, 1.0, 2.5], [0.0, 0.0, 1.0]]])
    camera_matrix = camera_matrix.expand(B, -1, -1)
    
    normals = depth_to_normals(depth, camera_matrix, normalize_points=True)
    print(normals)