import torch
import numpy as np

def depth_to_normals(depth, camera_matrix, normalize_points):
    if not isinstance(depth, torch.Tensor):
        raise TypeError("depth must be a Tensor")
    if not isinstance(camera_matrix, torch.Tensor):
        raise TypeError("camera_matrix must be a Tensor")
    if depth.shape!= (depth.shape[0], 1, depth.shape[2], depth.shape[3]):
        raise ValueError("depth must have shape (B, 1, H, W)")
    if camera_matrix.shape!= (camera_matrix.shape[0], 3, 3):
        raise ValueError("camera_matrix must have shape (B, 3, 3)")
    
    B, _, H, W = depth.shape
    device = depth.device
    
    # Compute the gradients of the depth map
    gradients_x = torch.zeros((B, 1, H, W), device=device)
    gradients_y = torch.zeros((B, 1, H, W), device=device)
    for b in range(B):
        for y in range(H):
            for x in range(W):
                if x < W - 1:
                    gradients_x[b, 0, y, x] = depth[b, 0, y, x+1] - depth[b, 0, y, x]
                if y < H - 1:
                    gradients_y[b, 0, y, x] = depth[b, 0, y+1, x] - depth[b, 0, y, x]
    
    # Compute the surface normals
    normals = torch.zeros((B, 3, H, W), device=device)
    for b in range(B):
        for y in range(H):
            for x in range(W):
                if x < W - 1 and y < H - 1:
                    grad_x = gradients_x[b, 0, y, x]
                    grad_y = gradients_y[b, 0, y, x]
                    grad_z = -1  # assuming the camera is looking at the depth image from the front
                    
                    normal_x = grad_y
                    normal_y = -grad_x
                    normal_z = grad_z
                    
                    # Normalize the normal vector
                    if normalize_points:
                        magnitude = torch.sqrt(normal_x ** 2 + normal_y ** 2 + normal_z ** 2)
                        normals[b, 0, y, x] = normal_x / magnitude
                        normals[b, 1, y, x] = normal_y / magnitude
                        normals[b, 2, y, x] = normal_z / magnitude
                    else:
                        normals[b, 0, y, x] = normal_x
                        normals[b, 1, y, x] = normal_y
                        normals[b, 2, y, x] = normal_z
    
    return normals

if __name__ == "__main__":
    # Create some sample input values
    B = 1
    H = 10
    W = 10
    
    depth = torch.randn(B, 1, H, W)
    camera_matrix = torch.randn(B, 3, 3)
    
    normalize_points = True
    
    # Call the function and print the results
    normals = depth_to_normals(depth, camera_matrix, normalize_points)
    print(normals)