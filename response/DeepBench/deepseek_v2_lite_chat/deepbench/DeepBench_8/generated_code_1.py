import torch
from trimesh import meshes

def cubify(voxels, thresh, feats=None, device='cpu', align='topleft'):
    if feats is not None:
        assert len(voxels.shape) == 4, "Feats should be a tensor of shape (N, K, D, H, W)"
    
    # Create a tensor of 1's and 0's where 1's represent occupied voxels
    voxel_tensor = torch.zeros_like(voxels).bernoulli(0.5)
    voxel_tensor[voxels > thresh] = 1
    
    # Convert to torch.device if it's not already
    device = torch.device(device)
    
    # Find the vertices and faces
    vertices = torch.where(voxel_tensor.bool()).view(-1, 4)
    faces = torch.empty((vertices.shape[1],), dtype=torch.long)
    faces[::6] = vertices[0]
    faces[1::6] = vertices[1]
    faces[2::6] = vertices[2]
    faces[3::6] = vertices[3]
    
    # Merge shared vertices
    unique_vertices, indices = torch.unique(vertices, return_inverse=True)
    unique_faces = faces[indices]
    
    # Remove internal faces
    mask = torch.isin(unique_faces, indices[:3])
    unique_faces = unique_faces[mask]
    unique_vertices = unique_vertices[mask]
    
    # Convert to a Meshes object
    mesh = meshes.Mesh(vertices=unique_vertices, faces=unique_faces)
    
    # Align vertices to the topleft, corner, or center
    if align == 'topleft':
        mesh.vertices += torch.tensor([[-0.5, -0.5, -0.5]] * mesh.vertices.shape[0], device=device)
    elif align == 'corner':
        mesh.vertices += torch.tensor([[[0.5, 0.5, 0.5]] * 8] * mesh.vertices.shape[0], device=device)
    elif align == 'center':
        mesh.vertices += torch.tensor([[[0.0, 0.0, 0.0]] * 8] * mesh.vertices.shape[0], device=device)
    
    return mesh

if __name__ == "__main__":
    import numpy as np
    import random

    N = 10
    D = H = W = 5
    K = 3
    thresh = 0.5
    device = 'cpu'
    align = 'topleft'
    
    # Generate random voxel data
    voxels = torch.randint(0, 2, (N, D, H, W))
    
    # Generate random color features (K=3 for RGB)
    feats = torch.rand(N, K, D, H, W) * 255
    
    # Create a sample input
    sample_input = (voxels, thresh, feats, device, align)
    
    # Call the function
    cubified_mesh = cubify(*sample_input)
    
    # Print the results
    print("Cubified Mesh:")
    print(cubified_mesh)