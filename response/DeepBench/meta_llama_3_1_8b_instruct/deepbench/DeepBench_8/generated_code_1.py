import torch
import pytorch3d
from pytorch3d import meshes
import numpy as np

def cubify(voxels, thresh, feats=None, device=None, align='topleft'):
    N, D, H, W = voxels.shape
    if feats is not None:
        K, _, _, _ = feats.shape
        assert K == N
    device = voxels.device if device is None else device

    # Create a tensor to store the mesh vertices
    vertices = torch.zeros((N * 8, 3), device=device)
    faces = torch.zeros((N * 36, 3), device=device)

    i = 0
    for n in range(N):
        for d in range(D):
            for h in range(H):
                for w in range(W):
                    if voxels[n, d, h, w] >= thresh:
                        # Calculate the vertex coordinates
                        if align == 'topleft':
                            v = torch.tensor([w, h, d], device=device)
                        elif align == 'center':
                            v = torch.tensor([(w + 0.5), (h + 0.5), (d + 0.5)], device=device)
                        elif align == 'corner':
                            v = torch.tensor([w + 0.5, h + 0.5, d + 0.5], device=device)
                        else:
                            raise ValueError('Invalid align value')

                        v = v * 1.0 / (W - 1) * 2 - 1

                        vertices[i * 8:(i + 1) * 8] = v.repeat(8)
                        i += 1

    # Create faces for each cube
    face_offset = torch.tensor([0, 4, 1, 5, 2, 6, 3, 7, 0, 4, 5, 1, 0, 1, 2, 3, 0, 3, 7, 4, 5, 6, 1, 2, 6, 7, 3, 4, 5, 6, 2, 7]).repeat(N).view(-1, 3)
    faces[:] = face_offset + (torch.arange(N) * 36).view(-1, 1).repeat(1, 3)

    # Create a Meshes object
    meshes_obj = meshes.Meshes(vertices, faces)
    return meshes_obj

if __name__ == "__main__":
    # Create sample input values
    voxels = torch.tensor([[[[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8], [0.9, 0.1, 0.2, 0.3]], 
                            [[0.4, 0.5, 0.6, 0.7], [0.8, 0.9, 0.1, 0.2], [0.3, 0.4, 0.5, 0.6]]]], 
                          device=torch.device('cpu'))
    thresh = 0.5
    feats = torch.tensor([[[[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8], [0.9, 0.1, 0.2, 0.3]], 
                            [[0.4, 0.5, 0.6, 0.7], [0.8, 0.9, 0.1, 0.2], [0.3, 0.4, 0.5, 0.6]]]], 
                          device=torch.device('cpu'))
    device = torch.device('cpu')
    align = 'topleft'

    # Call the function
    meshes_obj = cubify(voxels, thresh, feats, device, align)

    # Print the results
    print(meshes_obj)