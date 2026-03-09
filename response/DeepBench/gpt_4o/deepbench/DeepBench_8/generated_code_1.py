import torch
from torch import nn
from torch import Tensor
from pytorch3d.structures import Meshes
from pytorch3d.ops import cubify

def cubify(voxels: Tensor, thresh: float, feats: Tensor = None, device: str = 'cpu', align: str = 'topleft') -> Meshes:
    binary_voxels = (voxels > thresh).float()

    if align not in ["topleft", "corner", "center"]:
        raise ValueError("align must be one of 'topleft', 'corner', 'center'")

    if align == "topleft":
        offset = 0.0
    elif align == "corner":
        offset = -0.5
    elif align == "center":
        offset = -0.5 + 0.5 * binary_voxels.size(-1) / binary_voxels.size(-1)

    meshes = []
    for i in range(binary_voxels.size(0)):
        voxel = binary_voxels[i]
        verts, faces = cubify(voxel, thresh)
        
        verts += offset

        if feats is not None:
            feats_per_vertex = feats[i].permute(1, 2, 3, 0).reshape(-1, feats.size(1))
            meshes.append(Meshes(verts=verts.to(device), faces=faces.to(device), textures=feats_per_vertex.to(device)))
        else:
            meshes.append(Meshes(verts=verts.to(device), faces=faces.to(device)))

    return meshes

if __name__ == "__main__":
    N, D, H, W = 1, 3, 3, 3  # Dimensions for the voxel grid
    K = 3  # Number of features for each voxel

    voxels = torch.rand(N, D, H, W)
    feats = torch.rand(N, K, D, H, W)
    thresh = 0.5

    device = 'cpu'
    align = 'center'

    mesh = cubify(voxels, thresh, feats, device, align)
    print(mesh)