import torch
from pyvista import PolyData, Mesh

def cubify(voxels, thresh, feats=None, device='cpu', align='topleft'):
    voxels = voxels.to(device)
    occupied_indices = voxels > thresh
    
    vertices = []
    faces = []
    
    for i in range(voxels.shape[0]):
        for d in range(voxels.shape[1]):
            for h in range(voxels.shape[2]):
                for w in range(voxels.shape[3]):
                    if occupied_indices[i, d, h, w]:
                        vertex_indices = [
                            (w, h, d),
                            (w + 1, h, d),
                            (w + 1, h + 1, d),
                            (w, h + 1, d),
                            (w, h, d + 1),
                            (w + 1, h, d + 1),
                            (w + 1, h + 1, d + 1),
                            (w, h + 1, d + 1),
                        ]

                        if align == 'topleft':
                            offset = (w, h, d)
                        elif align == 'corner':
                            offset = (w, h, d + 0.5)
                        elif align == 'center':
                            offset = (w + 0.5, h + 0.5, d + 0.5)
                        else:
                            raise ValueError(f"Invalid alignment: {align}")

                        vertices.extend([torch.tensor(v + offset) for v in vertex_indices])
                        faces.extend([
                            [i*8 + j, i*8 + (j+1)%8, i*8 + (j+2)%8] for j in range(4)
                        ])

    vertices = torch.stack(vertices).cpu().numpy()
    faces = torch.stack(faces).cpu().numpy()

    mesh = PolyData(vertices, faces)

    return mesh
 
if __name__ == "__main__":
    voxels = torch.rand(1, 32, 32, 32)
    thresh = 0.5
    mesh = cubify(voxels, thresh)
    print(mesh)