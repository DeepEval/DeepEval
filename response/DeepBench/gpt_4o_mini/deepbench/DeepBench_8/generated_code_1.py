import torch
from pytorch3d.structures import Meshes

def cubify(voxels, thresh, feats=None, device='cpu', align='center'):
    N, D, H, W = voxels.shape
    occupied = voxels > thresh
    vertices = []
    faces = []
    textures = []

    for n in range(N):
        for d in range(D):
            for h in range(H):
                for w in range(W):
                    if occupied[n, d, h, w]:
                        v_offset = len(vertices)
                        if align == "topleft":
                            vertex_positions = [
                                [w, h, d],
                                [w + 1, h, d],
                                [w, h + 1, d],
                                [w + 1, h + 1, d],
                                [w, h, d + 1],
                                [w + 1, h, d + 1],
                                [w, h + 1, d + 1],
                                [w + 1, h + 1, d + 1],
                            ]
                        elif align == "corner":
                            vertex_positions = [
                                [w, h, d],
                                [w + 1, h, d],
                                [w, h + 1, d],
                                [w + 1, h + 1, d],
                                [w, h, d + 1],
                                [w + 1, h, d + 1],
                                [w, h + 1, d + 1],
                                [w + 1, h + 1, d + 1],
                            ]
                        elif align == "center":
                            vertex_positions = [
                                [w + 0.5, h + 0.5, d + 0.5],
                                [w + 1.5, h + 0.5, d + 0.5],
                                [w + 0.5, h + 1.5, d + 0.5],
                                [w + 1.5, h + 1.5, d + 0.5],
                                [w + 0.5, h + 0.5, d + 1.5],
                                [w + 1.5, h + 0.5, d + 1.5],
                                [w + 0.5, h + 1.5, d + 1.5],
                                [w + 1.5, h + 1.5, d + 1.5],
                            ]
                        vertices.extend(vertex_positions)

                        face_indices = [
                            [v_offset, v_offset + 1, v_offset + 2],
                            [v_offset + 1, v_offset + 3, v_offset + 2],
                            [v_offset + 4, v_offset + 5, v_offset + 6],
                            [v_offset + 5, v_offset + 7, v_offset + 6],
                            [v_offset, v_offset + 1, v_offset + 5],
                            [v_offset + 1, v_offset + 5, v_offset + 7],
                            [v_offset + 2, v_offset + 3, v_offset + 6],
                            [v_offset + 3, v_offset + 6, v_offset + 7],
                            [v_offset, v_offset + 2, v_offset + 4],
                            [v_offset + 2, v_offset + 4, v_offset + 6],
                            [v_offset + 1, v_offset + 3, v_offset + 5],
                            [v_offset + 3, v_offset + 5, v_offset + 7],
                        ]
                        faces.extend(face_indices)

                        if feats is not None:
                            texture = feats[n, :, d, h, w].unsqueeze(0)
                            textures.append(texture)

    vertices_tensor = torch.tensor(vertices, device=device).float()
    faces_tensor = torch.tensor(faces, device=device).long()
    mesh = Meshes(verts=[vertices_tensor], faces=[faces_tensor])
    
    if feats is not None:
        textures_tensor = torch.cat(textures, dim=0) if textures else None
        # Note: Assign textures to mesh here if needed, depending on how you'd like to handle it.

    return mesh

if __name__ == "__main__":
    voxels = torch.rand((1, 2, 2, 2))  # N=1, D=2, H=2, W=2
    voxels[0, 0, 0, 0] = 0.9  # Set some occupied voxels
    voxels[0, 1, 1, 1] = 0.9
    thresh = 0.5
    feats = torch.rand((1, 3, 2, 2, 2))  # Optional color information
    device = 'cpu'
    align = 'center'
    
    mesh = cubify(voxels, thresh, feats, device, align)
    print(mesh)