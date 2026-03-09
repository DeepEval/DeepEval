import torch
import trimesh
from pytorch3d.structures import Meshes
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.transforms import Transform3d

def cubify(voxels, thresh, feats=None, device='cpu', align='center'):
    N, D, H, W = voxels.shape
    vertices = []
    faces = []
    vertex_to_face_mapping = {}

    def add_vertex(v):
        if v not in vertex_to_face_mapping:
            vertices.append(v)
            vertex_to_face_mapping[v] = len(vertices) - 1
        return vertex_to_face_mapping[v]

    for n in range(N):
        for d in range(D):
            for h in range(H):
                for w in range(W):
                    if voxels[n, d, h, w] > thresh:
                        v0 = [w, h, d]
                        v1 = [w, h+1, d]
                        v2 = [w+1, h+1, d]
                        v3 = [w+1, h, d]
                        v4 = [w, h, d+1]
                        v5 = [w, h+1, d+1]
                        v6 = [w+1, h+1, d+1]
                        v7 = [w+1, h, d+1]

                        if align == 'topleft':
                            pass
                        elif align == 'corner':
                            v0 = [w, H-h-1, D-d-1]
                            v1 = [w, H-h, D-d-1]
                            v2 = [w+1, H-h, D-d-1]
                            v3 = [w+1, H-h-1, D-d-1]
                            v4 = [w, H-h-1, D-d]
                            v5 = [w, H-h, D-d]
                            v6 = [w+1, H-h, D-d]
                            v7 = [w+1, H-h-1, D-d]
                        elif align == 'center':
                            v0 = [w - 0.5, h - 0.5, d - 0.5]
                            v1 = [w - 0.5, h + 0.5, d - 0.5]
                            v2 = [w + 0.5, h + 0.5, d - 0.5]
                            v3 = [w + 0.5, h - 0.5, d - 0.5]
                            v4 = [w - 0.5, h - 0.5, d + 0.5]
                            v5 = [w - 0.5, h + 0.5, d + 0.5]
                            v6 = [w + 0.5, h + 0.5, d + 0.5]
                            v7 = [w + 0.5, h - 0.5, d + 0.5]

                        for i in range(12):
                            face = []
                            if i < 4:
                                face = [v0, v1, v2, v3][i % 4]
                            else:
                                face = [v4, v5, v6, v7][i % 4]
                            face_verts = [add_vertex(face[0]), add_vertex(face[1]), add_vertex(face[2]), add_vertex(face[3])]
                            faces.append(face_verts)

    vertices = torch.tensor(vertices, device=device).float()
    faces = torch.tensor(faces, device=device).int()
    meshes = Meshes(verts=[vertices], faces=[faces])

    if feats is not None:
        meshes.textures = Meshes(verts=[feats.view(N, -1, 3)], faces=[faces])

    return meshes

if __name__ == "__main__":
    voxels = torch.rand(2, 3, 4, 4)
    thresh = 0.5
    meshes = cubify(voxels, thresh, device='cpu', align='center')
    print(meshes)