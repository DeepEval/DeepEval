import torch
from meshes import Meshes

def cubify(voxels, thresh, feats=None, device="cuda", align="topleft"):
    # Convert voxels to a mesh
    mesh = voxels.to(device)

    # Threshold voxels
    mesh = mesh > thresh

    # Get the mesh vertices and faces
    vertices, faces = mesh.to_vertices_and_faces()

    # Compute the vertex normals
    vertex_normals = torch.cross(vertices[faces[:, 1]] - vertices[faces[:, 0]],
                                 vertices[faces[:, 2]] - vertices[faces[:, 0]])

    # Compute the face normals
    face_normals = torch.cross(vertices[faces[:, 1]] - vertices[faces[:, 0]],
                               vertices[faces[:, 2]] - vertices[faces[:, 0]])

    # Compute the cube vertices
    cube_vertices = torch.cat([vertices, vertices + face_normals], dim=1)

    # Compute the cube faces
    cube_faces = torch.cat([faces, faces + 8], dim=0)

    # Merge shared vertices
    unique_vertices, unique_indices = torch.unique(cube_vertices, dim=0, return_inverse=True)

    # Remove internal faces
    cube_faces = cube_faces[unique_indices]

    # Compute the final mesh
    mesh = Meshes(unique_vertices, cube_faces)

    # Optionally add color information
    if feats is not None:
        mesh.add_attribute("colors", feats)

    # Return the final mesh
    return mesh

if __name__ == "__main__":
    # Create sample input values
    voxels = torch.randn(1, 3, 3, 3)
    thresh = 0.5
    feats = torch.randn(1, 3, 3, 3)
    device = "cuda"
    align = "topleft"

    # Call the function and print the results
    mesh = cubify(voxels, thresh, feats, device, align)
    print(mesh.vertices)
    print(mesh.faces)