from typing import Union, Optional, List, Tuple
from PIL import Image
import numpy as np
import cv2

def rasterize_meshes(meshes, image_size: Union[int, List[int], Tuple[int, int]] = (256, 256), blur_radius: float = 0.0, faces_per_pixel: int = 8, bin_size: Optional[int] = None, max_faces_per_bin: Optional[int] = None, perspective_correct: bool = False, clip_barycentric_coords: bool = False, cull_backfaces: bool = False, z_clip_value: Optional[float] = None, cull_to_frustum: bool = False):
    # Assuming each mesh is a list of vertices and faces, and each face is a list of indices
    vertices = np.array([mesh[0] for mesh in meshes])
    faces = np.array([mesh[1] for mesh in meshes])

    num_meshes = len(meshes)
    pix_to_face = np.zeros((image_size[1], image_size[0], 3), dtype=np.float32)
    zbuf = np.zeros((image_size[1], image_size[0]), dtype=np.float32)
    barycentric = np.zeros((image_size[1], image_size[0], 3), dtype=np.float32)
    pix_dists = np.zeros((image_size[1], image_size[0]), dtype=np.float32)

    if blur_radius > 0:
        blurred_image = cv2.GaussianBlur(np.array(Image.new('RGB', image_size, (0, 0, 0))), (blur_radius, blur_radius), 0)
        blurred_image = cv2.cvtColor(blurred_image, cv2.COLOR_RGB2BGR)
        blurred_image = Image.fromarray(blurred_image)
        pix_to_face = np.array(blurred_image.copy().convert('L'))
        zbuf = np.array(blurred_image.copy().convert('L'))
        barycentric = np.array(blurred_image.copy().convert('L'))
        pix_dists = np.array(blurred_image.copy().convert('L'))

    for i in range(num_meshes):
        mesh = meshes[i]
        vertices_mesh = vertices[faces == mesh[0]]
        num_vertices_mesh = len(vertices_mesh)

        if cull_to_frustum:
            frustum = cull_to_frustum(meshes, image_size, z_clip_value)
            valid_vertices = vertices_mesh[np.where(frustum[:, 2] >= 0)]
        else:
            valid_vertices = vertices_mesh

        if perspective_correct:
            valid_vertices = perspective_correct_vertices(valid_vertices)

        if clip_barycentric_coords:
            valid_vertices = clip_barycentric_coords_vertices(valid_vertices)

        if cull_backfaces:
            valid_vertices = cull_backfaces_vertices(valid_vertices)

        if num_vertices_mesh > 0:
            barycentric_mesh = compute_barycentric_coords(valid_vertices)
            distance = compute_point_distance(vertices_mesh)

            for j in range(num_vertices_mesh):
                x = valid_vertices[j, 0]
                y = valid_vertices[j, 1]
                z = valid_vertices[j, 2]
                if z_clip_value is not None and z < z_clip_value:
                    continue
                bary_x, bary_y, bary_z = barycentric_mesh[j]
                pix_x = int((x - bary_x) * image_size[0] / (2 * z) + image_size[0] / 2)
                pix_y = int((y - bary_y) * image_size[1] / (2 * z) + image_size[1] / 2)
                pix_dist = distance[j]

                if 0 <= pix_x < image_size[0] and 0 <= pix_y < image_size[1]:
                    pix_to_face[pix_y, pix_x, :] = i
                    zbuf[pix_y, pix_x] = z
                    barycentric[pix_y, pix_x, :] = (bary_x, bary_y, bary_z)
                    pix_dists[pix_y, pix_x] = pix_dist

    return pix_to_face, zbuf, barycentric, pix_dists

def cull_to_frustum(meshes, image_size, z_clip_value):
    frustum = np.zeros((len(meshes), 3))
    for i, mesh in enumerate(meshes):
        frustum[i, :2] = mesh[0][:2] / mesh[0][2]
    frustum = frustum.astype(int)
    return frustum[np.where(frustum[:, 2] >= 0)]

def perspective_correct_vertices(vertices):
    centroid = np.mean(vertices, axis=0)
    scale_factor = 1 / np.max(np.abs(vertices), axis=0)[2]
    return vertices * scale_factor

def cull_backfaces_vertices(vertices):
    return vertices[np.argmax(vertices[:, 2])]

def compute_barycentric_coords(vertices):
    num_vertices = len(vertices)
    barycentric = np.zeros((num_vertices, 3))
    for i in range(num_vertices):
        barycentric[i] = [0, 0, 0]
        for j in range(num_vertices):
            if i != j:
                barycentric[i] += vertices[j] / np.linalg.norm(vertices[j] - vertices[i])
    return barycentric

def compute_point_distance(vertices):
    return np.sqrt((vertices[:, 0] - vertices[:, 0][0])**2 + (vertices[:, 1] - vertices[:, 1][0])**2)

if __name__ == "__main__":
    meshes = [
        ([0, 0, 1], [0, 0, 1, 1]),
        ([1, 1, 1], [1, 0, 1, 1]),
        ([2, 2, 1], [0, 1, 1, 1])
    ]
    image_size = (512, 512)
    output = rasterize_meshes(meshes, image_size)
    print(output)