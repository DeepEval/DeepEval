import numpy as np
from typing import Union, List, Tuple, Optional
from pytorch3d.rasterizer import MeshRasterizer, Fragments
from pytorch3d.structures import Meshes
from pytorch3d.renderer import look_at_view_transform, FoVPerspectiveCameras, RasterizationSettings

def rasterize_meshes(meshes, image_size: Union[int, List[int], Tuple[int, int]] = 256, blur_radius: float = 0.0, faces_per_pixel: int = 8, bin_size: Optional[int] = None, max_faces_per_bin: Optional[int] = None, perspective_correct: bool = False, clip_barycentric_coords: bool = False, cull_backfaces: bool = False, z_clip_value: Optional[float] = None, cull_to_frustum: bool = False):
    # Convert image_size to tuple if it's an integer
    if isinstance(image_size, int):
        image_size = (image_size, image_size)
    
    # Create rasterization settings
    raster_settings = RasterizationSettings(
        image_size=image_size,
        blur_radius=blur_radius,
        faces_per_pixel=faces_per_pixel,
        bin_size=bin_size,
        max_faces_per_bin=max_faces_per_bin,
        perspective_correct=perspective_correct,
        clip_barycentric_coords=clip_barycentric_coords,
        cull_backfaces=cull_backfaces,
        z_clip_value=z_clip_value,
        cull_to_frustum=cull_to_frustum
    )
    
    # Create a rasterizer
    rasterizer = MeshRasterizer(raster_settings=raster_settings)
    
    # Rasterize the meshes
    fragments = rasterizer(meshes)
    
    # Extract the required outputs
    pix_to_face = fragments.pix_to_face
    zbuf = fragments.zbuf
    barycentric = fragments.bary_coords
    pix_dists = fragments.pix_dists
    
    return pix_to_face, zbuf, barycentric, pix_dists

if __name__ == "__main__":
    from pytorch3d.utils import xyz_to_mesh_face_uvs
    from pytorch3d.io import load_objs_as_meshes
    import torch

    # Load a sample mesh
    mesh = load_objs_as_meshes(["./data/cow.obj"])
    meshes = Meshes(mesh.verts_list(), mesh.faces_list())

    # Define some parameters
    image_size = (256, 256)
    blur_radius = 0.0
    faces_per_pixel = 8
    perspective_correct = False
    clip_barycentric_coords = False
    cull_backfaces = False
    z_clip_value = None
    cull_to_frustum = False

    # Call the function
    pix_to_face, zbuf, barycentric, pix_dists = rasterize_meshes(meshes, image_size, blur_radius, faces_per_pixel, perspective_correct, clip_barycentric_coords, cull_backfaces, z_clip_value, cull_to_frustum)

    # Print the results
    print("pix_to_face:", pix_to_face)
    print("zbuf:", zbuf)
    print("barycentric:", barycentric)
    print("pix_dists:", pix_dists)