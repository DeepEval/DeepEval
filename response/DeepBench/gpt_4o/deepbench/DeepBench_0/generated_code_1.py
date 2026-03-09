import torch
from pytorch3d.renderer import (
    RasterizationSettings,
    MeshRasterizer,
    Meshes,
    PerspectiveCameras,
)
from pytorch3d.structures import Meshes
from typing import Union, List, Tuple, Optional

def rasterize_meshes(
    meshes: Meshes,
    image_size: Union[int, List[int], Tuple[int, int]] = 256,
    blur_radius: float = 0.0,
    faces_per_pixel: int = 8,
    bin_size: Optional[int] = None,
    max_faces_per_bin: Optional[int] = None,
    perspective_correct: bool = False,
    clip_barycentric_coords: bool = False,
    cull_backfaces: bool = False,
    z_clip_value: Optional[float] = None,
    cull_to_frustum: bool = False,
):
    device = meshes.device
    cameras = PerspectiveCameras(device=device)
    
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
        cull_to_frustum=cull_to_frustum,
    )
    
    rasterizer = MeshRasterizer(
        cameras=cameras,
        raster_settings=raster_settings
    )
    
    fragments = rasterizer(meshes)
    pix_to_face = fragments.pix_to_face
    zbuf = fragments.zbuf
    barycentric = fragments.bary_coords
    pix_dists = fragments.dists

    return pix_to_face, zbuf, barycentric, pix_dists

if __name__ == "__main__":
    # Create a simple mesh
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    verts = torch.tensor(
        [
            [-1, -1, 0],
            [1, -1, 0],
            [0, 1, 0],
        ], dtype=torch.float32, device=device
    )
    faces = torch.tensor(
        [
            [0, 1, 2],
        ], dtype=torch.int64, device=device
    )
    mesh = Meshes(verts=[verts], faces=[faces])
    
    # Rasterize the mesh
    pix_to_face, zbuf, barycentric, pix_dists = rasterize_meshes(mesh, image_size=256)
    
    # Print the results
    print("pix_to_face:", pix_to_face)
    print("zbuf:", zbuf)
    print("barycentric:", barycentric)
    print("pix_dists:", pix_dists)