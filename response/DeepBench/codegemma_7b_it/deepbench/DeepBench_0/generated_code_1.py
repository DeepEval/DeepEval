import numpy as np
import trimesh

def rasterize_meshes(meshes, image_size=(256, 256), blur_radius=0.0, faces_per_pixel=8, bin_size=None, max_faces_per_bin=None, perspective_correct=False, clip_barycentric_coords=False, cull_backfaces=False, z_clip_value=None, cull_to_frustum=False):
    """
    Rasterizes a batch of meshes onto separate images based on the desired output image shape.

    Args:
        meshes: A batch of meshes.
        image_size: Size of the output image in pixels.
        blur_radius: Radius of the Gaussian blur to apply to the rendered images.
        faces_per_pixel: Number of faces to sample at each pixel.
        bin_size: Size of the bins for face rasterization.
        max_faces_per_bin: Maximum number of faces to include in each bin.
        perspective_correct: Whether to perform perspective correction.
        clip_barycentric_coords: Whether to clip barycentric coordinates outside the [-1, 1] range.
        cull_backfaces: Whether to cull faces that are facing away from the camera.
        z_clip_value: Value to use for clipping faces outside the view frustum.
        cull_to_frustum: Whether to cull faces outside the view frustum.

    Returns:
        A 4-element tuple containing:
            - pix_to_face: Indices of the nearest faces at each pixel.
            - zbuf: NDC z-coordinates of the nearest faces at each pixel.
            - barycentric: Barycentric coordinates of the nearest faces at each pixel.
            - pix_dists: Signed Euclidean distance in the x/y plane of each point closest to the pixel.
    """

    # Create a view of the meshes in clip space
    view = trimesh.view.View(mesh.vertices, mesh.faces)
    view.camera_position = np.array([0, 0, 5])
    view.camera_target = np.array([0, 0, 0])
    view.camera_up = np.array([0, 1, 0])

    # Rasterize the meshes
    rasterized = view.rasterize(image_size=image_size, blur_radius=blur_radius, faces_per_pixel=faces_per_pixel, bin_size=bin_size, max_faces_per_bin=max_faces_per_bin, perspective_correct=perspective_correct, clip_barycentric_coords=clip_barycentric_coords, cull_backfaces=cull_backfaces, z_clip_value=z_clip_value, cull_to_frustum=cull_to_frustum)

    # Extract the rasterized data
    pix_to_face = rasterized.pix_to_face
    zbuf = rasterized.zbuf
    barycentric = rasterized.barycentric
    pix_dists = rasterized.pix_dists

    return pix_to_face, zbuf, barycentric, pix_dists


if __name__ == "__main__":
    # Create a sample mesh
    mesh = trimesh.Trimesh(vertices=np.array([[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0]]), faces=np.array([[0, 1, 2], [0, 2, 3]]))

    # Rasterize the mesh
    pix_to_face, zbuf, barycentric, pix_dists = rasterize_meshes(meshes=[mesh], image_size=(256, 256))

    # Print the results
    print("pix_to_face:", pix_to_face)
    print("zbuf:", zbuf)
    print("barycentric:", barycentric)
    print("pix_dists:", pix_dists)