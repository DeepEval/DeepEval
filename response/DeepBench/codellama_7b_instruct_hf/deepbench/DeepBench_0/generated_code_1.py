import numpy as np

def rasterize_meshes(meshes, image_size=(256, 256), blur_radius=0.0, faces_per_pixel=8, bin_size=None, max_faces_per_bin=None, perspective_correct=False, clip_barycentric_coords=False, cull_backfaces=False, z_clip_value=None, cull_to_frustum=False):
    # Initialize output arrays
    pix_to_face = np.zeros((image_size[0], image_size[1], faces_per_pixel), dtype=np.int32)
    zbuf = np.zeros((image_size[0], image_size[1], faces_per_pixel), dtype=np.float32)
    barycentric = np.zeros((image_size[0], image_size[1], faces_per_pixel, 3), dtype=np.float32)
    pix_dists = np.zeros((image_size[0], image_size[1], faces_per_pixel), dtype=np.float32)

    # Loop over meshes and faces
    for mesh in meshes:
        # Get the vertices and faces of the current mesh
        vertices = mesh.vertices
        faces = mesh.faces

        # Loop over faces and rasterize them
        for face in faces:
            # Get the vertices of the current face
            v1 = vertices[face[0]]
            v2 = vertices[face[1]]
            v3 = vertices[face[2]]

            # Calculate the bounding box of the face
            bbox = [min(v1[0], v2[0], v3[0]), min(v1[1], v2[1], v3[1]), max(v1[0], v2[0], v3[0]), max(v1[1], v2[1], v3[1])]

            # Calculate the barycentric coordinates of the face
            barycentric_coords = [v1[0] - bbox[0], v1[1] - bbox[1], v2[0] - bbox[0], v2[1] - bbox[1], v3[0] - bbox[0], v3[1] - bbox[1]]

            # Calculate the signed Euclidean distance of the face from the center of the image
            dist = np.sqrt((bbox[0] - image_size[0] / 2) ** 2 + (bbox[1] - image_size[1] / 2) ** 2)

            # Clip the barycentric coordinates if necessary
            if clip_barycentric_coords:
                barycentric_coords = np.clip(barycentric_coords, -1.0, 1.0)

            # Cull the face if it is outside the view frustum
            if cull_to_frustum:
                # Check if the face is inside the view frustum
                if bbox[2] > bbox[0] and bbox[3] > bbox[1]:
                    continue

            # Calculate the z-coordinate of the face
            z = (v1[2] + v2[2] + v3[2]) / 3

            # Clip the z-coordinate if necessary
            if z_clip_value is not None:
                z = np.clip(z, z_clip_value, None)

            # Calculate the barycentric coordinates and signed Euclidean distance of the face
            barycentric_coords = np.array([barycentric_coords])
            pix_dists = np.array([dist])

            # Calculate the 2D coordinates of the face in the image
            uv = np.array([[bbox[0], bbox[1]], [bbox[2], bbox[1]], [bbox[2], bbox[3]], [bbox[0], bbox[3]]])

            # Rasterize the face
            for pixel in uv:
                # Calculate the barycentric coordinates and signed Euclidean distance of the pixel
                barycentric_coord = np.array([[pixel[0] - bbox[0], pixel[1] - bbox[1]]])
                pix_dist = np.array([[np.sqrt((pixel[0] - image_size[0] / 2) ** 2 + (pixel[1] - image_size[1] / 2) ** 2)]])

                # Clip the barycentric coordinates if necessary
                if clip_barycentric_coords:
                    barycentric_coord = np.clip(barycentric_coord, -1.0, 1.0)

                # Cull the pixel if it is outside the view frustum
                if cull_to_frustum:
                    # Check if the pixel is inside the view frustum
                    if bbox[0] < image_size[0] and bbox[1] < image_size[1] and bbox[2] > 0 and bbox[3] > 0:
                        continue

                # Calculate the z-coordinate of the pixel
                z_pixel = (z * barycentric_coord[0][0] + v1[2] * barycentric_coord[0][1] + v2[2] * barycentric_coord[0][2] + v3[2] * barycentric_coord[0][3]) / (barycentric_coord[0][0] + barycentric_coord[0][1] + barycentric_coord[0][2] + barycentric_coord[0][3])

                # Clip the z-coordinate if necessary
                if z_clip_value is not None:
                    z_pixel = np.clip(z_pixel, z_clip_value, None)

                # Calculate the 2D coordinates of the pixel in the image
                pixel_coord = np.array([[pixel[0], pixel[1]]])

                # Calculate the indices of the nearest faces at the pixel
                nearest_face_idx = np.argmin(pix_dist)

                # Calculate the barycentric coordinates of the nearest face at the pixel
                barycentric_coord = barycentric_coords[nearest_face_idx]

                # Calculate the signed Euclidean distance of the nearest face at the pixel
                pix_dist = pix_dists[nearest_face_idx]

                # Calculate the z-buffer value of the nearest face at the pixel
                z_buffer = zbuf[pixel_coord[0][0], pixel_coord[0][1]]

                # Cull the pixel if the z-buffer value is greater than the z-coordinate of the pixel
                if z_buffer > z_pixel:
                    continue

                # Update the z-buffer value
                zbuf[pixel_coord[0][0], pixel_coord[0][1]] = z_pixel

                # Update the barycentric coordinates, signed Euclidean distance, and indices of the nearest faces at the pixel
                barycentric[pixel_coord[0][0], pixel_coord[0][1]] = barycentric_coord
                pix_dists[pixel_coord[0][0], pixel_coord[0][1]] = pix_dist
                pix_to_face[pixel_coord[0][0], pixel_coord[0][1]] = nearest_face_idx

    return pix_to_face, zbuf, barycentric, pix_dists

if __name__ == "__main__":
    class MockMesh:
        def __init__(self):
            self.vertices = np.array([[50, 50, 10], [150, 50, 10], [100, 150, 10]], dtype=np.float32)
            self.faces = np.array([[0, 1, 2]], dtype=np.int32)
    meshes = [MockMesh()]
    pix_to_face, zbuf, barycentric, pix_dists = rasterize_meshes(meshes)
    print("pix_to_face shape:", pix_to_face.shape)
    print("zbuf shape:", zbuf.shape)
    print("barycentric shape:", barycentric.shape)
    print("pix_dists shape:", pix_dists.shape)