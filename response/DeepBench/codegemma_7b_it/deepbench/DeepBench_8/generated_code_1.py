import torch
import meshpy.triangle as triangle
import numpy as np
from pyquaternion import Quaternion
from typing import Optional

def cubify(
    voxels: torch.Tensor,
    thresh: float,
    feats: Optional[torch.Tensor] = None,
    device: str = "cpu",
    align: str = "center",
) -> meshpy.triangle.Mesh:
    """Creates a mesh from a voxel grid.

    Args:
        voxels: A torch.Tensor of shape (N, D, H, W) containing occupancy probabilities.
        thresh: A scalar threshold.
        feats: An optional torch.Tensor of shape (N, K, D, H, W) containing color information.
        device: The device of the output meshes.
        align: A string defining the alignment of the mesh vertices and grid locations.

    Returns:
        A Meshes object of the corresponding meshes.
    """

    # Convert voxel grid to a numpy array.
    voxels = voxels.detach().cpu().numpy()

    # Determine the size of the voxel grid.
    size = voxels.shape[1:]

    # Create a mesh.
    mesh = triangle.Mesh(size, dtype=np.float32)

    # Iterate over the voxels.
    for i in range(size[0]):
        for j in range(size[1]):
            for k in range(size[2]):
                # Check if the voxel is occupied.
                if voxels[0, i, j, k] > thresh:
                    # Create a cube.
                    cube = np.zeros((6, 4, 3))

                    # Set the vertices of the cube.
                    cube[0, 0] = np.array([i, j, k])
                    cube[0, 1] = np.array([i + 1, j, k])
                    cube[0, 2] = np.array([i + 1, j + 1, k])
                    cube[0, 3] = np.array([i, j + 1, k])
                    cube[1, 0] = np.array([i, j, k + 1])
                    cube[1, 1] = np.array([i + 1, j, k + 1])
                    cube[1, 2] = np.array([i + 1, j + 1, k + 1])
                    cube[1, 3] = np.array([i, j + 1, k + 1])
                    cube[2, 0] = np.array([i, j, k])
                    cube[2, 1] = np.array([i + 1, j, k])
                    cube[2, 2] = np.array([i + 1, j + 1, k])
                    cube[2, 3] = np.array([i, j + 1, k])
                    cube[3, 0] = np.array([i, j, k + 1])
                    cube[3, 1] = np.array([i + 1, j, k + 1])
                    cube[3, 2] = np.array([i + 1, j + 1, k + 1])
                    cube[3, 3] = np.array([i, j + 1, k + 1])
                    cube[4, 0] = np.array([i, j + 1, k])
                    cube[4, 1] = np.array([i + 1, j + 1, k])
                    cube[4, 2] = np.array([i + 1, j + 1, k + 1])
                    cube[4, 3] = np.array([i, j + 1, k + 1])
                    cube[5, 0] = np.array([i, j, k])
                    cube[5, 1] = np.array([i, j, k + 1])
                    cube[5, 2] = np.array([i + 1, j, k + 1])
                    cube[5, 3] = np.array([i + 1, j, k])

                    # Add the cube to the mesh.
                    mesh.add_points(cube.reshape(-1, 3))

    # Return the mesh.
    return mesh

if __name__ == "__main__":
    # Create sample input values.
    voxels = torch.rand(1, 32, 32, 32)
    voxels[0, 16, 16, 16] = 1.0
    voxels[0, 17, 16, 16] = 1.0
    voxels[0, 16, 17, 16] = 1.0
    voxels[0, 17, 17, 16] = 1.0
    voxels[0, 16, 16, 17] = 1.0
    voxels[0, 17, 16, 17] = 1.0
    voxels[0, 16, 17, 17] = 1.0
    voxels[0, 17, 17, 17] = 1.0

    # Call the cubify function.
    mesh = cubify(voxels, thresh=0.5)

    # Print the results.
    print(mesh)