import numpy as np

def unproject_points_orthographic(points_in_camera, extension):
    """
    Unprojects points from the canonical z=1 plane into the camera frame.

    Parameters
    ----------
    points_in_camera : numpy array
        Tensor representing the points to unproject with shape (..., 2)
    extension : numpy array
        Tensor representing the extension of the points to unproject with shape (..., 1)

    Returns
    -------
    points_in_camera : numpy array
        Tensor representing the unprojected points with shape (..., 3)
    """
    x = points_in_camera[:, 0]
    y = points_in_camera[:, 1]
    z = np.ones_like(x)
    x_prime = x / z
    y_prime = y / z
    z_prime = z / z
    points_in_camera = np.stack((x_prime, y_prime, z_prime), axis=1)
    return points_in_camera

if __name__ == "__main__":
    # Create sample input values
    points_in_camera = np.array([[0, 0, 1], [1, 0, 1], [0, 1, 1]])
    extension = np.array([[1], [1], [1]])

    # Call the function and print the results
    unprojected_points = unproject_points_orthographic(points_in_camera, extension)
    print(unprojected_points)