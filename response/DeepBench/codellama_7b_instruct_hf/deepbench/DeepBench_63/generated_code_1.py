import numpy as np
import tensorflow as tf

def project_points_orthographic(points_in_camera):
    # Get the x, y, and z coordinates of the points
    x, y, z = tf.unstack(points_in_camera, axis=-1)

    # Compute the projected points
    u = x / z
    v = y / z

    # Stack the projected points back into a tensor
    projected_points = tf.stack([u, v, z], axis=-1)

    return projected_points

if __name__ == "__main__":
    # Create some sample input values
    points = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
    ])

    # Call the function and print the results
    projected_points = project_points_orthographic(points)
    print(projected_points)