import numpy as np
import tensorflow as tf

def unproject_points_orthographic(points_in_camera, extension):
    # In this implementation, we assume the camera intrinsic parameters
    # are provided in the form of focal length and principal point.
    # The camera extrinsic parameters are provided as rotation matrix,
    # translation vector, and a flag indicating orthographic projection.
    # The flag is True for orthographic projection, False for perspective.
    
    # Define the intrinsic and extrinsic parameters
    focal_length = 1
    principal_point = np.array([0.5, 0.5])
    rotation_matrix = np.eye(3)
    translation_vector = np.array([0, 0, -1])
    orthographic_flag = True
    
    # Create a TensorFlow tensor for the points in the camera frame
    points_in_camera_tf = tf.constant(points_in_camera, dtype=tf.float32)
    
    # Create a TensorFlow tensor for the camera extension
    extension_tf = tf.constant(extension, dtype=tf.float32)
    
    # Convert the TensorFlow tensors to numpy arrays
    points_in_camera_np = points_in_camera_tf.numpy()
    extension_np = extension_tf.numpy()
    
    # Calculate the unprojection matrix
    if orthographic_flag:
        unprojection_matrix = np.array([[-1 / focal_length, 0, principal_point[0], 0],
                                        [0, -1 / focal_length, principal_point[1], 0],
                                        [0, 0, 1, 0],
                                        [0, 0, 0, 1]])
    else:
        # In perspective projection, we need to calculate the
        # perspective division and then unproject the points
        points_in_camera_normalized = points_in_camera_np / extension_np
        points_in_camera_perspective_divided = points_in_camera_normalized - np.tile(principal_point, (points_in_camera_np.shape[0], 1))
        points_in_camera_perspective_divided_homogeneous = np.concatenate((points_in_camera_perspective_divided, np.ones((points_in_camera_np.shape[0], 1))), axis=1)
        unprojection_matrix = np.concatenate((rotation_matrix, translation_vector.reshape(1, 3)), axis=0)
        unprojection_matrix = np.concatenate((unprojection_matrix, np.array([[0, 0, 0, 1]]).reshape(1, 4)), axis=0)
        unprojection_matrix = unprojection_matrix / np.array([[extension_np[0], 0, 0, 0],
                                                             [0, extension_np[1], 0, 0],
                                                             [0, 0, -1, 0],
                                                             [0, 0, -1, 0]])
    
    # Unproject the points
    unprojected_points = np.dot(unprojection_matrix, points_in_camera_perspective_divided_homogeneous)
    
    # Convert the numpy arrays to TensorFlow tensors
    unprojected_points_tf = tf.constant(unprojected_points, dtype=tf.float32)
    
    # Return the unprojected points
    return unprojected_points_tf

if __name__ == "__main__":
    # Create sample input values
    points_in_camera = np.array([[0, 0], [1, 1], [2, 2]], dtype=np.float32)
    extension = np.array([10, 10], dtype=np.float32)
    
    # Call the function and print the results
    unprojected_points = unproject_points_orthographic(points_in_camera, extension)
    print(unprojected_points.numpy())