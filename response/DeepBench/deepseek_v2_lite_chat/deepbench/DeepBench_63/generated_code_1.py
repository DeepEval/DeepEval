import numpy as np
import cv2

def project_points_orthographic(points_in_camera):
    # Convert to homogeneous coordinates
    points_in_canonical = cv2.perspectiveTransform(points_in_camera, cv2.getPerspectiveTransform(np.float32([[0,0,0], [0,0,1]])))
    # Project to canonical plane z=1
    points_projected = points_in_canonical[:, :2] / points_in_canonical[:, 2, None]
    return points_projected

if __name__ == "__main__":
    # Create sample input values
    points_in_camera = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    
    # Call the function
    projected_points = project_points_orthographic(points_in_camera)
    
    # Print the results
    print("Projected points:")
    print(projected_points)