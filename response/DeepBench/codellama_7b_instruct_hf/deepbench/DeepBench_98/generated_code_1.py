import numpy as np

def run_8point(points1, points2, weights):
    # Compute the number of points in each image
    num_points1 = points1.shape[1]
    num_points2 = points2.shape[1]

    # Check that the number of points is at least 8 for both images
    if num_points1 < 8 or num_points2 < 8:
        raise ValueError("The number of points must be at least 8 for both images")

    # Convert points to homogeneous coordinates
    points1_hom = np.concatenate((points1, np.ones((points1.shape[0], 1))), axis=1)
    points2_hom = np.concatenate((points2, np.ones((points2.shape[0], 1))), axis=1)

    # Compute the 8 point correspondences
    correspondences = np.array([[points1_hom[i], points2_hom[i]] for i in range(num_points1)])

    # Compute the weighted least squares solution
    W = np.array([[weights[i]] for i in range(num_points1)])
    X = np.array([[correspondences[i]] for i in range(num_points1)])
    WX = np.dot(W, X)

    # Compute the fundamental matrix
    F = np.linalg.solve(WX, np.dot(W.T, WX))

    return F

if __name__ == "__main__":
    # Example usage
    # Generate some sample input values
    points1 = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14], [15, 16]])
    points2 = np.array([[2, 4], [4, 6], [6, 8], [8, 10], [10, 12], [12, 14], [14, 16], [16, 18]])
    weights = np.array([1, 1, 1, 1, 1, 1, 1, 1])

    # Call the function and print the results
    F = run_8point(points1, points2, weights)
    print(F)