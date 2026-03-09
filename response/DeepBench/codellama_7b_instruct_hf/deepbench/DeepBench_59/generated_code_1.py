import numpy as np

def get_perspective_transform(points_src, points_dst):
    # Calculate the matrix of the perspective transform
    h = np.zeros((3, 3))
    for i in range(4):
        h[0, i] = points_src[i, 0]
        h[1, i] = points_src[i, 1]
        h[2, i] = 1
    for i in range(4):
        h[3, i] = points_dst[i, 0]
        h[4, i] = points_dst[i, 1]
        h[5, i] = 1
    return h

# Example usage
if __name__ == "__main__":
    # Create sample input values
    points_src = np.array([[0, 0], [100, 0], [0, 100], [100, 100]])
    points_dst = np.array([[0, 0], [100, 0], [0, 100], [100, 100]])

    # Call the function
    h = get_perspective_transform(points_src, points_dst)

    # Print the results
    print(h)