import numpy as np

def get_perspective_transform(points_src, points_dst):
    A = np.zeros((8, 9))
    for i in range(4):
        x, y = points_src[:, i]
        x_, y_ = points_dst[:, i]
        A[2*i] = [x, y, 1, 0, 0, 0, -x_*x, -x_*y, -x_*1]
        A[2*i+1] = [0, 0, 0, x, y, 1, -y_*x, -y_*y, -y_*1]
    U, S, Vh = np.linalg.svd(A)
    H = Vh.T @ np.diag([1, 1, 0]) @ U.T
    return H[:3, :3]

if __name__ == "__main__":
    # Create sample input values
    np.random.seed(0)
    B = 1
    points_src = np.random.rand(B, 4, 2)
    points_dst = np.random.rand(B, 4, 2)
    
    # Call the function
    H = get_perspective_transform(points_src, points_dst)
    
    # Print the results
    print(H)