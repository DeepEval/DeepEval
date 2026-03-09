import numpy as np

def get_perspective_transform(points_src, points_dst):
    A = np.zeros((8, 9))
    for i in range(points_src.shape[0]):
        A[i*2] = points_src[i, :2]
        A[i*2+1] = points_dst[i, :2]
    _, _, V = np.linalg.svd(A)
    T = V[-1, :]
    T[:2] /= T[2]
    T[2] = 1
    return T

if __name__ == "__main__":
    # Example usage:
    import cv2
    import matplotlib.pyplot as plt

    # Define four points:
    src = np.array([[200, 70], [750, 70], [750, 575], [200, 575]])
    dst = np.array([[100, 70], [300, 70], [300, 575], [100, 575]])

    # Call the function and print the result
    T = get_perspective_transform(src, dst)
    print("Perspective Transformation Matrix:")
    print(T)

    # Convert the points to homogeneous coordinates
    src_hom = np.hstack((src, np.ones((4, 1))))
    dst_hom = np.hstack((dst, np.ones((4, 1))))

    # Draw the source and destination points on an image
    image = np.zeros((700, 1000, 3), dtype="uint8")
    cv2.polylines(image, [np.int32(src_hom)], True, (0, 250, 150), 3)
    cv2.polylines(image, [np.int32(dst_hom)], True, (0, 250, 150), 3)
    cv2.imshow("Source and Destination Points", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()