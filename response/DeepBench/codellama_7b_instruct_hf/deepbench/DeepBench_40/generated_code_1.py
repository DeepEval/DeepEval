import numpy as np

def get_sobel_kernel2d(device=None, dtype=None):
    if device is None:
        device = "cpu"
    if dtype is None:
        dtype = np.float32

    sobel_kernel_x = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    sobel_kernel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    sobel_kernel_xy = np.stack([sobel_kernel_x, sobel_kernel_y], axis=0)

    return sobel_kernel_xy

if __name__ == "__main__":
    # Test the function
    sobel_kernel = get_sobel_kernel2d(device="cpu", dtype=np.float32)
    print(sobel_kernel)