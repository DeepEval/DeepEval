import torch
import torch.nn.functional as F

def get_sobel_kernel2d_2nd_order(device=None, dtype=None):
    # Get the 5x5 2nd-order Sobel kernel for the second derivative in the x-direction (gxx)
    gxx = get_sobel_kernel_5x5_2nd_order(device=device, dtype=dtype)

    # Transpose the gxx kernel to obtain the kernel for the second derivative in the y-direction (gyy)
    gyy = gxx.transpose(-2, -1)

    # Use get_sobel_kernel_5x5_2nd_order_xy to obtain the mixed partial derivative kernel (gxy)
    gxy = get_sobel_kernel_5x5_2nd_order_xy(device=device, dtype=dtype)

    # Stack the gxx, gxy, and gyy kernels into a single tensor
    kernel = torch.stack((gxx, gxy, gyy), dim=0)

    return kernel

if __name__ == "__main__":
    # Create sample input values
    input = torch.randn(1, 3, 5, 5)

    # Call the function and print the results
    kernel = get_sobel_kernel2d_2nd_order(device="cpu")
    print(kernel)