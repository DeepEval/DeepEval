import torch
import numpy as np

def get_sobel_kernel2d(device=None, dtype=torch.float32):
    sobel_x_kernel = np.array([[-1, 0, 1],
                               [-2, 0, 2],
                               [-1, 0, 1]])
    sobel_y_kernel = np.array([[-1, -2, -1],
                               [ 0,  0,  0],
                               [ 1,  2,  1]])

    sobel_x_kernel_tensor = torch.tensor(sobel_x_kernel, device=device, dtype=dtype)
    sobel_y_kernel_tensor = torch.tensor(sobel_y_kernel, device=device, dtype=dtype)

    return torch.stack((sobel_x_kernel_tensor, sobel_y_kernel_tensor))

if __name__ == "__main__":
    # Create sample input values
    input_values = np.random.rand(3, 3)

    # Convert numpy array to torch tensor
    input_tensor = torch.tensor(input_values)

    # Get Sobel kernel
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    kernel = get_sobel_kernel2d(device=device)

    print("Sobel kernel (x direction):")
    print(kernel[0])
    print("\nSobel kernel (y direction):")
    print(kernel[1])