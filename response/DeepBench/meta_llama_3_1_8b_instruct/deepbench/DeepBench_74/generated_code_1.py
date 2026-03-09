import torch
import torch.nn.functional as F
import numpy as np

def canny(input, low_threshold, high_threshold, kernel_size, sigma, hysteresis, eps=1e-8):
    # Check the input tensor and its shape
    if len(input.shape)!= 4 or input.shape[1]!= 3:
        raise ValueError("Input tensor must be of shape (B, 3, H, W)")

    # Convert the input to grayscale
    input_gray = torch.mean(input, dim=1, keepdim=True)

    # Apply Gaussian blur
    blurred_gray = F.gaussian_filter(input_gray, [0, 0, sigma, sigma])

    # Compute gradients
    grad_x = F.pad(blurred_gray, (1, 1, 0, 0), mode='replicate')
    grad_x = grad_x[:, :, 1:, :] - grad_x[:, :, :-1, :]
    grad_y = F.pad(blurred_gray, (0, 0, 1, 1), mode='replicate')
    grad_y = grad_y[:, :, :, 1:] - grad_y[:, :, :, :-1]

    # Compute gradient magnitude and angle
    grad_mag = torch.sqrt(grad_x ** 2 + grad_y ** 2)
    grad_angle = torch.atan2(grad_y, grad_x)

    # Perform non-maximal suppression
    nms_mag, nms_angle = F.non_max_suppression(grad_mag, grad_angle, kernel_size=kernel_size, return_indices=False)

    # Apply thresholding
    thresholded_mag = torch.where(nms_mag > low_threshold, nms_mag, torch.zeros_like(nms_mag))
    thresholded_mag = torch.where(thresholded_mag > high_threshold, high_threshold, thresholded_mag)

    # Apply hysteresis if specified
    if hysteresis:
        hysteresis_mag = torch.where((nms_mag >= high_threshold) | (nms_mag >= low_threshold * nms_mag), nms_mag, torch.zeros_like(nms_mag))
    else:
        hysteresis_mag = thresholded_mag

    # Clip values to avoid NaN during backpropagation
    canny_mag = torch.clamp(hysteresis_mag, min=0, max=1)
    canny_edge = torch.where((hysteresis_mag > 0) | (nms_mag >= low_threshold), 1, 0)

    return canny_mag, canny_edge

if __name__ == "__main__":
    # Create sample input values
    input_tensor = torch.randn(1, 3, 256, 256)
    low_threshold = 0.1
    high_threshold = 0.2
    kernel_size = 3
    sigma = 1.0
    hysteresis = True

    # Call the function and print the results
    canny_mag, canny_edge = canny(input_tensor, low_threshold, high_threshold, kernel_size, sigma, hysteresis)
    print(canny_mag.shape, canny_edge.shape)
    print(canny_mag.min(), canny_mag.max())
    print(canny_edge.min(), canny_edge.max())