import numpy as np
import cv2
import torch
import torch.nn.functional as F

def canny(input, low_threshold, high_threshold, kernel_size, sigma, hysteresis, eps):
    if input.dim() != 4 or input.size(1) not in [1, 3]:
        raise ValueError("Input tensor must have shape (B,C,H,W) and C must be 1 or 3.")

    B, C, H, W = input.shape
    if C == 3:
        input = input.permute(0, 2, 3, 1).cpu().numpy()  # (B, H, W, C)
        input = np.mean(input, axis=-1, keepdims=True)  # Convert to grayscale
        input = torch.from_numpy(input).permute(0, 3, 1, 2).to(input.device)  # (B, 1, H, W)

    input = input.squeeze(1).cpu().numpy()
    blurred = cv2.GaussianBlur(input, (kernel_size, kernel_size), sigma)

    gradient_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=kernel_size)
    gradient_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=kernel_size)
    gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2) + eps
    gradient_angle = np.arctan2(gradient_y, gradient_x) * (180.0 / np.pi) + 180.0

    non_max_suppressed = np.zeros_like(gradient_magnitude)
    for i in range(1, H-1):
        for j in range(1, W-1):
            angle = gradient_angle[i, j]
            if (angle >= 0 and angle < 45) or (angle >= 180 and angle < 225):
                q, r = gradient_magnitude[i, j+1], gradient_magnitude[i, j-1]
            elif (angle >= 45 and angle < 90) or (angle >= 225 and angle < 270):
                q, r = gradient_magnitude[i+1, j-1], gradient_magnitude[i-1, j+1]
            elif (angle >= 90 and angle < 135) or (angle >= 270 and angle < 315):
                q, r = gradient_magnitude[i+1, j], gradient_magnitude[i-1, j]
            else:
                q, r = gradient_magnitude[i, j+1], gradient_magnitude[i, j-1]

            if gradient_magnitude[i, j] >= q and gradient_magnitude[i, j] >= r:
                non_max_suppressed[i, j] = gradient_magnitude[i, j]
            else:
                non_max_suppressed[i, j] = 0

    strong_edges = (non_max_suppressed >= high_threshold)
    thresholded_edges = np.zeros_like(non_max_suppressed, dtype=np.uint8)
    weak_edges = (non_max_suppressed >= low_threshold) & (non_max_suppressed < high_threshold)

    if hysteresis:
        for i in range(1, H-1):
            for j in range(1, W-1):
                if strong_edges[i, j]:
                    thresholded_edges[i, j] = 255
                elif weak_edges[i, j]:
                    if np.any(strong_edges[i-1:i+2, j-1:j+2]):
                        thresholded_edges[i, j] = 255

    canny_edge_magnitudes = torch.tensor(gradient_magnitude).unsqueeze(1)
    canny_edges = torch.tensor(thresholded_edges).unsqueeze(1)

    return canny_edge_magnitudes, canny_edges

if __name__ == "__main__":
    input_image = torch.rand((1, 3, 100, 100))  # Example input tensor
    low_threshold = 50
    high_threshold = 150
    kernel_size = 5
    sigma = 1.0
    hysteresis = True
    eps = 1e-7

    magnitudes, edges = canny(input_image, low_threshold, high_threshold, kernel_size, sigma, hysteresis, eps)
    print("Canny Edge Magnitudes:", magnitudes.shape)
    print("Canny Edges:", edges.shape)