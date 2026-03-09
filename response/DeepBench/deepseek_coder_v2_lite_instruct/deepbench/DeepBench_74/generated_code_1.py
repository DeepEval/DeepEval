import torch
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import gaussian_filter

def canny(input, low_threshold, high_threshold, kernel_size=3, sigma=1.0, hysteresis=True, eps=1e-7):
    # Check input tensor and its shape
    if input.ndim != 4 or input.shape[1] not in [1, 3]:
        raise ValueError("Input tensor must have shape (B, C, H, W) where C is either 1 or 3.")

    # Convert to grayscale if input has 3 channels
    if input.shape[1] == 3:
        input = 0.299 * input[:, 0] + 0.587 * input[:, 1] + 0.114 * input[:, 2]

    # Apply Gaussian blur
    input = F.pad(input, (kernel_size // 2, kernel_size // 2, kernel_size // 2, kernel_size // 2), mode='reflect')
    input = gaussian_filter(input.numpy(), sigma=sigma)
    input = torch.tensor(input).to(input.device)

    # Compute gradients
    Gx = F.pad(input, (1, 1, 0, 0), mode='reflect')[:, :, :, :-1] - F.pad(input, (1, 1, 1, 0), mode='reflect')[:, :, :-1, :-1]
    Gy = F.pad(input, (0, 0, 1, 1), mode='reflect')[:, :, :-1, :] - F.pad(input, (1, 0, 1, 1), mode='reflect')[:, :, :-1, :-1]

    # Compute gradient magnitude and angle
    G = torch.sqrt(Gx**2 + Gy**2 + eps)
    theta = torch.atan2(Gy, Gx)

    # Perform non-maximal suppression
    angle = theta * (180 / np.pi)
    angle[angle < 0] += 180

    suppressed = torch.zeros_like(G)
    for i in range(1, G.shape[2] - 1):
        for j in range(1, G.shape[3] - 1):
            if (0 <= angle[0, 0, i, j] < 22.5) or (157.5 <= angle[0, 0, i, j] <= 180):
                ddf = G[0, 0, i, j+1]
                ddb = G[0, 0, i, j-1]
            elif 22.5 <= angle[0, 0, i, j] < 67.5:
                ddf = G[0, 0, i+1, j+1]
                ddb = G[0, 0, i-1, j-1]
            elif 67.5 <= angle[0, 0, i, j] < 112.5:
                ddf = G[0, 0, i+1, j]
                ddb = G[0, 0, i-1, j]
            else:
                ddf = G[0, 0, i-1, j+1]
                ddb = G[0, 0, i+1, j-1]

            if G[0, 0, i, j] >= ddf and G[0, 0, i, j] >= ddb:
                suppressed[0, 0, i, j] = G[0, 0, i, j]

    # Apply thresholding and hysteresis
    weak = 50
    strong = 255
    strong_i, strong_j = (suppressed >= high_threshold).nonzero(), (suppressed >= high_threshold).nonzero()
    weak_i, weak_j = (suppressed >= low_threshold) & (suppressed < high_threshold).nonzero(), (suppressed >= low_threshold) & (suppressed < high_threshold).nonzero()

    edges = torch.zeros_like(suppressed)
    if hysteresis:
        for i, j in zip(strong_i, strong_j):
            edges[i[0], i[1], j[0], j[1]] = strong
        for i, j in zip(weak_i, weak_j):
            if (edges[i[0], i[1], max(0, i[2]-1):min(edges.shape[2], i[2]+2), max(0, i[3]-1):min(edges.shape[3], i[3]+2)] == strong).any():
                edges[i[0], i[1], i[2], i[3]] = weak

    return G, edges

if __name__ == "__main__":
    # Create sample input values
    input_tensor = torch.randn(1, 3, 256, 256)
    low_threshold = 50
    high_threshold = 150
    kernel_size = 3
    sigma = 1.0
    hysteresis = True

    # Call the function and print the results
    magnitudes, edges = canny(input_tensor, low_threshold, high_threshold, kernel_size, sigma, hysteresis)
    print("Magnitude shape:", magnitudes.shape)
    print("Edges shape:", edges.shape)