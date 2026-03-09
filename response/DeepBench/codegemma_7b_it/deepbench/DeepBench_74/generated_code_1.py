import torch
import torch.nn as nn

def canny(input, low_threshold, high_threshold, kernel_size=5, sigma=1.4, hysteresis=True, eps=1e-6):
    if input.shape[1] == 3:
        # Convert to grayscale
        input = torch.mean(input, dim=1, keepdim=True)

    # Apply Gaussian blur
    gaussian_kernel = torch.nn.functional.gaussian_kernel2d(kernel_size, sigma)
    input = torch.nn.functional.conv2d(input, gaussian_kernel[None, None, :, :].repeat(input.shape[0], 1, 1, 1), padding="same")

    # Compute gradients
    grad_x = torch.nn.functional.conv2d(input, torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]).float().to(input.device)[None, None, :, :].repeat(input.shape[0], 1, 1, 1), padding="same")
    grad_y = torch.nn.functional.conv2d(input, torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]).float().to(input.device)[None, None, :, :].repeat(input.shape[0], 1, 1, 1), padding="same")

    # Compute gradient magnitude and angle
    grad_magnitude = torch.sqrt(grad_x ** 2 + grad_y ** 2)
    grad_angle = torch.atan2(grad_y, grad_x)

    # Non-maximal suppression
    nms_mask = torch.zeros(grad_magnitude.shape, dtype=torch.bool).to(input.device)
    nms_mask[0:-2, ...] = (grad_magnitude[0:-2, ...] > grad_magnitude[1:-1, ...]) & (grad_magnitude[0:-2, ...] > grad_magnitude[2:, ...])
    nms_mask[2:, ...] = (grad_magnitude[2:, ...] > grad_magnitude[1:-1, ...]) & (grad_magnitude[2:, ...] > grad_magnitude[0:-2, ...])
    nms_mask[1:-1, ...] = (grad_magnitude[1:-1, ...] > grad_magnitude[0:-2, ...]) & (grad_magnitude[1:-1, ...] > grad_magnitude[2:, ...])

    # Thresholding
    threshold_mask = (grad_magnitude >= high_threshold) | (grad_magnitude <= low_threshold)

    # Hysteresis edge tracking
    if hysteresis:
        hysteresis_mask = torch.zeros(grad_magnitude.shape, dtype=torch.bool).to(input.device)
        hysteresis_mask[1:-1, ...] = (threshold_mask[0:-2, ...] & threshold_mask[2:, ...]) | (threshold_mask[1:-1, ...] & (grad_magnitude[1:-1, ...] > low_threshold))
    else:
        hysteresis_mask = threshold_mask

    # Combine results
    canny_edges = grad_magnitude * hysteresis_mask.float()
    canny_edges = torch.max(canny_edges, dim=1, keepdim=True)[0]

    return canny_edges, canny_edges * threshold_mask.float()

if __name__ == "__main__":
    # Sample input values
    input = torch.randn(1, 3, 256, 256)
    low_threshold = 50
    high_threshold = 150

    # Call the function
    canny_edges, canny_edges_thresholded = canny(input, low_threshold, high_threshold)

    # Print the results
    print("Canny Edges:", canny_edges.shape)
    print("Canny Edges (Thresholded):", canny_edges_thresholded.shape)