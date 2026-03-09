import torch
import torch.nn.functional as F

def canny(input_tensor, low_threshold, high_threshold, kernel_size, sigma, hysteresis, eps):
    # Check if input tensor is a valid PyTorch tensor
    if not isinstance(input_tensor, torch.Tensor):
        raise TypeError("Input tensor must be a PyTorch tensor.")
    
    # Check if input tensor has the correct shape
    if input_tensor.ndim != 4:
        raise ValueError("Input tensor must have 4 dimensions (batch, channels, height, width).")
    
    # Convert input tensor to grayscale if it has 3 channels
    if input_tensor.shape[1] == 3:
        input_tensor = torch.mean(input_tensor, dim=1, keepdim=True)
    
    # Convert to float32 for numerical stability
    input_tensor = input_tensor.float()
    
    # Apply Gaussian blur
    blurred_tensor = F.conv2d(input_tensor, torch.tensor([[1.0 / (2*3.1415*sigma**2)] * kernel_size**2 for _ in range(kernel_size)]))
    
    # Compute gradients
    grad_x = F.conv2d(input_tensor, torch.tensor([[-1.0, 0.0, 1.0]] * kernel_size**2))
    grad_y = F.conv2d(input_tensor, torch.tensor([[-1.0, -1.0, -1.0]] * kernel_size**2).transpose(0, 1))
    
    # Compute gradient magnitude and angle
    mag_tensor = torch.sqrt(grad_x**2 + grad_y**2)
    angle_tensor = torch.atan2(grad_y, grad_x)
    
    # Perform non-maximal suppression
    suppressed_tensor = mag_tensor.clone()
    suppressed_tensor[angle_tensor < (0.75 * torch.pi / 2) + eps] = 0
    suppressed_tensor[angle_tensor > (1.75 * torch.pi / 2) - eps] = 0
    
    # Apply thresholding
    edges_tensor = (suppressed_tensor > high_threshold) | (suppressed_tensor < low_threshold)
    
    # Apply hysteresis
    if hysteresis:
        for h in range(1, kernel_size - 1):
            for w in range(1, kernel_size - 1):
                # Connect edge if it is strong and at least one of its neighbors is also strong
                if edges_tensor[0, 0, h, w] and edges_tensor[0, 0, h - 1, w] and edges_tensor[0, 0, h + 1, w] and edges_tensor[0, 0, h, w - 1] and edges_tensor[0, 0, h, w + 1]:
                    edges_tensor[0, 0, h, w] = True
                # If the current pixel is strong and both its neighbors are weak, mark it as undetermined
                elif edges_tensor[0, 0, h, w] and not edges_tensor[0, 0, h - 1, w] and not edges_tensor[0, 0, h + 1, w] and not edges_tensor[0, 0, h, w - 1] and not edges_tensor[0, 0, h, w + 1]:
                    edges_tensor[0, 0, h, w] = False
    
    # Return canny edge magnitudes map and canny edge detection filtered by thresholds and hysteresis
    return torch.abs(input_tensor), edges_tensor

if __name__ == "__main__":
    # Sample input values
    batch_size = 1
    channels = 1
    height = 5
    width = 5
    low_threshold = 50
    high_threshold = 150
    kernel_size = 3
    sigma = 1
    hysteresis = True
    eps = 1e-6
    
    # Create sample input tensor
    input_tensor = torch.randn(batch_size, channels, height, width)
    
    # Call the function and print the results
    canny_output = canny(input_tensor, low_threshold, high_threshold, kernel_size, sigma, hysteresis, eps)
    print("Canny Edge Magnitudes:", canny_output[0])
    print("Canny Edge Detection (Filtered by Thresholds and Hysteresis):", canny_output[1].int())