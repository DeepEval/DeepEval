import torch
import torch.nn.functional as F

def flow_to_image(flow):
    """
    Converts an optical flow tensor into an RGB image tensor.
    
    Args:
    flow (torch.Tensor): Input tensor with shape (2, H, W) or (N, 2, H, W).
    
    Returns:
    torch.Tensor: Output tensor with shape (3, H, W) or (N, 3, H, W) if flow tensor is N-dimensional.
    """
    # Check if the input tensor has the correct shape and type
    if flow.ndim != 4 or flow.shape[1] != 2 or flow.dtype != torch.float:
        raise ValueError("Expected input tensor with shape (2, H, W) or (N, 2, H, W) and dtype torch.float.")
    
    # Normalize the flow values
    norm_flow = F.normalize(flow, p=2, dim=0)
    
    # Convert the normalized flow into an RGB image
    image = norm_flow @ torch.tensor([[0.5, 0, 0.5], [0, 0.5, 0], [0.5, 0, 0.5]])
    
    # Resize the image tensor to have 3 channels
    image = F.interpolate(image.unsqueeze(0), size=(flow.shape[-2], flow.shape[-1]), mode='bilinear', align_corners=False).squeeze(0)
    
    return image

if __name__ == "__main__":
    # Create a sample 2D flow tensor
    H, W = 10, 10
    flow = torch.randn(2, 2, H, W)
    
    # Convert flow to image
    rgb_image = flow_to_image(flow)
    
    # Print the resulting image tensor
    print(rgb_image.shape)  # Should print (3, H, W)