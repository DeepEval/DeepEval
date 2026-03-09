import torch
import torch.nn.functional as F

def flow_to_image(flow):
    """
    Converts an optical flow tensor into an RGB image tensor.

    Args:
        flow (torch.Tensor): Input flow tensor of shape (2, H, W) or (N, 2, H, W)

    Returns:
        torch.Tensor: Output image tensor of shape (3, H, W) or (N, 3, H, W)

    Raises:
        ValueError: If the flow tensor does not meet the expected shape or dtype
    """
    if not isinstance(flow, torch.Tensor):
        raise ValueError("Input flow must be of type torch.Tensor")

    if not (flow.ndimension() == 2 or flow.ndimension() == 4):
        raise ValueError("Input flow must have 2 or 4 dimensions")

    if not (flow.shape[0] == 2 or flow.shape[1] == 2):
        raise ValueError("Input flow must have 2 or 4 dimensions")

    if not flow.dtype == torch.float:
        raise ValueError("Input flow must be of type torch.float")

    # Normalize flow values
    flow_norm = flow / torch.norm(flow, dim=1, keepdim=True)

    # Convert normalized flow into RGB image
    img = torch.zeros(flow_norm.shape[0], 3, flow_norm.shape[2], flow_norm.shape[3])
    img[..., 0] = flow_norm[..., 0]
    img[..., 1] = flow_norm[..., 1]
    img[..., 2] = 1.0 - torch.norm(flow_norm, dim=1, keepdim=True)

    return img

if __name__ == "__main__":
    # Create sample input values
    flow = torch.randn(2, 2, 3, 3)

    # Call function and print results
    output = flow_to_image(flow)
    print(output.shape)  # Output image shape: (2, 3, 3)
    print(output.dtype)  # Output image dtype: torch.float