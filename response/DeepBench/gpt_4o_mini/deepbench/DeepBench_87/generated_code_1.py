import torch
import numpy as np

def flow_to_image(flow):
    if not isinstance(flow, torch.Tensor) or flow.dtype != torch.float:
        raise ValueError("Flow must be a torch.float tensor.")
    
    if flow.ndim == 3:
        if flow.shape[0] != 2:
            raise ValueError("Flow tensor must have shape (2, H, W) or (N, 2, H, W).")
        N = 1
        H, W = flow.shape[1], flow.shape[2]
    elif flow.ndim == 4:
        if flow.shape[1] != 2:
            raise ValueError("Flow tensor must have shape (2, H, W) or (N, 2, H, W).")
        N = flow.shape[0]
        H, W = flow.shape[2], flow.shape[3]
    else:
        raise ValueError("Flow tensor must have shape (2, H, W) or (N, 2, H, W).")

    flow = flow.permute(0, 2, 3, 1)  # Change to (N, H, W, 2) or (1, H, W, 2)
    max_flow = flow.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0]  # Maximum flow magnitude
    min_flow = flow.min(dim=-1, keepdim=True)[0].min(dim=-2, keepdim=True)[0]  # Minimum flow magnitude
    
    normalized_flow = (flow - min_flow) / (max_flow - min_flow + 1e-8)  # Normalize the flow

    flow_rgb = torch.zeros((N, H, W, 3), dtype=torch.float)
    flow_rgb[..., 0] = normalized_flow[..., 0]  # Red channel
    flow_rgb[..., 1] = normalized_flow[..., 1]  # Green channel
    flow_rgb[..., 2] = torch.sqrt(normalized_flow[..., 0]**2 + normalized_flow[..., 1]**2)  # Blue channel

    flow_rgb = flow_rgb.permute(0, 3, 1, 2)  # Change back to (N, 3, H, W)
    
    return flow_rgb

if __name__ == "__main__":
    flow_tensor = torch.randn((2, 100, 100), dtype=torch.float)  # Example flow tensor
    flow_image = flow_to_image(flow_tensor)
    print(flow_image.shape)  # Expected output: (3, 100, 100)
    print(flow_image)