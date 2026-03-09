import torch
import numpy as np

def flow_to_image(flow):
    if flow.dtype!= torch.float:
        raise ValueError("Input flow tensor must be of type torch.float")
    
    if flow.ndim == 3:
        if flow.shape[0]!= 2:
            raise ValueError("Input flow tensor must have shape (2, H, W)")
        H, W = flow.shape[1:]
    elif flow.ndim == 4:
        if flow.shape[1]!= 2:
            raise ValueError("Input flow tensor must have shape (N, 2, H, W)")
        N, _, H, W = flow.shape
    else:
        raise ValueError("Input flow tensor must have shape (2, H, W) or (N, 2, H, W)")
    
    flow_norm = torch.sqrt(flow[0, :, :]**2 + flow[1, :, :]**2)
    
    rgb = torch.zeros(flow.shape[:-1] + (3,))
    
    rgb[0, :, :] = flow[0, :, :] / flow_norm
    rgb[1, :, :] = flow[1, :, :] / flow_norm
    rgb[2, :, :] = torch.sqrt(1 - flow[0, :, :]**2 / flow_norm**2)
    
    if flow.ndim == 4:
        rgb = rgb.unsqueeze(0)
    
    return rgb

if __name__ == "__main__":
    # Create sample input values
    flow = torch.rand(2, 64, 64)
    flow = torch.cat([flow, torch.rand(2, 64, 64)], dim=0)
    
    # Call the function and print the results
    rgb = flow_to_image(flow)
    print(rgb.shape)
    print(rgb.dtype)
    
    flow = torch.rand(3, 2, 64, 64)
    rgb = flow_to_image(flow)
    print(rgb.shape)
    print(rgb.dtype)