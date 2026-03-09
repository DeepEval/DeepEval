import torch
import numpy as np

def flow_to_image(flow):
    if not isinstance(flow, torch.Tensor):
        raise TypeError("Input flow must be a torch.Tensor.")
    if flow.dtype != torch.float:
        raise ValueError("Input flow must be of type torch.float.")
    if flow.dim() == 3:
        N = 1
        _, H, W = flow.shape
    elif flow.dim() == 4:
        N, _, H, W = flow.shape
    else:
        raise ValueError("Input flow must have a shape of (2, H, W) or (N, 2, H, W).")
    
    if flow.size(-3) != 2:
        raise ValueError("The size of the channel dimension must be 2.")

    # Compute magnitude and angle
    u = flow[..., 0, :, :]
    v = flow[..., 1, :, :]
    magnitude = torch.sqrt(u**2 + v**2)
    angle = torch.atan2(v, u)

    # Normalize magnitude
    max_magnitude = torch.max(magnitude)
    if max_magnitude > 0:
        magnitude = magnitude / max_magnitude

    # Convert to HSV
    hsv = torch.zeros((N, 3, H, W), dtype=torch.float)
    hsv[..., 0, :, :] = (angle / (2 * np.pi)) % 1.0
    hsv[..., 1, :, :] = 1.0
    hsv[..., 2, :, :] = magnitude

    # Convert HSV to RGB
    rgb = hsv_to_rgb(hsv)
    
    if N == 1:
        return rgb[0]
    return rgb

def hsv_to_rgb(hsv):
    h, s, v = hsv[:, 0], hsv[:, 1], hsv[:, 2]
    h = h * 6
    c = v * s
    x = c * (1 - torch.abs(h % 2 - 1))
    m = v - c

    zeros = torch.zeros_like(h)
    cm = torch.stack([c, x, zeros], dim=0)
    xc = torch.stack([x, c, zeros], dim=0)

    rgb = torch.where((0 <= h) & (h < 1), cm, torch.where(
          (1 <= h) & (h < 2), xc, torch.where(
          (2 <= h) & (h < 3), torch.stack([zeros, c, x], dim=0), torch.where(
          (3 <= h) & (h < 4), torch.stack([zeros, x, c], dim=0), torch.where(
          (4 <= h) & (h < 5), torch.stack([x, zeros, c], dim=0),
          torch.stack([c, zeros, x], dim=0)
    )))))

    rgb = rgb + m
    return rgb.permute(1, 0, 2, 3)

if __name__ == "__main__":
    # Example usage:
    N, H, W = 1, 3, 3  # Single flow example
    flow = torch.tensor([[[1.0, 0.5, -1.0],
                          [0.0, 0.0, 0.0],
                          [-0.5, -0.5, 1.0]],
                         [[0.5, 1.0, 0.5],
                          [0.0, 0.0, 0.0],
                          [-1.0, 0.5, -0.5]]], dtype=torch.float)

    # Call the function and print the result
    img = flow_to_image(flow)
    print("RGB Image tensor:", img)