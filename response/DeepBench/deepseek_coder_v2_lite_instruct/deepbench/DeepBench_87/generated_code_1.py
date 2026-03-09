import torch
import numpy as np
from PIL import Image
import matplotlib.cm as cm

def flow_to_image(flow):
    if not isinstance(flow, torch.Tensor) or flow.dtype != torch.float:
        raise ValueError("Input must be a torch.float tensor")
    if flow.ndim not in [3, 4] or flow.shape[1] != 2:
        raise ValueError("Input must have shape (2, H, W) or (N, 2, H, W)")

    # Normalize the flow values to the range [0, 1]
    flow_min = flow.view(flow.shape[0], 2, -1).min(2)[0].view(flow.shape[0], 2, 1, 1)
    flow_max = flow.view(flow.shape[0], 2, -1).max(2)[0].view(flow.shape[0], 2, 1, 1)
    normalized_flow = (flow - flow_min) / (flow_max - flow_min)

    # Convert the normalized flow to an RGB image
    rgb_flow = torch.cat([normalized_flow[..., 0:1], torch.zeros_like(normalized_flow[..., :1]), -normalized_flow[..., 1:2]], dim=-1)
    rgb_flow = cm.jet(rgb_flow.numpy())[..., :3]
    rgb_flow = torch.from_numpy(rgb_flow).permute(0, 3, 1, 2)

    return rgb_flow

if __name__ == "__main__":
    # Create a sample flow tensor
    flow = torch.randn(2, 2, 100, 100) * 10  # N, 2, H, W
    flow_image = flow_to_image(flow)
    print(flow_image.shape)  # Should be (2, 3, 100, 100)
    flow_image.clamp_(0, 1)  # Ensure pixel values are in [0, 1]
    flow_image = (flow_image * 255).byte()  # Convert to uint8
    flow_image = flow_image.permute(0, 2, 3, 1).cpu().numpy()  # Convert to numpy and move to CPU
    for i, img in enumerate(flow_image):
        Image.fromarray(img).save(f"flow_image_{i}.png")