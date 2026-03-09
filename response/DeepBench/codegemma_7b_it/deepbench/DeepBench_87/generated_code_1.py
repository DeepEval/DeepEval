import torch
import torch.nn as nn

class Normalize(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.mean = torch.tensor(mean).float().view(-1, 1, 1)
        self.std = torch.tensor(std).float().view(-1, 1, 1)

    def forward(self, x):
        return (x - self.mean) / self.std

def flow_to_image(flow):
    if flow.ndim not in (2, 4) or flow.dtype != torch.float:
        raise ValueError(
            "Input flow tensor must have a shape of (2, H, W) or (N, 2, H, W) and dtype torch.float."
        )

    n, _, h, w = flow.shape

    # Normalize flow values
    flow_x, flow_y = flow[0], flow[1]
    flow_x = (flow_x - (-0.5)) / (0.5 - (-0.5))
    flow_y = (flow_y - (-0.5)) / (0.5 - (-0.5))

    # Convert normalized flow to image
    img = torch.stack([flow_x, flow_y, torch.zeros_like(flow_x)], dim=0)
    img = img.permute(1, 2, 0)
    img = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img)

    if n == 1:
        img = img.squeeze(0)

    return img

if __name__ == "__main__":
    # Example usage
    flow = torch.randn(2, 100, 100)
    img = flow_to_image(flow)
    print(img.shape)  # Output: torch.Size([3, 100, 100])