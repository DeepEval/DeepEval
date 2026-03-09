import torch
import torch.nn as nn
import torch.nn.functional as F

class Affine(nn.Module):
    def __init__(self, 
                 angle: Optional[torch.Tensor] = None, 
                 translation: Optional[torch.Tensor] = None, 
                 scale_factor: Optional[torch.Tensor] = None, 
                 shear: Optional[torch.Tensor] = None, 
                 center: Optional[torch.Tensor] = None, 
                 mode: str = "bilinear", 
                 padding_mode: str = "zeros", 
                 align_corners: bool = True, 
                 ) -> None:
        super(Affine, self).__init__()
        self.angle = angle
        self.translation = translation
        self.scale_factor = scale_factor
        self.shear = shear
        self.center = center
        self.mode = mode
        self.padding_mode = padding_mode
        self.align_corners = align_corners

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        batch_size, channels, height, width = input.shape
        if self.angle is not None:
            theta = torch.tensor([[1, self.shear, self.center[0]], 
                                  [self.shear, 1, self.center[1]], 
                                  [0, 0, 1]], dtype=torch.float32).to(input.device)
            theta[:2, :2] = theta[:2, :2] * self.scale_factor.unsqueeze(-1)
            theta[:2, :2] = torch.rot90(theta[:2, :2], -self.angle.item(), (1, 1))
            theta[:2, :2] = theta[:2, :2] * torch.eye(2, device=input.device)
            theta = theta.unsqueeze(0).repeat(batch_size, 1, 1, 1)
        else:
            theta = torch.eye(3, device=input.device).unsqueeze(0).repeat(batch_size, 1, 1)

        grid = F.affine_grid(theta, input.size())
        output = F.grid_sample(input, grid, mode=self.mode, padding_mode=self.padding_mode, align_corners=self.align_corners)
        return output

if __name__ == "__main__":
    # Create a random input tensor
    input_tensor = torch.randn(1, 3, 224, 224)

    # Create an instance of the Affine class
    affine = Affine(angle=torch.tensor(30), 
                     translation=torch.tensor([10, 10]), 
                     scale_factor=torch.tensor(2), 
                     shear=torch.tensor(1), 
                     center=torch.tensor([100, 100]))

    # Apply the affine transformation to the input tensor
    output_tensor = affine(input_tensor)

    # Print the shape of the output tensor
    print(output_tensor.shape)

    # Print the first 5 elements of the output tensor
    print(output_tensor[:1, :, :5, :5])