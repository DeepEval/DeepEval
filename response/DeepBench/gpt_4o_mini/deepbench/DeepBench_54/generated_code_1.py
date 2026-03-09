import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Affine(nn.Module):
    def __init__(self, angle: Optional[torch.Tensor] = None, translation: Optional[torch.Tensor] = None,
                 scale_factor: Optional[torch.Tensor] = None, shear: Optional[torch.Tensor] = None,
                 center: Optional[torch.Tensor] = None, mode: str = "bilinear", padding_mode: str = "zeros",
                 align_corners: bool = True) -> None:
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
        if self.center is None:
            self.center = torch.tensor([width / 2, height / 2]).to(input.device)
        
        angle_rad = self.angle * (math.pi / 180) if self.angle is not None else 0
        cos_a = torch.cos(angle_rad)
        sin_a = torch.sin(angle_rad)

        if self.scale_factor is None:
            scale_x = scale_y = 1.0
        else:
            scale_x, scale_y = self.scale_factor.unbind(dim=-1)

        if self.shear is None:
            shear_x = shear_y = 0.0
        else:
            shear_x, shear_y = self.shear.unbind(dim=-1)

        if self.translation is None:
            translation_x = translation_y = 0.0
        else:
            translation_x, translation_y = self.translation.unbind(dim=-1)

        transformation_matrix = torch.tensor([
            [scale_x * cos_a + shear_y * sin_a, -scale_x * sin_a + shear_y * cos_a, translation_x],
            [shear_x * cos_a - scale_y * sin_a, shear_x * sin_a + scale_y * cos_a, translation_y],
            [0, 0, 1]
        ], device=input.device)

        grid = F.affine_grid(transformation_matrix[:2, :].unsqueeze(0).repeat(batch_size, 1, 1), 
                             input.size(), align_corners=self.align_corners)
        output = F.grid_sample(input, grid, mode=self.mode, padding_mode=self.padding_mode, align_corners=self.align_corners)
        return output

if __name__ == "__main__":
    batch_size = 2
    channels = 3
    height = 64
    width = 64

    input_tensor = torch.rand(batch_size, channels, height, width)
    angle = torch.tensor([30.0, -15.0])
    translation = torch.tensor([[5.0, 10.0], [-5.0, -10.0]])
    scale_factor = torch.tensor([[1.0, 1.0], [0.5, 0.5]])
    shear = torch.tensor([[0.1, 0.0], [0.0, 0.1]])

    affine_transform = Affine(angle=angle, translation=translation, scale_factor=scale_factor, shear=shear)
    output_tensor = affine_transform(input_tensor)

    print("Output Tensor Shape:", output_tensor.shape)