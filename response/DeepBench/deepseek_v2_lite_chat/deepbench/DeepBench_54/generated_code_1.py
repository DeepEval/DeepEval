import torch
import torch.nn as nn
import torch.nn.functional as F

class Affine(nn.Module):
    def __init__(self, angle: Optional[torch.Tensor] = None, translation: Optional[torch.Tensor] = None, scale_factor: Optional[torch.Tensor] = None, shear: Optional[torch.Tensor] = None, center: Optional[torch.Tensor] = None, mode: str = "bilinear", padding_mode: str = "zeros", align_corners: bool = True, ):
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
        if self.angle is not None:
            rotation_matrix = self._create_rotation_matrix(self.angle)
            input = F.affine(input, rotation_matrix, self.translation, self.scale_factor, self.shear, self.center, self.mode, self.padding_mode, self.align_corners)
        else:
            raise ValueError("Angle must be provided for Affine transformation")
        return input

    def _create_rotation_matrix(self, angle_radians):
        angle_degrees = angle_radians * (180.0 / 3.14159265359)
        rotation_matrix = torch.zeros(3, 3)
        rotation_matrix[0, 0] = torch.cos(angle_degrees)
        rotation_matrix[0, 1] = -torch.sin(angle_degrees)
        rotation_matrix[0, 2] = 0
        rotation_matrix[1, 0] = torch.sin(angle_degrees)
        rotation_matrix[1, 1] = torch.cos(angle_degrees)
        rotation_matrix[1, 2] = 0
        rotation_matrix[2, 0] = 0
        rotation_matrix[2, 1] = 0
        rotation_matrix[2, 2] = 1
        return rotation_matrix

if __name__ == "__main__":
    batch_size = 2
    channels = 3
    height = 100
    width = 100
    angle = torch.tensor([45])
    translation = torch.tensor([[10, 10]])
    scale_factor = torch.tensor([[2, 2]])
    shear = torch.tensor([[0.1, 0.1]])
    center = torch.tensor([[50, 50]])
    mode = "bilinear"
    padding_mode = "zeros"
    align_corners = True

    input = torch.randn(batch_size, channels, height, width)
    affine = Affine(angle, translation, scale_factor, shear, center, mode, padding_mode, align_corners)
    output = affine(input)
    print(output.shape)