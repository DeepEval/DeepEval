import torch
from torch import nn

class Affine(nn.Module):
    def __init__( self, angle: Optional[torch.Tensor] = None, translation: Optional[torch.Tensor] = None, scale_factor: Optional[torch.Tensor] = None, shear: Optional[torch.Tensor] = None, center: Optional[torch.Tensor] = None, mode: str = "bilinear", padding_mode: str = "zeros", align_corners: bool = True, ) -> None: 
        super().__init__()
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

        # Construct affine transformation matrix
        transform_matrix = torch.eye(3).float().to(input.device)

        if self.angle is not None:
            angle_rad = self.angle.item() * torch.pi / 180
            transform_matrix[:2, :2] = torch.tensor([[torch.cos(angle_rad), -torch.sin(angle_rad)], [torch.sin(angle_rad), torch.cos(angle_rad)]])

        if self.translation is not None:
            transform_matrix[:2, 2] = self.translation.view(1, -1)

        if self.scale_factor is not None:
            transform_matrix[:2, :2] *= self.scale_factor.view(1, 1)

        if self.shear is not None:
            transform_matrix[0, 1] = self.shear.item()

        if self.center is not None:
            transform_matrix[:2, 2] = transform_matrix[:2, 2] - self.center.view(1, -1) 

        # Apply transformation using F.affine()
        return F.affine(input, transform_matrix, mode=self.mode, padding_mode=self.padding_mode, align_corners=self.align_corners)

if __name__ == "__main__":
    input_tensor = torch.randn(1, 3, 10, 10)
    affine = Affine(angle=30, translation=torch.tensor([1, 2]), scale_factor=1.5)
    transformed_tensor = affine(input_tensor)
    print(transformed_tensor.shape)