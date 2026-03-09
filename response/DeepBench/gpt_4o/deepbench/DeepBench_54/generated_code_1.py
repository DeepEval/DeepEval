import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Affine(nn.Module):
    def __init__(
        self,
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
        batch_size, channels, height, width = input.size()
        
        # Initialize transformation matrix
        angle = self.angle if self.angle is not None else torch.zeros(batch_size)
        translation = self.translation if self.translation is not None else torch.zeros(batch_size, 2)
        scale_factor = self.scale_factor if self.scale_factor is not None else torch.ones(batch_size, 2)
        shear = self.shear if self.shear is not None else torch.zeros(batch_size, 2)
        center = self.center if self.center is not None else torch.tensor([width * 0.5, height * 0.5]).repeat(batch_size, 1)
        
        # Convert angles from degrees to radians
        angle = torch.deg2rad(angle)
        
        # Create rotation matrices
        cos_a = torch.cos(angle)
        sin_a = torch.sin(angle)
        
        rotation_matrix = torch.zeros(batch_size, 2, 2)
        rotation_matrix[:, 0, 0] = cos_a
        rotation_matrix[:, 0, 1] = -sin_a
        rotation_matrix[:, 1, 0] = sin_a
        rotation_matrix[:, 1, 1] = cos_a

        # Scale matrix
        scale_matrix = torch.diag_embed(scale_factor)

        # Shear matrix
        shear_x, shear_y = shear[:, 0], shear[:, 1]
        shear_matrix = torch.eye(2).repeat(batch_size, 1, 1)
        shear_matrix[:, 0, 1] = torch.tan(torch.deg2rad(shear_x))
        shear_matrix[:, 1, 0] = torch.tan(torch.deg2rad(shear_y))

        # Combine transformations
        transform_matrix = rotation_matrix @ scale_matrix @ shear_matrix

        # Add translation and center adjustment
        offset = center - center @ transform_matrix.transpose(1, 2) + translation
        final_transform = torch.zeros(batch_size, 2, 3)
        final_transform[:, :, :2] = transform_matrix
        final_transform[:, :, 2] = offset
        
        # Grid sample
        grid = F.affine_grid(final_transform, input.size(), align_corners=self.align_corners)
        output = F.grid_sample(input, grid, mode=self.mode, padding_mode=self.padding_mode, align_corners=self.align_corners)
        
        return output

if __name__ == "__main__":
    # Sample input
    batch_size, channels, height, width = 1, 1, 5, 5
    input_tensor = torch.arange(batch_size * channels * height * width, dtype=torch.float).view(batch_size, channels, height, width)
    
    # Define transformation parameters
    angle = torch.tensor([45.0])  # Rotate 45 degrees
    translation = torch.tensor([[1.0, 1.0]])  # Translate by (1, 1)
    scale_factor = torch.tensor([[1.0, 1.0]])  # No scaling
    shear = torch.tensor([[0.0, 0.0]])  # No shear
    center = torch.tensor([[2.5, 2.5]])  # Center of the image
    
    # Initialize affine transform
    affine_transform = Affine(angle=angle, translation=translation, scale_factor=scale_factor, shear=shear, center=center)
    
    # Apply transformation
    output_tensor = affine_transform(input_tensor)
    
    # Print input and output
    print("Input Tensor:")
    print(input_tensor)
    print("Transformed Tensor:")
    print(output_tensor)