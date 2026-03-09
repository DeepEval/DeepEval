import torch
import torch.nn as nn

class Affine(nn.Module):
    def __init__(self, angle: Optional[torch.Tensor] = None, translation: Optional[torch.Tensor] = None, scale_factor: Optional[torch.Tensor] = None, shear: Optional[torch.Tensor] = None, center: Optional[torch.Tensor] = None, mode: str = "bilinear", padding_mode: str = "zeros", align_corners: bool = True):
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
        # Construct affine transformation matrix
        affine_matrix = torch.zeros((3, 3), dtype=torch.float32)
        if self.angle is not None:
            affine_matrix[0, 0] = math.cos(self.angle)
            affine_matrix[0, 1] = -math.sin(self.angle)
            affine_matrix[1, 0] = math.sin(self.angle)
            affine_matrix[1, 1] = math.cos(self.angle)
        if self.translation is not None:
            affine_matrix[0, 2] = self.translation[0]
            affine_matrix[1, 2] = self.translation[1]
        if self.scale_factor is not None:
            affine_matrix[0, 0] *= self.scale_factor[0]
            affine_matrix[1, 1] *= self.scale_factor[1]
        if self.shear is not None:
            affine_matrix[0, 1] += self.shear[0]
            affine_matrix[1, 0] += self.shear[1]
        if self.center is not None:
            affine_matrix[0, 2] -= self.center[0]
            affine_matrix[1, 2] -= self.center[1]

        # Apply affine transformation to input tensor
        output = torch.nn.functional.affine_grid(affine_matrix, input.size())
        return output
    

if __name__ == "__main__":
    input1 = torch.randn(1, 3, 32, 32)
    # Define affine transformation parameters
    angle = torch.tensor(0.1)
    translation = torch.tensor([0.5, 0.5])
    scale_factor = torch.tensor([2, 2])
    shear = torch.tensor([0.5, 0.5])
    center = torch.tensor([16, 16])

    # Create Affine module
    affine = Affine(angle, translation, scale_factor, shear, center)

    # Forward pass
    output = affine(input1)

    # Print output
    print(output.shape) 