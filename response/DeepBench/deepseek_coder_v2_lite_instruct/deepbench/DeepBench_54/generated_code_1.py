import torch
import torch.nn as nn
import torch.nn.functional as F

class Affine(nn.Module):
    def __init__(self, angle: Optional[torch.Tensor] = None, translation: Optional[torch.Tensor] = None, scale_factor: Optional[torch.Tensor] = None, shear: Optional[torch.Tensor] = None, center: Optional[torch.Tensor] = None, mode: str = "bilinear", padding_mode: str = "zeros", align_corners: bool = True) -> None:
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

        # Create the affine transformation matrix
        transform_matrix = torch.eye(3).repeat(batch_size, 1, 1).type_as(input)

        if self.angle is not None:
            theta = torch.eye(2).type_as(input)
            theta = F.rotate(theta, self.angle)
            transform_matrix[:, :2, :2] = theta

        if self.translation is not None:
            transform_matrix[:, :2, 2] = self.translation

        if self.scale_factor is not None:
            transform_matrix[:, 0, 0] *= self.scale_factor
            transform_matrix[:, 1, 1] *= self.scale_factor

        if self.shear is not None:
            shear_matrix = torch.eye(2).type_as(input)
            shear_matrix[0, 1] = torch.tan(self.shear)
            transform_matrix[:, :2, :2] = torch.matmul(transform_matrix[:, :2, :2], shear_matrix)

        if self.center is not None:
            transform_matrix[:, :2, 2] -= self.center * (transform_matrix[:, :2, :2] - torch.eye(2).type_as(input))

        # Apply the transformation to the input tensor
        grid = F.affine_grid(transform_matrix[:, :2, :], [batch_size, channels, height, width], align_corners=self.align_corners)
        output = F.grid_sample(input, grid, mode=self.mode, padding_mode=self.padding_mode, align_corners=self.align_corners)

        return output

if __name__ == "__main__":
    # Create a sample input tensor
    batch_size = 1
    channels = 3
    height = 256
    width = 256
    input = torch.randn(batch_size, channels, height, width)

    # Create an instance of the Affine class
    affine_transform = Affine(angle=15, translation=[10, 20], scale_factor=1.2, shear=0.1, center=[0.5, 0.5])

    # Apply the transformation
    output = affine_transform(input)

    # Print the results
    print(output.shape)