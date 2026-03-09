import torch
from typing import Optional

class Transform3d:
    def __init__(self, dtype: torch.dtype = torch.float32, device: str = "cpu", matrix: Optional[torch.Tensor] = None):
        self.device = device
        self.dtype = dtype
        if matrix is None:
            self._matrix = torch.eye(4, dtype=self.dtype, device=self.device)
        else:
            self._matrix = matrix.to(dtype=self.dtype, device=self.device)
        self._transforms = []

    def add_transform(self, transform):
        self._transforms.append(transform)

    def get_matrix(self) -> torch.Tensor:
        result = self._matrix.unsqueeze(0)  # Start with the current transform matrix
        for transform in self._transforms:
            result = result @ transform.get_matrix().unsqueeze(0)  # Matrix multiplication
        return result.squeeze(0)  # Remove the batch dimension

class Rotation3d(Transform3d):
    def __init__(self, angle: float, dtype: torch.dtype = torch.float32, device: str = "cpu"):
        super().__init__(dtype=dtype, device=device)
        cos_theta = torch.cos(torch.tensor(angle, dtype=dtype, device=device))
        sin_theta = torch.sin(torch.tensor(angle, dtype=dtype, device=device))
        self._matrix = torch.tensor([[cos_theta, -sin_theta, 0, 0],
                                      [sin_theta, cos_theta, 0, 0],
                                      [0, 0, 1, 0],
                                      [0, 0, 0, 1]], dtype=dtype, device=device)

class Translation3d(Transform3d):
    def __init__(self, x: float, y: float, z: float, dtype: torch.dtype = torch.float32, device: str = "cpu"):
        super().__init__(dtype=dtype, device=device)
        self._matrix = torch.tensor([[1, 0, 0, x],
                                      [0, 1, 0, y],
                                      [0, 0, 1, z],
                                      [0, 0, 0, 1]], dtype=dtype, device=device)

if __name__ == "__main__":
    rotation = Rotation3d(angle=3.14 / 2)  # 90 degrees rotation
    translation = Translation3d(x=1.0, y=2.0, z=3.0)

    transform = Transform3d()
    transform.add_transform(rotation)
    transform.add_transform(translation)

    result_matrix = transform.get_matrix()
    print(result_matrix)