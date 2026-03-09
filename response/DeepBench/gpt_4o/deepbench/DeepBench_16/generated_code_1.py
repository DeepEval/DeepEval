import torch
from typing import Optional

class Transform3d:
    def __init__(self, dtype: torch.dtype = torch.float32, device: str = "cpu", matrix: Optional[torch.Tensor] = None):
        self.dtype = dtype
        self.device = device
        if matrix is not None:
            self._matrix = matrix.to(dtype=self.dtype, device=self.device)
        else:
            self._matrix = torch.eye(4, dtype=self.dtype, device=self.device)
        self._transforms = []

    def add_transform(self, transform: torch.Tensor):
        self._transforms.append(transform.to(dtype=self.dtype, device=self.device))

    def get_matrix(self) -> torch.Tensor:
        result_matrix = self._matrix
        for transform in self._transforms:
            result_matrix = torch.matmul(result_matrix, transform)
        return result_matrix

if __name__ == "__main__":
    # Create a Transform3d object
    transform_obj = Transform3d()

    # Create some example transformation matrices
    translation = torch.tensor([
        [1, 0, 0, 1],
        [0, 1, 0, 2],
        [0, 0, 1, 3],
        [0, 0, 0, 1]
    ], dtype=torch.float32)

    scaling = torch.tensor([
        [2, 0, 0, 0],
        [0, 2, 0, 0],
        [0, 0, 2, 0],
        [0, 0, 0, 1]
    ], dtype=torch.float32)

    rotation = torch.tensor([
        [0, -1, 0, 0],
        [1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ], dtype=torch.float32)

    # Add transformations to the object
    transform_obj.add_transform(translation)
    transform_obj.add_transform(scaling)
    transform_obj.add_transform(rotation)

    # Get the resulting transformation matrix
    result = transform_obj.get_matrix()

    # Print the result
    print("Resulting transformation matrix:")
    print(result)