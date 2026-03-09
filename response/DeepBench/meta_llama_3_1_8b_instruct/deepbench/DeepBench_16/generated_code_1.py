import torch

class Transform3d:
    def __init__(self, dtype=torch.float32, device="cpu", matrix=None):
        self._matrix = matrix if matrix is not None else torch.eye(4, dtype=dtype,
  device=device)
        self._transforms = []
        self.device = device
        self.dtype = dtype

    def get_matrix(self) -> torch.Tensor:
        matrix = self._matrix
        for transform in self._transforms:
            matrix = torch.mm(transform, matrix)
        return matrix

    def compose(self, transform):
        self._transforms.append(transform)

if __name__ == "__main__":
    # Create sample input values
    transform1 = torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0,
  0, 1]], dtype=torch.float32)
    transform2 = torch.tensor([[2, 0, 0, 0], [0, 2, 0, 0], [0, 0, 1, 0], [0, 0,
  0, 1]], dtype=torch.float32)
    transform3 = torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0,
  0, 1]], dtype=torch.float32)

    # Create Transform3d instance
    t = Transform3d()
    t.compose(transform1)
    t.compose(transform2)
    t.compose(transform3)

    # Get and print the resulting matrix
    matrix = t.get_matrix()
    print(matrix)