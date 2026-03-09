import torch

class Transform3d:
    def __init__(self, dtype: torch.dtype = torch.float32, device: Device = "cpu", matrix: Optional[torch.Tensor] = None):
        self._matrix = matrix
        self._transforms = []
        self.device = device
        self.dtype = dtype

    def get_matrix(self) -> torch.Tensor:
        if self._matrix is not None:
            return self._matrix
        else:
            matrix = torch.eye(4, dtype=self.dtype, device=self.device)
            for transform in self._transforms:
                matrix = matrix @ transform.matrix
            return matrix

if __name__ == "__main__":
    transform1 = Transform3d(dtype=torch.float32, device="cpu", matrix=torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]))
    transform2 = Transform3d(dtype=torch.float32, device="cpu", matrix=torch.tensor([[1, 0, 0, 1], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]))
    transform3 = Transform3d(dtype=torch.float32, device="cpu", matrix=torch.tensor([[1, 0, 0, 2], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]))
    transforms = [transform1, transform2, transform3]
    matrix = Transform3d(dtype=torch.float32, device="cpu", matrix=None).get_matrix()
    print(matrix)