import torch

class Transform3d:
    def __init__(self, dtype: torch.dtype = torch.float32, device: str = "cpu", matrix: Optional[torch.Tensor] = None):
        self._matrix = matrix
        self._transforms = []
        self.device = torch.device(device)
        self.dtype = dtype

    def get_matrix(self) -> torch.Tensor:
        matrix = torch.eye(4, dtype=self.dtype, device=self.device)
        for transform in self._transforms:
            matrix = torch.matmul(matrix, transform)
        return matrix

if __name__ == "__main__":
    transform1 = torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=torch.float32, device="cpu")
    transform2 = torch.tensor([[1, 0, 0, 2], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=torch.float32, device="cpu")

    t = Transform3d()
    t._transforms = [transform1, transform2]
    result_matrix = t.get_matrix()
    print(result_matrix)