import torch

class Transform3d:
    def __init__(self, dtype: torch.dtype = torch.float32, device: torch.device = torch.device("cpu"), matrix: Optional[torch.Tensor] = None):
        self._matrix = matrix
        self._transforms = []
        self.device = device
        self.dtype = dtype

    def get_matrix(self) -> torch.Tensor:
        matrix = self._matrix
        for transform in self._transforms:
            matrix = torch.matmul(transform._matrix, matrix)
        return matrix

if __name__ == "__main__":
    # Sample input values
    transform = Transform3d()
    transform._matrix = torch.tensor([[1.0, 0.0, 0.0, 1.0],
                                   [0.0, 1.0, 0.0, 2.0],
                                   [0.0, 0.0, 1.0, 3.0],
                                   [0.0, 0.0, 0.0, 1.0]])
    transform._transforms.append(Transform3d(matrix=torch.tensor([[2.0, 0.0, 0.0, 4.0],
                                                             [0.0, 3.0, 0.0, 5.0],
                                                             [0.0, 0.0, 4.0, 6.0],
                                                             [0.0, 0.0, 0.0, 1.0]])))

    # Call the function and print the results
    matrix = transform.get_matrix()
    print(matrix)