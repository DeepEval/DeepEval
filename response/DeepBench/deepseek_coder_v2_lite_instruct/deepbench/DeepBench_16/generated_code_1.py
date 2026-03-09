import torch

class Transform3d:
    def __init__(self, dtype: torch.dtype = torch.float32, device: str = "cpu", matrix: Optional[torch.Tensor] = None):
        if matrix is not None:
            self._matrix = matrix.to(device=device, dtype=dtype)
        else:
            self._matrix = torch.eye(4, dtype=dtype, device=device)
        self._transforms = []
        self.device = device
        self.dtype = dtype

    def get_matrix(self) -> torch.Tensor:
        matrix = self._matrix.clone().to(self.device, self.dtype)
        for t in self._transforms:
            matrix = t.matmul(matrix)
        return matrix

    def add_transform(self, transform):
        self._transforms.append(transform)

if __name__ == "__main__":
    # Create a sample 4x4 transformation matrix
    matrix1 = torch.tensor([[1, 0, 0, 1],
                            [0, 1, 0, 0],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1]], dtype=torch.float32)
    
    # Create another sample 4x4 transformation matrix
    matrix2 = torch.tensor([[1, 0, 0, 0],
                            [0, 1, 0, 1],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1]], dtype=torch.float32)
    
    # Create a Transform3d instance
    transform3d = Transform3d(matrix=matrix1)
    
    # Add another transform to the instance
    transform3d.add_transform(matrix2)
    
    # Get the resulting transformation matrix
    result_matrix = transform3d.get_matrix()
    
    # Print the result
    print(result_matrix)