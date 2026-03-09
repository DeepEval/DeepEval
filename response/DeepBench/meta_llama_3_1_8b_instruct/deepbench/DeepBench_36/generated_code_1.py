import torch
from torch import TensorType
from typing import Any

class ArcCosine:
    def __init__(self, order: int):
        self.order = order

    def _J(self, theta: TensorType) -> TensorType:
        if self.order == 0:
            return torch.acos(theta)
        elif self.order == 1:
            return -torch.sin(theta) / torch.sqrt(1 - theta**2)
        elif self.order == 2:
            return -torch.cos(theta) / torch.sqrt(1 - theta**2) + theta * torch.sin(theta) / (1 - theta**2)**1.5
        else:
            raise ValueError("Invalid order. Supported orders are 0, 1, and 2.")

if __name__ == "__main__":
    # Create a sample input tensor
    theta = torch.tensor([-0.5, 0.1, 0.8])

    # Create an instance of the ArcCosine class with order 1
    arc_cosine = ArcCosine(1)

    # Call the _J function
    result = arc_cosine._J(theta)

    # Print the result
    print(result)