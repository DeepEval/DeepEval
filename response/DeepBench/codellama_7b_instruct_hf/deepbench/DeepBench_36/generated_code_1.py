from typing import TensorType
import torch

class ArcCosine(Kernel):
    def __init__(self, order: int):
        self.order = order

    def _J(self, theta: TensorType) -> TensorType:
        if self.order == 0:
            return torch.cos(theta)
        elif self.order == 1:
            return -torch.sin(theta)
        elif self.order == 2:
            return -2 * torch.cos(theta)
        else:
            raise ValueError("Invalid order")

# Example usage
if __name__ == "__main__":
    # Create sample input values
    theta = torch.randn(10)

    # Call the function
    result = ArcCosine(0)._J(theta)

    # Print the results
    print(result)