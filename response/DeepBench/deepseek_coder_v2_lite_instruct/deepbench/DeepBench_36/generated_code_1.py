import torch
from torch import Tensor

class ArcCosine(object):
    def __init__(self, order: int):
        self.order = order

    def _J(self, theta: Tensor) -> Tensor:
        if self.order == 0:
            return torch.acos(theta)
        elif self.order == 1:
            return theta * torch.acos(theta) - torch.sqrt(1 - theta**2)
        elif self.order == 2:
            return (3 * theta * torch.acos(theta) - torch.sqrt(1 - theta**2) * theta - torch.pi) / 2
        else:
            raise ValueError("Order must be 0, 1, or 2")

if __name__ == "__main__":
    # Create an instance of ArcCosine for order 1
    arc_cosine = ArcCosine(order=1)

    # Create sample input values
    theta = torch.tensor([0.0, 0.5, 1.0])

    # Call the function and print the results
    result = arc_cosine._J(theta)
    print(result)