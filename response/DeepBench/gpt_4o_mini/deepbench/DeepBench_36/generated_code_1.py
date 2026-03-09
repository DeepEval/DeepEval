import numpy as np
import torch
from torch import Tensor

class Kernel:
    pass

class ArcCosine(Kernel):
    def __init__(self, order: int):
        self.order = order

    def _J(self, theta: Tensor) -> Tensor:
        if self.order == 0:
            return torch.ones_like(theta)
        elif self.order == 1:
            return -torch.sin(theta)
        elif self.order == 2:
            return -torch.cos(theta)
        else:
            raise ValueError("Order must be 0, 1, or 2")

if __name__ == "__main__":
    order = 1
    arc_cosine = ArcCosine(order)
    theta = torch.tensor([0.0, np.pi/4, np.pi/2])
    result = arc_cosine._J(theta)
    print(result)