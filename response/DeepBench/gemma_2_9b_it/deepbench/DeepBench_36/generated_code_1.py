import torch
from torch import Tensor

class ArcCosine(Kernel):
    def _J(self, theta: TensorType) -> TensorType:
        order = self.order
        if order == 0:
            return torch.ones_like(theta)
        elif order == 1:
            return -torch.sin(theta) / torch.sqrt(1 - torch.pow(torch.cos(theta), 2))
        elif order == 2:
            return (torch.sin(theta) * torch.cos(theta)) / torch.pow(torch.sqrt(1 - torch.pow(torch.cos(theta), 2)), 3)
        else:
            raise ValueError(f"Unsupported order: {order}")

if __name__ == "__main__":
    theta = torch.randn(3)
    arccos = ArcCosine()
    arccos.order = 1
    result = arccos._J(theta)
    print(result)