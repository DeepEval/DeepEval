import torch

class ArcCosine:
    def __init__(self, order=0):
        self.order = order

    def _J(self, theta: torch.Tensor) -> torch.Tensor:
        if self.order == 0:
            return torch.acos(theta)
        elif self.order == 1:
            return torch.sqrt(1 - torch.square(theta))
        elif self.order == 2:
            return 0.5 * (torch.sqrt(1 + 2 * torch.square(theta)) - 1)
        else:
            raise ValueError(f"Unsupported order: {self.order}")

if __name__ == "__main__":
    # Create sample input values
    theta = torch.tensor(0.5)

    # Instantiate the ArcCosine class
    arccosine = ArcCosine()

    # Call the _J function
    result = arccosine._J(theta)

    # Print the result
    print(result)