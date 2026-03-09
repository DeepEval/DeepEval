import torch

class Kron:
    def __init__(self, factors):
        self.factors = factors

    @classmethod
    def init_from_model(cls, model, device):
        if not isinstance(model, torch.nn.Module):
            raise ValueError("Input model must be an instance of torch.nn.Module")

        factors = []
        for param in model.parameters():
            if param.dim() == 1:
                factors.append(torch.zeros(param.size(0), param.size(0)))
            elif param.dim() >= 2:
                factors.append(torch.zeros(param.size(0), param.size(0)))
                factors.append(torch.zeros(param.size(1), param.size(1)))
            else:
                raise ValueError("Invalid parameter shape")

        return cls(factors)
    
if __name__ == "__main__":
    # Create a sample model
    model = torch.nn.Linear(3, 4)

    # Initialize a Kron object from the model
    kron = Kron.init_from_model(model, device="cpu")

    # Print the Kronecker factors
    print(kron.factors)