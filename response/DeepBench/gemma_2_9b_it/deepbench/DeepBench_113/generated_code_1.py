import torch
import torch.nn as nn

class Kron:
    def __init__(self, kron_factors):
        self.kron_factors = kron_factors

    @classmethod
    def init_from_model(cls, model, device):
        if isinstance(model, nn.Module):
            params = model.parameters()
        elif isinstance(model, (tuple, list)):
            params = model
        else:
            raise ValueError("Input model must be an instance of nn.Module or an iterable of nn.Parameter.")

        kron_factors = []
        for param in params:
            if param.dim() == 1:
                kron_factors.append(torch.zeros(param.shape[0], param.shape[0], device=device))
            elif param.dim() >= 2:
                kron_factors.append(torch.zeros(param.shape[0], param.shape[0], device=device))
                kron_factors.append(torch.zeros(param.shape[1], param.shape[1], device=device))
            else:
                raise ValueError("Invalid parameter shape.")
        return cls(kron_factors)

if __name__ == "__main__":
    device = torch.device("cpu")
    model = nn.Linear(10, 20)
    kron = Kron.init_from_model(model, device)
    print(kron.kron_factors)