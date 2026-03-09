import torch
import torch.nn as nn

class Kron:
    def init_from_model(model, device):
        if not isinstance(model, nn.Module) and not isinstance(model, Iterable):
            raise TypeError(f"Model must be an instance of torch.nn.Module or an iterable of nn.Parameter, but got {type(model)}")
        if isinstance(model, nn.Module):
            model_parameters = model.parameters()
        else:
            model_parameters = model

        kron_factors = []
        for p in model_parameters:
            if p.dim() == 1:
                kron_factors.append(torch.zeros((p.shape[0], p.shape[0]), device=device))
            elif p.dim() >= 2:
                kron_factors.append(torch.zeros((p.shape[0], p.shape[0]), device=device))
                kron_factors.append(torch.zeros((p.shape[1], p.shape[1]), device=device))
            else:
                raise ValueError(f"Invalid parameter dimension. Got {p.dim()}")

        return Kron(kron_factors)

# Example usage
if __name__ == "__main__":
    # Create a sample model
    model = nn.Linear(10, 20)

    # Get Kronecker factors from the model
    kron = Kron.init_from_model(model, torch.device("cpu"))

    # Print Kronecker factors
    for factor in kron.kron_factors:
        print(factor)