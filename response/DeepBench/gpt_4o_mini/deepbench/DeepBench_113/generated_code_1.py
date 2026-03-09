import torch
import torch.nn as nn

class Kron:
    def __init__(self, factors):
        self.factors = factors

    @classmethod
    def init_from_model(cls, model_or_params, device):
        if isinstance(model_or_params, nn.Module):
            parameters = list(model_or_params.parameters())
        elif isinstance(model_or_params, (list, tuple)):
            parameters = model_or_params
        else:
            raise ValueError("Input must be an instance of nn.Module or an iterable of nn.Parameter")
        
        factors = []
        for param in parameters:
            if param.dim() == 1:  # Bias term
                factors.append(torch.zeros(param.size(0), param.size(0), device=device))
            elif param.dim() >= 2:  # Weight term
                in_features = param.size(0)
                out_features = param.size(1)
                factor = torch.zeros(in_features, out_features, device=device)
                factors.append(factor)
            else:
                raise ValueError(f"Invalid parameter shape: {param.shape}")

        return cls(factors)

if __name__ == "__main__":
    class SampleModel(nn.Module):
        def __init__(self):
            super(SampleModel, self).__init__()
            self.fc1 = nn.Linear(10, 5)
            self.fc2 = nn.Linear(5, 2)
            self.bias = nn.Parameter(torch.randn(5))

        def forward(self, x):
            x = self.fc1(x)
            return self.fc2(x)

    device = torch.device("cpu")
    model = SampleModel()
    kron_instance = Kron.init_from_model(model, device)
    print(kron_instance.factors)