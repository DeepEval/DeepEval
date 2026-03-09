import torch
import torch.nn as nn

class Kron:
    def __init__(self, factors):
        self.factors = factors

    @staticmethod
    def init_from_model(model, device):
        factors = []
        for param in model.parameters():
            param = param.to(device)
            if param.ndim == 1:  # bias
                factors.append(torch.zeros(int(param.shape[0] ** 0.5), int(param.shape[0] ** 0.5)))
            elif param.ndim >= 2:  # fully connected or convolutional layers
                in_dim = param.shape[0]
                out_dim = param.shape[-1]
                factors.append(torch.zeros(in_dim, in_dim))
                factors.append(torch.zeros(out_dim, out_dim))
            else:
                raise ValueError("Invalid parameter shape")
        return Kron(factors)

if __name__ == "__main__":
    # Example usage
    class SimpleModel(nn.Module):
        def __init__(self):
            super(SimpleModel, self).__init__()
            self.fc1 = nn.Linear(4, 8)
            self.fc2 = nn.Linear(8, 2)
            self.bias1 = nn.Parameter(torch.randn(8))
            self.bias2 = nn.Parameter(torch.randn(2))

    model = SimpleModel()
    device = torch.device("cpu")
    kron = Kron.init_from_model(model, device)
    print(kron.factors)