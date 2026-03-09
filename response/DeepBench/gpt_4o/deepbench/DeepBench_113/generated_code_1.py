import torch
import torch.nn as nn

class Kron:
    def __init__(self, kronecker_factors):
        self.kronecker_factors = kronecker_factors

    @classmethod
    def init_from_model(cls, model, device):
        if isinstance(model, nn.Module):
            parameters = model.parameters()
        elif isinstance(model, (list, tuple, set)):
            parameters = model
        else:
            raise ValueError("Input must be an instance of nn.Module or an iterable of nn.Parameter")

        kronecker_factors = []

        for param in parameters:
            shape = param.shape
            if len(shape) == 1:
                # Bias: create a square matrix of zeros
                kronecker_factors.append((torch.zeros((shape[0], shape[0]), device=device),))
            elif len(shape) >= 2:
                # Fully connected or convolutional layers
                in_features = shape[0]
                out_features = shape[1]
                kronecker_factors.append(
                    (torch.zeros((in_features, in_features), device=device),
                     torch.zeros((out_features, out_features), device=device))
                )
            else:
                raise ValueError(f"Invalid parameter shape: {shape}")

        return cls(kronecker_factors)

if __name__ == "__main__":
    # Create a simple neural network model
    class SimpleNN(nn.Module):
        def __init__(self):
            super(SimpleNN, self).__init__()
            self.fc1 = nn.Linear(4, 3)
            self.fc2 = nn.Linear(3, 2)

        def forward(self, x):
            x = self.fc1(x)
            x = self.fc2(x)
            return x

    # Initialize model and device
    model = SimpleNN()
    device = torch.device('cpu')

    # Initialize Kron object from the model
    kron_instance = Kron.init_from_model(model, device)

    # Print the Kronecker factors
    for idx, factors in enumerate(kron_instance.kronecker_factors):
        print(f"Layer {idx + 1}:")
        for factor in factors:
            print(factor)