import torch
import torch.nn as nn
import torch.optim as optim

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

def function_name(args):
    input_dim, hidden_dim, output_dim = args
    model = MLP(input_dim, hidden_dim, output_dim)
    return model

if __name__ == "__main__":
    # Sample input values
    input_dim = 784
    hidden_dim = 256
    output_dim = 10
    args = (input_dim, hidden_dim, output_dim)

    # Create a model
    model = function_name(args)

    # Print the model parameters
    for param in model.parameters():
        print(param.size())

    # Create a random input tensor
    x = torch.randn(1, input_dim)

    # Forward pass
    output = model(x)
    print(output.shape)