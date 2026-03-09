import torch
import torch.nn as nn
import torch.optim as optim

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

if __name__ == "__main__":
    # Define the network parameters
    input_size = 28 * 28
    hidden_size = 128
    output_size = 10

    # Create a simple feed-forward network
    mlp = MLP(input_size, hidden_size, output_size)

    # Create sample input values (e.g., a batch of 3 images of 28x28 pixels)
    sample_input = torch.randn(3, input_size)

    # Call the function and print the results
    output = mlp(sample_input)
    print(output)