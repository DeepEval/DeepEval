import torch
import torch.nn as nn
import torch.nn.functional as F

# Define the simple neural network
class SimpleNetwork(nn.Module):
    def __init__(self):
        super(SimpleNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 50)
        self.fc2 = nn.Linear(50, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Instantiate the network
input_size = 10  # Example input size
output_size = 2  # Example output size
net = SimpleNetwork()

# Example input data
input_data = torch.randn(1, input_size)

# Call the method and get the output
output = method()

# Print the output
print(output)