import torch
import torch.nn as nn

class SimpleNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SimpleNet, self).__init__()
        self.linear1 = nn.Linear(input_dim, 10)
        self.linear2 = nn.Linear(10, output_dim)

    def forward(self, x):
        x = torch.relu(self.linear1(x))
        x = self.linear2(x)
        return x

# Instantiate the network
input_dim = 784  # Example input dimension for MNIST
output_dim = 10  # Example output dimension for MNIST
model = SimpleNet(input_dim, output_dim)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Training loop (for demonstration purposes only)
for epoch in range(10):
    # ... Training logic ...

# Example input
input_data = torch.randn(1, input_dim)

# Generate output
output = model(input_data)

# Print output
print(output)