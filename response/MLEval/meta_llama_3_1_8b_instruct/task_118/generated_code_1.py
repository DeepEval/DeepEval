import torch
import torch.nn as nn

# Define a simple neural network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(5, 10)  # input layer (5) -> hidden layer (10)
        self.fc2 = nn.Linear(10, 5)  # hidden layer (10) -> output layer (5)

    def forward(self, x):
        x = torch.relu(self.fc1(x))  # activation function for hidden layer
        x = self.fc2(x)
        return x

# Initialize the network and the tensor
net = Net()
x = torch.randn(10, 5)

# Define the method to evaluate the gradient of the network
def method():
    # Compute the output of the network
    output = net(x)

    # Evaluate the gradient of the output with respect to the input
    output.backward(torch.ones_like(output))
    
    # Return the input, which now contains the gradients
    return x

# Call the method for validation
output = method()

# Print the output to verify that the gradients are being computed correctly
print(output)