import torch
import torch.nn as nn
import torch.nn.functional as F


def method():
    """
    Define a simple neural network with one hidden layer.
    
    Returns:
    output (torch.Tensor): The output of the network for an input of shape (1, 784).
    """
    
    # Define the network architecture
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.fc1 = nn.Linear(784, 128)  # Input layer (28x28 images) -> Hidden layer (128 units)
            self.fc2 = nn.Linear(128, 10)  # Hidden layer (128 units) -> Output layer (10 units)

        def forward(self, x):
            x = torch.relu(self.fc1(x))  # Activation function for hidden layer
            x = self.fc2(x)
            return x

    # Initialize the network
    network = Net()

    # Define the input and set it to the network
    input_data = torch.randn(1, 784)  # Random input of shape (1, 784)
    output = network(input_data)

    return output


if __name__ == "__main__":
    # Call the method for validation
    output = method()
    print("Output Shape:", output.shape)
    print("Output Values:", output.detach().numpy())