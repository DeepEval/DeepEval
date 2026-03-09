import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(10, 50)  # Example input size of 10, adjust as needed
        self.fc2 = nn.Linear(50, 2)   # Adjust output size as needed

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

if __name__ == "__main__":
    # Create an instance of the MLP
    net = MLP()
    
    # Example input values (replace these with actual data)
    x = torch.randn(100, 10)  # 100 samples, each with 10 features
    
    # Forward pass
    output = net(x)
    
    # Print the results
    print("Output shape:", output.shape)
    print("Sample output values:", output[:5, :])