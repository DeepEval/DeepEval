import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

def toy_model(train_loader, n_epochs=500, fit=True, in_dim=1, out_dim=1, regression=True):
    class SimpleNN(nn.Module):
        def __init__(self):
            super(SimpleNN, self).__init__()
            self.fc1 = nn.Linear(in_dim, 32)
            self.fc2 = nn.Linear(32, out_dim)

        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = self.fc2(x)
            return x

    model = SimpleNN()
    criterion = nn.MSELoss() if regression else nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    if fit:
        for epoch in range(n_epochs):
            for data in train_loader:
                inputs, labels = data
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

    return model

if __name__ == "__main__":
    # Example usage
    # Create dummy data
    inputs = torch.randn(100, 1)
    labels = torch.randn(100, 1)
    dataset = TensorDataset(inputs, labels)
    train_loader = DataLoader(dataset, batch_size=10, shuffle=True)

    # Call the function
    model = toy_model(train_loader, n_epochs=20, fit=True, in_dim=1, out_dim=1, regression=True)

    # Print the model
    print(model)