import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

def toy_model(train_loader, n_epochs=500, in_dim=1, out_dim=1, fit=True, regression=True):
    model = nn.Sequential(
        nn.Linear(in_dim, 10),
        nn.ReLU(),
        nn.Linear(10, out_dim)
    )
    
    if fit:
        criterion = nn.MSELoss() if regression else nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        for epoch in range(n_epochs):
            for inputs, targets in train_loader:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

    return model

if __name__ == "__main__":
    # Create some sample data
    x = torch.randn(100, 1)
    y = x * 2 + 1 if True else torch.randint(0, 2, (100,))
    dataset = TensorDataset(x, y)
    train_loader = DataLoader(dataset, batch_size=10, shuffle=True)
    
    # Train the model
    trained_model = toy_model(train_loader, n_epochs=100, in_dim=1, out_dim=1, fit=True, regression=True)
    
    # Print the model
    print(trained_model)