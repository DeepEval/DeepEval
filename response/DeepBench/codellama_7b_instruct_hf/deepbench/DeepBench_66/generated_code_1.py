import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

def toy_model(train_loader, n_epochs=500, fit=True, in_dim=1, out_dim=1, regression=True):
    # Build the model
    model = nn.Sequential(
        nn.Linear(in_dim, 16),
        nn.ReLU(),
        nn.Linear(16, out_dim)
    )

    # Define the loss function and optimizer
    if regression:
        loss_fn = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
    else:
        loss_fn = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    if fit:
        for epoch in range(n_epochs):
            for batch in train_loader:
                inputs, labels = batch
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = loss_fn(outputs, labels)
                loss.backward()
                optimizer.step()

    return model
    
if __name__ == "__main__":
    # Example usage
    inputs = torch.randn(10, 1)
    labels = torch.randn(10, 1)
    train_loader = DataLoader(inputs, labels, batch_size=32)
    model = toy_model(train_loader, n_epochs=100, fit=True, in_dim=1, out_dim=1, regression=True)

    # Print the trained model
    print(model)