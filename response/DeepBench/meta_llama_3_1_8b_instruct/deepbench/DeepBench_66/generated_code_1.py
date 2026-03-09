import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

def toy_model(train_loader, n_epochs=500, fit=True, in_dim=1, out_dim=1, regression=True):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    model = nn.Sequential(
        nn.Linear(in_dim, 10),
        nn.ReLU(),
        nn.Linear(10, 10),
        nn.ReLU(),
        nn.Linear(10, out_dim)
    ).to(device)

    criterion = nn.MSELoss() if regression else nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    if fit:
        model.train()
        for epoch in range(n_epochs):
            for batch in train_loader:
                inputs, labels = batch
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            print(f'Epoch {epoch+1}, Loss: {loss.item()}')
    else:
        model.eval()

    return model

if __name__ == "__main__":
    # Create sample input values
    x = np.random.rand(100, 1)
    y = np.random.rand(100, 1)
    train_loader = DataLoader(list(zip(x, y)), batch_size=10, shuffle=True)

    # Call the function and print the results
    model = toy_model(train_loader)
    print(model)