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
    # Creating sample data
    x_train = torch.randn(100, 1)
    y_train = (x_train > 0).float()  # Binary classification
    
    train_dataset = TensorDataset(x_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)
    
    # Calling the toy_model function
    model = toy_model(train_loader, n_epochs=10, in_dim=1, out_dim=1, fit=True, regression=False)
    
    # Print the model to validate
    print(model)