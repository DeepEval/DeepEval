import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader

def toy_model(train_loader, n_epochs=500, fit=True, in_dim=1, out_dim=1, regression=True):
    # Define the model (a simple linear layer)
    model = nn.Sequential(nn.Linear(in_dim, out_dim))
    
    if fit:
        criterion = nn.MSELoss() if regression else nn.CrossEntropyLoss()
        optimizer = Adam(model.parameters())
        
        for epoch in range(n_epochs):
            for data, target in train_loader:
                output = model(data)
                loss = criterion(output, target)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
    return model

if __name__ == "__main__":
    # Sample input values
    batch_size = 32
    num_samples = 1000
    num_features = 4
    num_classes = 10

    # Generate sample input data
    x = torch.randn(num_samples, num_features)
    y = torch.randint(num_classes, (num_samples,))

    # Convert to DataLoader
    train_loader = DataLoader(list(zip(x, y)), batch_size=batch_size)

    # Create and train the model
    model = toy_model(train_loader, in_dim=num_features, out_dim=num_classes, regression=True)

    # Example of using the trained model
    test_x = torch.randn(1, num_features)
    test_y = model(test_x)
    print("Output for test input:", test_y)