import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

def toy_model(train_loader, n_epochs=500, fit=True, in_dim=1, out_dim=1, regression=True):
    model = nn.Sequential(
        nn.Linear(in_dim, 10),
        nn.ReLU(),
        nn.Linear(10, out_dim)
    )

    if fit:
        loss_fn = nn.MSELoss() if regression else nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters())

        for epoch in range(n_epochs):
            for batch_idx, (data, target) in enumerate(train_loader):
                optimizer.zero_grad()
                output = model(data)
                loss = loss_fn(output, target)
                loss.backward()
                optimizer.step()

    return model

if __name__ == "__main__":
    # Example usage
    input_values = torch.randn(10, 1)
    target_values = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])

    train_dataset = torch.utils.data.TensorDataset(input_values, target_values)
    train_loader = DataLoader(train_dataset, batch_size=4)

    model = toy_model(train_loader, in_dim=1, out_dim=1, regression=True)

    # Print trained model output for sample inputs
    with torch.no_grad():
        output = model(input_values)
        print(output)