import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

def toy_model(train_loader, n_epochs=500, fit=True, in_dim=1, out_dim=1, regression=True):
  model = nn.Sequential(
    nn.Linear(in_dim, 64),
    nn.ReLU(),
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, out_dim)
  )
  if fit:
    if regression:
      criterion = nn.MSELoss()
    else:
      criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    for epoch in range(n_epochs):
      for data, target in train_loader:
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
  return model

if __name__ == "__main__":
  sample_data = torch.randn(100, 1)
  sample_target = torch.randint(0, 2, (100, ))
  train_loader = DataLoader(torch.utils.data.TensorDataset(sample_data, sample_target), batch_size=32)
  model = toy_model(train_loader, fit=True, regression=False)
  print(model)