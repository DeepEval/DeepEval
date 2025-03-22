import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )
    
    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

def method():
    # Load MNIST dataset
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_dataset = datasets.MNIST(root=".", train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root=".", train=False, download=True, transform=transform)

    # Create neural network and optimizer
    model = NeuralNet()
    optimizer = optim.SGD(model.parameters(), lr=1e-3)

    # Train the model
    for epoch in range(3):
        for batch_idx, (data, target) in enumerate(train_dataset):
            optimizer.zero_grad()
            output = model(data)
            loss = nn.CrossEntropyLoss()(output, target)
            loss.backward()
            optimizer.step()

    # Test the model
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_dataset:
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total += 1
            correct += (predicted == target).item()

    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy}%")

    return accuracy

# Call method for validation
accuracy = method()