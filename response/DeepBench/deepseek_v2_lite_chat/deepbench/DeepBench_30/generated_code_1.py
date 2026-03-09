import torch
from torch import nn

def function_name(in_channels: int, classes: int, num_filters: int = 64) -> None:
    model = nn.Sequential(
        nn.Conv2d(in_channels, num_filters, kernel_size=4, stride=1, padding=1),
        nn.LeakyReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Conv2d(num_filters, num_filters*2, kernel_size=4, stride=1, padding=1),
        nn.LeakyReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Conv2d(num_filters*2, num_filters*4, kernel_size=4, stride=1, padding=1),
        nn.LeakyReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Conv2d(num_filters*4, num_filters*8, kernel_size=4, stride=1, padding=1),
        nn.LeakyReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.ConvTranspose2d(num_filters*8, num_filters*4, kernel_size=4, stride=2, padding=1),
        nn.LeakyReLU(),
        nn.ConvTranspose2d(num_filters*4, num_filters*2, kernel_size=4, stride=2, padding=1),
        nn.LeakyReLU(),
        nn.ConvTranspose2d(num_filters*2, num_filters, kernel_size=4, stride=2, padding=1),
        nn.LeakyReLU(),
        nn.ConvTranspose2d(num_filters, classes, kernel_size=4, stride=2)
    )
    return model

if __name__ == "__main__":
    model = function_name(in_channels=3, classes=10)
    print(model)