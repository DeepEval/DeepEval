import torch
import torch.nn as nn
import torch.nn.functional as F

class FCN(nn.Module):
    def __init__(self, in_channels: int, classes: int, num_filters: int = 64) -> None:
        super(FCN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, num_filters, kernel_size=3, padding='same')
        self.conv2 = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding='same')
        self.conv3 = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding='same')
        self.conv4 = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding='same')
        self.conv5 = nn.Conv2d(num_filters, classes, kernel_size=3, padding='same')
        self.leaky_relu = nn.LeakyReLU(0.1)

    def forward(self, x):
        x = self.leaky_relu(self.conv1(x))
        x = self.leaky_relu(self.conv2(x))
        x = self.leaky_relu(self.conv3(x))
        x = self.leaky_relu(self.conv4(x))
        x = self.conv5(x)  # No activation on the final layer for logits
        return x

if __name__ == "__main__":
    # Create a sample input tensor with shape (batch_size, in_channels, height, width)
    sample_input = torch.randn(1, 3, 256, 256)  # Example: batch size of 1, 3 input channels, 256x256 image

    # Initialize the FCN model
    model = FCN(in_channels=3, classes=10, num_filters=64)

    # Run the sample input through the model
    output = model(sample_input)

    # Print the shape of the output tensor
    print("Output shape:", output.shape)