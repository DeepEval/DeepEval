import torch
import torch.nn as nn

class FCN(nn.Module):
    def __init__(self, in_channels: int, classes: int, num_filters: int = 64) -> None:
        super(FCN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, num_filters, kernel_size=3, padding='same')
        self.relu1 = nn.LeakyReLU(0.1)
        self.conv2 = nn.Conv2d(num_filters, num_filters * 2, kernel_size=3, padding='same')
        self.relu2 = nn.LeakyReLU(0.1)
        self.conv3 = nn.Conv2d(num_filters * 2, num_filters * 4, kernel_size=3,
  padding='same')
        self.relu3 = nn.LeakyReLU(0.1)
        self.conv4 = nn.Conv2d(num_filters * 4, num_filters * 8, kernel_size=3,
  padding='same')
        self.relu4 = nn.LeakyReLU(0.1)
        self.conv5 = nn.Conv2d(num_filters * 8, classes, kernel_size=1)
    

if __name__ == "__main__":
    model = FCN(in_channels=3, classes=10)
    sample_input = torch.randn(1, 3, 224, 224)
    output = model(sample_input)
    print(output.shape)