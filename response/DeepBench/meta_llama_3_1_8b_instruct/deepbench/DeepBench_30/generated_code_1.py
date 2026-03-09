import torch
import torch.nn as nn
import torch.nn.functional as F

class FCN(nn.Module):
    def __init__(self, in_channels: int, classes: int, num_filters: int = 64):
        super(FCN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, num_filters, kernel_size=3, padding='same')
        self.conv2 = nn.Conv2d(num_filters, num_filters*2, kernel_size=3, padding='same')
        self.conv3 = nn.Conv2d(num_filters*2, num_filters*4, kernel_size=3, padding='same')
        self.conv4 = nn.Conv2d(num_filters*4, num_filters*8, kernel_size=3, padding='same')
        self.conv5 = nn.Conv2d(num_filters*8, classes, kernel_size=3, padding='same')
        self.relu = nn.LeakyReLU(0.1)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.conv5(x)
        return x

def function_name(args):
    return FCN(*args)

if __name__ == "__main__":
    in_channels = 3
    classes = 10
    num_filters = 64
    args = (in_channels, classes, num_filters)
    fcn = function_name(args)
    print(fcn)
    print(fcn.state_dict().keys())
    print(fcn.state_dict().values())
    print(torch.randn(1, in_channels, 32, 32).shape)
    print(fcn(torch.randn(1, in_channels, 32, 32)).shape)