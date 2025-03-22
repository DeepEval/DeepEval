import torch
import torch.nn as nn

class Conv1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super(Conv1D, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation)
        self.padding = nn.ZeroPad1d(((kernel_size - 1) * dilation, 0))

    def forward(self, x):
        x = self.padding(x)
        x = self.conv(x)
        return x

class Model(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super(Model, self).__init__()
        self.conv1 = Conv1D(in_channels, out_channels, kernel_size, dilation)
        self.conv2 = Conv1D(out_channels, out_channels, kernel_size, dilation)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x

# Example usage:
model = Model(1, 1, 3, 2)
input = torch.randn(1, 1, 10)
output = model(input)