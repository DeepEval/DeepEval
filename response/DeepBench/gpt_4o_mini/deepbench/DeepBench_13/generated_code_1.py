import torch
import torch.nn as nn

class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(Conv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)

    def forward(self, x):
        return self.conv(x)

if __name__ == "__main__":
    input_tensor = torch.randn(1, 3, 64, 64)  # Batch size 1, 3 channels, 64x64 image
    model = Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
    output = model(input_tensor)
    print(output.shape)  # Should print the shape of the output tensor