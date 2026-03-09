import torch
import torch.nn as nn

class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(Conv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)

    def forward(self, x):
        return self.conv(x)

if __name__ == "__main__":
    # Create sample input values
    input_tensor = torch.randn(1, 3, 32, 32)  # Batch size 1, 3 channels, 32x32 image

    # Create the Conv2d layer
    conv_layer = Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)

    # Call the forward method
    output_tensor = conv_layer(input_tensor)

    # Print the results
    print(output_tensor.shape)  # Expected output shape: (1, 16, 32, 32)