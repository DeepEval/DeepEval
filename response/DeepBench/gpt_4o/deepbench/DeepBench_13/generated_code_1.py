import torch
import torch.nn as nn

class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'):
        super(Conv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode)

    def forward(self, x):
        return self.conv(x)

if __name__ == "__main__":
    # Define the Conv2d layer
    conv_layer = Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1)
    
    # Create a sample input tensor (e.g., a single-channel 5x5 image)
    input_tensor = torch.randn(1, 1, 5, 5)
    
    # Pass the input through the convolutional layer
    output = conv_layer(input_tensor)
    
    # Print the output
    print(output)