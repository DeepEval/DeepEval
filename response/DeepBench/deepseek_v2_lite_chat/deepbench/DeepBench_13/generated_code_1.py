import torch
from torch import nn

class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(Conv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        
    def forward(self, x):
        return self.conv(x)

if __name__ == "__main__":
    # Create sample input values
    input_tensor = torch.randn(1, 3, 64, 64)
    
    # Call the function
    layer = Conv2d(3, 64, 3)
    output = layer(input_tensor)
    
    # Print the results
    print(output)