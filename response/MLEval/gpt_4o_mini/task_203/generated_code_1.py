import torch
import torch.nn as nn
import torch.nn.functional as F

class CausalConv1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super(CausalConv1D, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, 
                              padding=(kernel_size - 1) * dilation, 
                              dilation=dilation)

    def forward(self, x):
        return self.conv(x)

def method():
    # Sample input (batch_size=1, channels=1, sequence_length=10)
    input_tensor = torch.randn(1, 1, 10)  # Example input tensor

    # Define the model
    model = CausalConv1D(in_channels=1, out_channels=1, kernel_size=3, dilation=2)

    # Forward pass
    output = model(input_tensor)

    return output

# Call the method for validation
output = method()
print(output)