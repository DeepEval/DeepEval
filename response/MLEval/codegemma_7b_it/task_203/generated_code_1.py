import torch
import torch.nn as nn

class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super(CausalConv1d, self).__init__()
        self.padding = (dilation * (kernel_size - 1)) // 2
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation)

    def forward(self, x):
        return self.conv(nn.functional.pad(x, (self.padding, 0)))

# Define the model architecture
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv = CausalConv1d(1, 1, kernel_size=3, dilation=2)

    def forward(self, x):
        x = self.conv(x)
        return x

# Create an instance of the model
model = MyModel()

# Generate some dummy input data
input_data = torch.randn(1, 1, 10)

# Call the method and print the output
output = model(input_data)
print(output.shape)