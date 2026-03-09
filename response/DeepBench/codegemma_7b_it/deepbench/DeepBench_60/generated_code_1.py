import torch
import torch.nn as nn

class DeformConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, mask=None):
        super(DeformConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.mask = mask

        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.Tensor(out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.zeros_(self.bias)

    def forward(self, input, offset, weight, bias):
        assert input.size(1) == self.in_channels
        assert weight.size(0) == self.out_channels
        assert weight.size(1) == self.in_channels
        assert weight.size(2) == self.kernel_size
        assert weight.size(3) == self.kernel_size
        assert offset.size(1) == 2 * self.kernel_size ** 2

        output = deform_conv2d(input, offset, weight, bias, self.stride, self.padding, self.dilation, self.mask)
        return output

def deform_conv2d(input, offset, weight, bias, stride, padding, dilation, mask):
    input_shape = input.shape
    offset_shape = offset.shape
    weight_shape = weight.shape

    # Reshape and transpose weight for efficient computation
    weight = weight.view(-1, weight_shape[2], weight_shape[3], weight_shape[0], weight_shape[1]).permute(3, 0, 4, 1, 2)

    # Expand input and offset for broadcasting
    input = input.unsqueeze(dim=1)
    offset = offset.unsqueeze(dim=2)

    # Compute deformable convolution
    output = torch.empty(input_shape[0], weight_shape[0], input_shape[2] // stride, input_shape[3] // stride)
    for i in range(output.shape[0]):
        for j in range(output.shape[1]):
            output[i, j] = torch.conv2d(input[i, j], weight[j], offset[i, j], stride, padding, dilation)

    # Add bias if provided
    if bias is not None:
        output += bias.view(-1, 1, 1)

    return output

if __name__ == "__main__":
    # Create sample input values
    input = torch.randn(1, 3, 28, 28)
    offset = torch.randn(1, 6, 28, 28)
    weight = torch.randn(6, 3, 3, 3)
    bias = torch.randn(6)

    # Create deformable convolution layer
    deform_conv = DeformConv2d(3, 6, kernel_size=3)

    # Perform convolution operation
    output = deform_conv(input, offset, weight, bias)

    # Print the results
    print(output.shape)