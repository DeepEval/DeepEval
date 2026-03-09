import numpy as np

class Conv2d:
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.weights = np.random.randn(out_channels, in_channels, kernel_size,
  kernel_size)
        self.bias = np.random.randn(out_channels)

if __name__ == "__main__":
    input_data = np.random.randn(1, 3, 10, 10)
    conv_layer = Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=2,
  padding=1)
    output = conv_layer.weights 
    print(output.shape)