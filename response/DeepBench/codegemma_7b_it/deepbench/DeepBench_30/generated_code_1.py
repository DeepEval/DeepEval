from torch.nn import Conv2d, ReLU, BatchNorm2d, MaxPool2d, Dropout2d, Upsample, Concat, Sequential, Module

class FCN(Module):
    def __init__(self, in_channels: int, classes: int, num_filters: int = 64) -> None:
        super().__init__()

        self.layers = Sequential(
            Conv2d(in_channels, num_filters, 3, padding='same', padding_mode='reflect'),
            BatchNorm2d(num_filters),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=3, stride=2),

            Conv2d(num_filters, num_filters * 2, 3, padding='same', padding_mode='reflect'),
            BatchNorm2d(num_filters * 2),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=3, stride=2),

            Conv2d(num_filters * 2, num_filters * 4, 3, padding='same', padding_mode='reflect'),
            BatchNorm2d(num_filters * 4),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=3, stride=2),

            Conv2d(num_filters * 4, num_filters * 8, 3, padding='same', padding_mode='reflect'),
            BatchNorm2d(num_filters * 8),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=3, stride=2),

            Conv2d(num_filters * 8, num_filters * 16, 3, padding='same', padding_mode='reflect'),
            BatchNorm2d(num_filters * 16),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=3, stride=2),

            Conv2d(num_filters * 16, num_filters * 16, 3, padding='same', padding_mode='reflect'),
            BatchNorm2d(num_filters * 16),
            ReLU(inplace=True),

            Conv2d(num_filters * 16, classes, 1, padding='same', padding_mode='reflect')
        )

    def forward(self, x):
        return self.layers(x)

if __name__ == "__main__":
    # Create sample input values
    inputs = torch.randn(1, 3, 256, 256)

    # Instantiate the FCN model
    model = FCN(in_channels=3, classes=2)

    # Call the model and print the output
    outputs = model(inputs)
    print(outputs.shape)