class FCN(nn.Module):
    def __init__(self, in_channels: int, classes: int, num_filters: int = 64):
        super(FCN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, num_filters, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(num_filters, classes, kernel_size=3, padding=1)

        self.relu1 = nn.LeakyReLU(inplace=True)
        self.relu2 = nn.LeakyReLU(inplace=True)
        self.relu3 = nn.LeakyReLU(inplace=True)
        self.relu4 = nn.LeakyReLU(inplace=True)
        self.relu5 = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.relu3(self.conv3(x))
        x = self.relu4(self.conv4(x))
        x = self.relu5(self.conv5(x))
        return x

# Test the FCN model
if __name__ == "__main__":
    # Create sample input values
    x = torch.randn(1, 3, 256, 256)

    # Initialize the FCN model
    model = FCN(in_channels=3, classes=10, num_filters=64)

    # Print the output
    print(model(x))