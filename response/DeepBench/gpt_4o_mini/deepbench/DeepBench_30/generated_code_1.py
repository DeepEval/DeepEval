import torch
import torch.nn as nn

class FCN(nn.Module):
    def __init__(self, in_channels: int, classes: int, num_filters: int = 64) -> None:
        super(FCN, self).__init__()
        self.layer1 = nn.Conv2d(in_channels, num_filters, kernel_size=3, padding='same')
        self.act1 = nn.LeakyReLU(negative_slope=0.01)
        self.layer2 = nn.Conv2d(num_filters, num_filters * 2, kernel_size=3, padding='same')
        self.act2 = nn.LeakyReLU(negative_slope=0.01)
        self.layer3 = nn.Conv2d(num_filters * 2, num_filters * 4, kernel_size=3, padding='same')
        self.act3 = nn.LeakyReLU(negative_slope=0.01)
        self.layer4 = nn.Conv2d(num_filters * 4, num_filters * 8, kernel_size=3, padding='same')
        self.act4 = nn.LeakyReLU(negative_slope=0.01)
        self.layer5 = nn.Conv2d(num_filters * 8, classes, kernel_size=3, padding='same')
        
    def forward(self, x):
        x = self.act1(self.layer1(x))
        x = self.act2(self.layer2(x))
        x = self.act3(self.layer3(x))
        x = self.act4(self.layer4(x))
        x = self.layer5(x)
        return x

if __name__ == "__main__":
    model = FCN(in_channels=3, classes=21, num_filters=64)
    sample_input = torch.randn(1, 3, 256, 256)
    output = model(sample_input)
    print(output.shape)