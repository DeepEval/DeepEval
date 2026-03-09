import torch
import torch.nn as nn
import torch.nn.functional as F

class FCN(nn.Module):
    def __init__(self, in_channels: int, classes: int, num_filters: int = 64) -> None:
        super(FCN, self).__init__()
        self.layer1 = nn.Conv2d(in_channels, num_filters, kernel_size=3, padding='same')
        self.layer2 = nn.Conv2d(num_filters, num_filters*2, kernel_size=3, padding='same')
        self.layer3 = nn.Conv2d(num_filters*2, num_filters*4, kernel_size=3, padding='same')
        self.layer4 = nn.Conv2d(num_filters*4, num_filters*8, kernel_size=3, padding='same')
        self.layer5 = nn.Conv2d(num_filters*8, classes, kernel_size=3, padding='same')

        # Initialize weights
        for layer in [self.layer1, self.layer2, self.layer3, self.layer4, self.layer5]:
            nn.init.kaiming_normal_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, x):
        x = F.leaky_relu(self.layer1(x))
        x = F.leaky_relu(self.layer2(x))
        x = F.leaky_relu(self.layer3(x))
        x = F.leaky_relu(self.layer4(x))
        x = self.layer5(x)
        return x

if __name__ == "__main__":
    # Create a sample input tensor with 3 channels, 64x64 resolution
    sample_input = torch.randn(1, 3, 64, 64)
    
    # Create an FCN instance
    fcn_model = FCN(in_channels=3, classes=10, num_filters=64)
    
    # Perform a forward pass
    output = fcn_model(sample_input)
    
    # Print the output shape
    print(output.shape)  # Should print [1, 10, 64, 64] for the given example