import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# Define a simple CNN architecture
class UndistortNet(nn.Module):
    def __init__(self):
        super(UndistortNet, self).__init__()
        # Define your CNN architecture here
        # For simplicity, let's assume a basic CNN structure
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(128 * 2 * 2, 256)  # Adjust dimensions based on your input shape
        self.fc2 = nn.Linear(256, 3)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 128 * 2 * 2)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Function to undistort an image
def undistort_image(image, K, dist):
    # Flatten the image and split into channels
    batch_size, _, height, width = image.shape
    image_flat = image.view(batch_size, 3, -1)
    
    # Unpack intrinsic camera matrix and distortion coefficients
    K = K.view(batch_size, 3, 3)
    dist = dist.view(batch_size, -1)
    
    # Undistort using the intrinsic camera matrix and distortion coefficients
    undistorted_image = F.affine_grid(K, image_flat.size()).view(-1, 3, height, width)
    undistorted_image = F.grid_sample(image_flat, undistorted_image)
    
    # Reshape back to the original image shape
    undistorted_image = undistorted_image.view(batch_size, 3, height, width)
    return undistorted_image

if __name__ == "__main__":
    import numpy as np

    # Create a sample input tensors
    batch_size = 1
    image = torch.randn(batch_size, 3, 256, 256)  # Example input image with random noise
    K = torch.eye(3)  # Intrinsic camera matrix (identity for simplicity)
    dist = torch.zeros(batch_size, 5)  # Distortion coefficients (5 for full model including radial, tangential, etc.)

    # Call the function
    output = undistort_image(image, K, dist)

    # Print the output shape
    print("Output shape:", output.shape)