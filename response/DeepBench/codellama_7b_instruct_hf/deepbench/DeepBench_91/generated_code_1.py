import torch
from torch.nn import Module

class UndistortImage(Module):
    def __init__(self, K, dist):
        super(UndistortImage, self).__init__()
        self.K = K
        self.dist = dist

    def forward(self, image):
        undistorted_image = image
        for i in range(len(self.dist)):
            undistorted_image = self.distort_image(undistorted_image, self.dist[i])
        return undistorted_image

    def distort_image(self, image, dist):
        # Radial distortion model
        radial_distortion = torch.sqrt(1 + (dist[:, 0] * image)**2)
        tangential_distortion = dist[:, 1] * image * radial_distortion
        thin_prism_distortion = dist[:, 2] * image * radial_distortion
        tilt_distortion = dist[:, 3] * image * radial_distortion

        # Inverse of the distortion matrix
        inv_distortion = torch.inverse(dist)

        # Apply the distortion
        distorted_image = image * inv_distortion

        return distorted_image

if __name__ == "__main__":
    # Generate sample input values
    image = torch.randn(1, 3, 256, 256)
    K = torch.randn(1, 3, 3)
    dist = torch.randn(1, 4)

    # Create an instance of the UndistortImage class
    undistort_image = UndistortImage(K, dist)

    # Call the forward pass
    undistorted_image = undistort_image(image)

    # Print the results
    print("Original image:")
    print(image)
    print("Undistorted image:")
    print(undistorted_image)