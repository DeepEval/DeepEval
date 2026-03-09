import torch
import cv2
import numpy as np

def undistort_image(image, K, dist):
    batch_size, _, H, W = image.shape
    undistorted_images = []

    for i in range(batch_size):
        img = image[i].permute(1, 2, 0).cpu().numpy()
        K_np = K[i].cpu().numpy()
        dist_np = dist[i].cpu().numpy()

        undistorted_img = cv2.undistort(img, K_np, dist_np)
        undistorted_images.append(torch.tensor(undistorted_img).permute(2, 0, 1))

    return torch.stack(undistorted_images)

if __name__ == "__main__":
    # Sample input values
    batch_size = 1
    C, H, W = 3, 100, 100
    image = torch.rand(batch_size, C, H, W)  # Random image
    K = torch.tensor([[500, 0, 50], [0, 500, 50], [0, 0, 1]], dtype=torch.float32).unsqueeze(0).repeat(batch_size, 1, 1)  # Intrinsic matrix
    dist = torch.tensor([[0.1, -0.2, 0, 0, 0]], dtype=torch.float32).repeat(batch_size, 1)  # Distortion coefficients

    undistorted_image = undistort_image(image, K, dist)
    print(undistorted_image.shape)