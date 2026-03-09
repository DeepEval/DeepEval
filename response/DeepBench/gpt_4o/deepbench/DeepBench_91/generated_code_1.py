import torch
import cv2
import numpy as np

def undistort_image(image: torch.Tensor, K: torch.Tensor, dist: torch.Tensor) -> torch.Tensor:
    batch_shape = image.shape[:-3]
    C, H, W = image.shape[-3], image.shape[-2], image.shape[-1]
    
    image_np = image.reshape(-1, C, H, W).permute(0, 2, 3, 1).cpu().numpy()
    K_np = K.reshape(-1, 3, 3).cpu().numpy()
    dist_np = dist.reshape(-1, dist.shape[-1]).cpu().numpy()
    
    undistorted_images = []
    for i in range(image_np.shape[0]):
        undistorted_image = cv2.undistort(image_np[i], K_np[i], dist_np[i])
        undistorted_images.append(undistorted_image)
    
    undistorted_images = np.stack(undistorted_images)
    undistorted_images = torch.from_numpy(undistorted_images).permute(0, 3, 1, 2)
    return undistorted_images.reshape(*batch_shape, C, H, W)

if __name__ == "__main__":
    batch_size = 1
    C, H, W = 3, 240, 320
    
    image = torch.rand(batch_size, C, H, W)
    K = torch.tensor([[[1000, 0, W/2], [0, 1000, H/2], [0, 0, 1]]], dtype=torch.float32)
    dist = torch.tensor([[0.1, -0.25, 0, 0, 0.1]], dtype=torch.float32)
    
    undistorted_image = undistort_image(image, K, dist)
    print("Original Image Shape:", image.shape)
    print("Undistorted Image Shape:", undistorted_image.shape)
    print("Undistorted Image:", undistorted_image)