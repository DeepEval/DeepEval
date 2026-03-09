import torch
import cv2

def undistort_image(image, K, dist):
  image_tensor = image.cpu().numpy()
  height, width = image_tensor.shape[1], image_tensor.shape[2]
  map_x, map_y = cv2.initUndistortRectifyMap(K, dist, None, (width, height), cv2.CV_32FC1)
  undistorted_image = cv2.remap(image_tensor, map_x, map_y, cv2.INTER_LINEAR)
  return torch.from_numpy(undistorted_image).permute(2, 0, 1).to(image.device)

if __name__ == "__main__":
    image = torch.randn(1, 3, 256, 256)
    K = torch.randn(1, 3, 3)
    dist = torch.randn(1, 5)
    undistorted_image = undistort_image(image, K, dist)
    print(f"Input image shape: {image.shape}")
    print(f"Undistorted image shape: {undistorted_image.shape}")