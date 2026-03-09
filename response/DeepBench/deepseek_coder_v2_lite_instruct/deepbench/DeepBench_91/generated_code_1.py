import torch
import numpy as np
import cv2

def undistort_image(image, K, dist):
    h, w = image.shape[-2:]
    new_K, roi = cv2.getOptimalNewCameraMatrix(K.numpy(), dist.numpy(), (w, h), 1, (w, h))
    new_K = torch.tensor(new_K).unsqueeze(0)
    map1, map2 = cv2.initUndistortRectifyMap(K.numpy(), dist.numpy(), None, new_K.numpy(), (w, h), cv2.CV_16SC2)
    map1 = torch.tensor(map1).unsqueeze(0)
    map2 = torch.tensor(map2).unsqueeze(0)
    undistorted_image = cv2.remap(image.numpy(), map1.numpy(), map2.numpy(), cv2.INTER_LINEAR)
    return torch.tensor(undistorted_image)

if __name__ == "__main__":
    # Create sample input values
    image = torch.randn(1, 3, 480, 640)  # Example image tensor
    K = torch.tensor([[480, 0, 320], [0, 480, 240], [0, 0, 1]])  # Example intrinsic camera matrix
    dist = torch.tensor([0.1, 0.01, 0.001, 0.0001, 0.0001, 0.0001])  # Example distortion coefficients

    # Call the function and print the results
    undistorted_img = undistort_image(image, K, dist)
    print(undistorted_img.shape)