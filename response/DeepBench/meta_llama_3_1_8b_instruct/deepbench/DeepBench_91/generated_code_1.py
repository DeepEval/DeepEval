import torch
import cv2

def undistort_image(image, K, dist):
    # Get the device of the input tensors
    device = image.device
    
    # Convert the input tensors to numpy arrays
    K = K.cpu().numpy()
    dist = dist.cpu().numpy()
    
    # Apply the undistortion function from OpenCV
    map_x, map_y = cv2.initUndistortRectifyMap(K, dist, None, K, (image.shape[-2], image.shape[-1]), cv2.CV_32F)
    undistorted_image = cv2.remap(image.permute(0, 2, 3, 1).cpu().numpy(), map_x, map_y, cv2.INTER_LINEAR).astype(torch.float32)
    
    # Convert the result back to a torch tensor
    undistorted_image = torch.from_numpy(undistorted_image).permute(0, 3, 1, 2).to(device)
    
    return undistorted_image

if __name__ == "__main__":
    # Create sample input values
    image = torch.randn(1, 3, 640, 480)
    K = torch.tensor([[[1000., 0., 320.],
                      [0., 1000., 240.],
                      [0., 0., 1.]]])
    dist = torch.tensor([[[0.1, -0.2, 0., 0., 0.]]])
    
    # Call the function
    undistorted_image = undistort_image(image, K, dist)
    
    # Print the results
    print(undistorted_image.shape)