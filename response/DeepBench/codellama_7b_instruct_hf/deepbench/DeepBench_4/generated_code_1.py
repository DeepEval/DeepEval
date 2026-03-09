import torch

class Camera:
    def __init__(self, tensor):
        self.tensor = tensor

class CamerasBase:
    def __init__(self, cameras_list):
        if not all(isinstance(camera, Camera) for camera in cameras_list):
            raise ValueError("All cameras must inherit from Camera")
        if not all(camera.tensor.device == cameras_list[0].tensor.device for camera in cameras_list):
            raise ValueError("All cameras must be on the same device")
        if not all(camera.tensor.shape[0] == cameras_list[0].tensor.shape[0] for camera in cameras_list):
            raise ValueError("All cameras must have the same number of images")
        self.cameras_list = cameras_list

    def join_cameras_as_batch(self):
        # Check that all attributes are consistent across cameras
        if not all(camera.tensor.shape[1:] == cameras_list[0].tensor.shape[1:] for camera in self.cameras_list):
            raise ValueError("All camera attributes must be consistent across cameras")
        # Check that all attributes are compatible for batching
        if not all(camera.tensor.dtype == cameras_list[0].tensor.dtype for camera in self.cameras_list):
            raise ValueError("All camera attributes must be compatible for batching")
        # Concatenate the tensor attributes of the cameras along the batch dimension
        batched_tensor = torch.cat([camera.tensor for camera in self.cameras_list], dim=0)
        # Create a new camera object with the batched tensor
        batched_camera = Camera(batched_tensor)
        return batched_camera

if __name__ == "__main__":
    # Create some sample input values
    cameras_list = [Camera(torch.randn(10, 3, 224, 224)) for _ in range(5)]
    # Create a CamerasBase object from the sample input values
    cameras = CamerasBase(cameras_list)
    # Call the join_cameras_as_batch function on the CamerasBase object
    batched_camera = cameras.join_cameras_as_batch()
    # Print the batched camera object
    print(batched_camera)