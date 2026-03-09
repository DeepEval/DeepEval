class CamerasBase:
    def __init__(self, tensor):
        self.tensor = tensor

    def to(self, device):
        self.tensor = self.tensor.to(device)
        return self

class Camera(CamerasBase):
    def __init__(self, tensor):
        super().__init__(tensor)

def join_cameras_as_batch(cameras_list):
    if not cameras_list:
        raise ValueError("The list of cameras is empty")

    first_camera = cameras_list[0]
    if not isinstance(first_camera, CamerasBase):
        raise ValueError("All cameras must inherit from CamerasBase")

    camera_type = type(first_camera)
    device = first_camera.tensor.device

    for camera in cameras_list:
        if not isinstance(camera, CamerasBase):
            raise ValueError("All cameras must inherit from CamerasBase")
        if type(camera) != camera_type:
            raise ValueError("All cameras must be of the same type")
        if camera.tensor.device != device:
            raise ValueError("All cameras must be on the same device")

    tensor_keys = cameras_list[0].tensor.keys()
    for camera in cameras_list[1:]:
        if camera.tensor.keys() != tensor_keys:
            raise ValueError("Inconsistent tensor attributes across cameras")

    concatenated_tensor = {}
    for key in tensor_keys:
        tensors = [camera.tensor[key] for camera in cameras_list]
        concatenated_tensor[key] = torch.stack(tensors)

    batched_camera = camera_type(**concatenated_tensor)
    return batched_camera

if __name__ == "__main__":
    import torch

    # Sample input values
    cam1 = Camera(torch.tensor({'a': torch.randn(3, 4), 'b': torch.randn(3, 5)}))
    cam2 = Camera(torch.tensor({'a': torch.randn(3, 4), 'b': torch.randn(3, 5)}))
    cam3 = Camera(torch.tensor({'a': torch.randn(3, 4), 'b': torch.randn(3, 5)}))

    # Call the function
    batched_camera = join_cameras_as_batch([cam1, cam2, cam3])

    # Print the results
    print(batched_camera.tensor)