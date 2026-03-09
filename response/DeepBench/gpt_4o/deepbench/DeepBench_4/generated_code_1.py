import torch
from typing import List, Type
from copy import deepcopy

class CamerasBase:
    def __init__(self, device: torch.device):
        self.device = device

class ExampleCamera(CamerasBase):
    def __init__(self, device: torch.device, tensor_attr: torch.Tensor):
        super().__init__(device)
        self.tensor_attr = tensor_attr

def join_cameras_as_batch(cameras_list: List[CamerasBase]):
    if not all(isinstance(cam, CamerasBase) for cam in cameras_list):
        raise ValueError("All cameras must inherit from CamerasBase.")
    
    first_camera_type = type(cameras_list[0])
    first_camera_device = cameras_list[0].device
    
    for cam in cameras_list:
        if type(cam) != first_camera_type:
            raise ValueError("All cameras must be of the same type.")
        if cam.device != first_camera_device:
            raise ValueError("All cameras must be on the same device.")
    
    # Assume all camera objects have the same set of attributes for simplicity
    attributes = vars(cameras_list[0]).keys()
    batched_attributes = {}
    
    for attr in attributes:
        attr_values = [getattr(cam, attr) for cam in cameras_list]
        try:
            batched_attributes[attr] = torch.cat(attr_values, dim=0)
        except Exception as e:
            raise ValueError(f"Error concatenating attribute {attr}: {str(e)}")
    
    batched_camera = deepcopy(cameras_list[0])
    for attr, value in batched_attributes.items():
        setattr(batched_camera, attr, value)
    
    return batched_camera

if __name__ == "__main__":
    # Sample input tensors
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cam1 = ExampleCamera(device, torch.tensor([[1.0, 2.0]], device=device))
    cam2 = ExampleCamera(device, torch.tensor([[3.0, 4.0]], device=device))

    # Create list of cameras
    cameras_list = [cam1, cam2]

    # Call function and print result
    batched_camera = join_cameras_as_batch(cameras_list)
    print("Batched Camera Tensor Attribute:", batched_camera.tensor_attr)