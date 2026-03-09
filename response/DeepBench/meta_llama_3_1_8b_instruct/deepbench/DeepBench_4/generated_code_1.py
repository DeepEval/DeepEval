import torch
from typing import List, Type

class CameraBase:
    def __init__(self):
        self.device = None
        self.tensor_attr = None

class CamerasBase(torch.nn.Module):
    def __init__(self):
        super().__init__()

class SampleCamera(CameraBase, CamerasBase):
    def __init__(self, tensor_attr, device):
        super().__init__()
        self.tensor_attr = tensor_attr
        self.device = device

def join_cameras_as_batch(cameras_list: List[CameraBase]) -> CameraBase:
    if not all(isinstance(camera, CamerasBase) for camera in cameras_list):
        raise ValueError("All cameras must inherit from CamerasBase")
    
    if not all(isinstance(camera, type(cameras_list[0])) for camera in cameras_list):
        raise ValueError("All cameras must be of the same type")
    
    if not all(camera.device == cameras_list[0].device for camera in cameras_list):
        raise ValueError("All cameras must be on the same device")
    
    attr_names = set()
    for camera in cameras_list:
        attr_names.update(camera.tensor_attr.keys())
    
    if len(attr_names)!= len(cameras_list[0].tensor_attr):
        raise ValueError("Inconsistent attributes across cameras")
    
    for attr in attr_names:
        if not all(attr in camera.tensor_attr for camera in cameras_list):
            raise ValueError("Inconsistent attributes across cameras")
        
        if not all(isinstance(value, torch.Tensor) for value in cameras_list):
            raise ValueError("Attributes must be tensors")
        
        if not all(value.shape == cameras_list[0].tensor_attr[attr].shape for value in cameras_list.tensor_attr[attr]):
            raise ValueError("Attributes must have compatible shapes")
    
    batched_tensor_attr = {attr: torch.cat([camera.tensor_attr[attr] for camera in cameras_list], dim=0) for attr in attr_names}
    return SampleCamera(batched_tensor_attr, cameras_list[0].device)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cameras_list = [
        SampleCamera({"tensor": torch.randn(1, 3)}, device),
        SampleCamera({"tensor": torch.randn(1, 3)}, device),
        SampleCamera({"tensor": torch.randn(1, 3)}, device)
    ]
    
    batched_camera = join_cameras_as_batch(cameras_list)
    print(batched_camera.tensor_attr["tensor"].shape)