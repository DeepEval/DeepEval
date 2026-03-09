import torch

class CamerasBase:
    def __init__(self, tensors):
        self.tensors = tensors

    def to(self, device):
        self.tensors = self.tensors.to(device)

class CameraA(CamerasBase):
    pass

class CameraB(CamerasBase):
    pass

def join_cameras_as_batch(cameras_list):
    if not all(isinstance(camera, CamerasBase) for camera in cameras_list):
        raise ValueError("All cameras must inherit from CamerasBase.")
    
    camera_type = type(cameras_list[0])
    if not all(isinstance(camera, camera_type) for camera in cameras_list):
        raise ValueError("All cameras must be of the same type.")
    
    device = cameras_list[0].tensors.device
    if not all(camera.tensors.device == device for camera in cameras_list):
        raise ValueError("All cameras must be on the same device.")

    batch_attributes = {}
    for camera in cameras_list:
        for attr_name, attr_value in camera.__dict__.items():
            if attr_name not in batch_attributes:
                batch_attributes[attr_name] = attr_value.unsqueeze(0)  # Start batch dimension
            else:
                if attr_value.size(1) != batch_attributes[attr_name].size(1):
                    raise ValueError(f"Incompatible attribute sizes for '{attr_name}'.")
                batch_attributes[attr_name] = torch.cat((batch_attributes[attr_name], attr_value.unsqueeze(0)), dim=0)
    
    return camera_type(**batch_attributes)

if __name__ == "__main__":
    tensor1 = torch.rand(3, 2, 2).to('cuda:0')
    tensor2 = torch.rand(3, 2, 2).to('cuda:0')
    camera1 = CameraA(tensor1)
    camera2 = CameraA(tensor2)

    batched_camera = join_cameras_as_batch([camera1, camera2])
    print(batched_camera.tensors)