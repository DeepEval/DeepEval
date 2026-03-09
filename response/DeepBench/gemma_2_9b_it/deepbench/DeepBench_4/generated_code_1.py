from typing import List, Any

class CamerasBase:
    pass

def join_cameras_as_batch(cameras_list: List['CamerasBase']):
    if not all(isinstance(camera, CamerasBase) for camera in cameras_list):
        raise ValueError("All cameras must inherit from CamerasBase.")
    if not all(type(camera) == type(cameras_list[0]) for camera in cameras_list):
        raise ValueError("All cameras must be of the same type.")
    if not all(camera.device == cameras_list[0].device for camera in cameras_list):
        raise ValueError("All cameras must be on the same device.")
    
    batch_cameras = type(cameras_list[0])()
    for camera in cameras_list:
        for attr_name in dir(camera):
            attr_value = getattr(camera, attr_name)
            if attr_name not in dir(batch_cameras) or not isinstance(attr_value, (list, tuple, torch.Tensor)):
                raise ValueError(f"Attribute '{attr_name}' is inconsistently present across cameras or is not compatible for batching.")
            setattr(batch_cameras, attr_name, torch.cat((getattr(batch_cameras, attr_name), attr_value), dim=0) if isinstance(attr_value, (list, tuple, torch.Tensor)) else (getattr(batch_cameras, attr_name) + attr_value))
    return batch_cameras


if __name__ == "__main__":
    camera1 = type(CamerasBase)()
    camera1.tensor_attr = torch.randn(3, 224, 224)
    camera2 = type(CamerasBase)()
    camera2.tensor_attr = torch.randn(3, 224, 224)
    camera3 = type(CamerasBase)()
    camera3.tensor_attr = torch.randn(3, 224, 224)

    batched_camera = join_cameras_as_batch([camera1, camera2, camera3])
    print(batched_camera.tensor_attr.shape)