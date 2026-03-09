import torch
import abc

class CamerasBase(abc.ABC):
    @abc.abstractmethod
    def get_tensor(self):
        pass

class Camera(CamerasBase):
    def __init__(self):
        self.tensor = torch.zeros(1, requires_grad=True)

    def get_tensor(self):
        return self.tensor

def join_cameras_as_batch(cameras_list):
    if not all(isinstance(cam, CamerasBase) for cam in cameras_list):
        raise ValueError("All cameras must inherit from CamerasBase.")

    if not all(isinstance(cam, Camera) for cam in cameras_list):
        raise ValueError("All cameras must be instances of Camera.")

    if len(set([cam.get_tensor().device for cam in cameras_list])) != 1:
        raise ValueError("All cameras must be on the same device.")

    tensors_list = [cam.get_tensor() for cam in cameras_list]
    batch_tensor = torch.cat(tensors_list, dim=0)
    return batch_tensor

if __name__ == "__main__":
    import random

    # Create a list of cameras
    camera_count = random.randint(2, 5)
    cameras_list = [Camera() for _ in range(camera_count)]

    # Join cameras as a batch
    try:
        batch = join_cameras_as_batch(cameras_list)
        print("Joined Camera Tensor:", batch)
        print("Device:", batch.device)
    except ValueError as e:
        print(str(e))