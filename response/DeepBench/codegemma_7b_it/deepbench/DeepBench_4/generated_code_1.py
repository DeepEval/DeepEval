import torch

class CamerasBase:
    pass

class Cameras(CamerasBase):
    def __init__(self, tensors):
        self.tensors = tensors

    @property
    def device(self):
        return self.tensors[0].device

def join_cameras_as_batch(cameras_list):
    if not all(isinstance(camera, CamerasBase) for camera in cameras_list):
        raise ValueError("All cameras must inherit from CamerasBase.")

    if not all(isinstance(camera, type(cameras_list[0])) for camera in cameras_list):
        raise ValueError("Cameras must be of the same type.")

    if not all(camera.device == cameras_list[0].device for camera in cameras_list):
        raise ValueError("Cameras must be on the same device.")

    tensors = [camera.tensors for camera in cameras_list]

    for tensors_list in zip(*tensors):
        if not all(tensor.shape == tensors_list[0].shape for tensor in tensors_list):
            raise ValueError("Tensors must have the same shape.")

    batched_tensors = torch.cat(tensors, dim=0)
    return Cameras(batched_tensors)

if __name__ == "__main__":
    tensors1 = torch.randn(4, 2, 2)
    tensors2 = torch.randn(4, 2, 2)

    cameras1 = Cameras(tensors1)
    cameras2 = Cameras(tensors2)

    batched_cameras = join_cameras_as_batch([cameras1, cameras2])

    print(batched_cameras.tensors.shape)