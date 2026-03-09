import torch

def quaternion_to_axis_angle(quaternion):
    if not isinstance(quaternion, torch.Tensor) or quaternion.shape not in [(4,), (4, 1)]:
        raise TypeError("Input quaternion must be a tensor of shape Nx4 or 4")

    quaternion = quaternion.view(4, -1)
    w, x, y, z = quaternion

    sqw = w * w
    sqx = x * x
    sqy = y * y
    sqz = z * z

    half_angle = torch.acos(2 * (sqw + sqx + sqy + sqz) - 1)

    axis_angle = torch.zeros(half_angle.shape, dtype=torch.float32)

    mask_w = sqw + sqx >= sqy + sqz
    axis_angle[mask_w] = torch.atan2(2 * (w * x + y * z), 1 - 2 * (sqx + sqw))
    mask_x = sqx + sqy >= sqw + sqz
    axis_angle[mask_x] = torch.atan2(2 * (w * y - z * x), 1 - 2 * (sqy + sqw))
    mask_y = sqy + sqz >= sqw + sqx
    axis_angle[mask_y] = torch.atan2(2 * (w * z + x * y), 1 - 2 * (sqz + sqw))

    return axis_angle

if __name__ == "__main__":
    quaternion = torch.tensor([0.7071068, 0, 0, 0.7071068])
    axis_angle = quaternion_to_axis_angle(quaternion)
    print(axis_angle)