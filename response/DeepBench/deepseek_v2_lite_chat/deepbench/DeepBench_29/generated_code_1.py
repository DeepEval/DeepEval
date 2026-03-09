import torch

def quaternion_to_axis_angle(quaternion):
    # Check if input is a tensor with the correct shape
    if isinstance(quaternion, torch.Tensor):
        if quaternion.ndim == 4:
            batch_size = quaternion.shape[0]
            quaternion_list = quaternion.unbind(dim=1)
        elif quaternion.ndim == 3:
            batch_size = quaternion.shape[0]
            quaternion_list = quaternion.unbind(dim=0)
        else:
            raise ValueError("Input must be a 3D or 4D tensor with shape Nx4 or 4.")

        # Unpack quaternions and compute axis angle
        axis_angle_list = []
        for quaternion in quaternion_list:
            w, x, y, z = quaternion
            norm = torch.sqrt(w * w + x * x + y * y + z * z)
            if norm == 0:
                axis_angle_list.append(torch.tensor([0, 0, 0]))
            else:
                axis_angle = torch.tensor([2 * w * norm, 2 * x * norm, 2 * y * norm, 2 * z * norm])
                axis_angle /= norm
                axis_angle = axis_angle / norm
                axis_angle = torch.nan_to_num(axis_angle, nan=0)
                axis_angle_list.append(axis_angle)

        # Reshape and return the output tensor
        if axis_angle_list[0].ndim == 1:
            output = torch.stack(axis_angle_list)
        else:
            output = torch.cat(axis_angle_list, dim=1)
        return output
    else:
        raise ValueError("Input must be a tensor.")

if __name__ == "__main__":
    import random
    random_quaternion = torch.rand(5, 4)
    print("Input quaternion:\n", random_quaternion)
    axis_angle = quaternion_to_axis_angle(random_quaternion)
    print("Output axis angle:\n", axis_angle)