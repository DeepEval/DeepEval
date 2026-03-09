import torch

def look_at_rotation(camera_position, at, up, device_type):
    # Normalize the input vectors
    camera_position = camera_position / torch.norm(camera_position, dim=1, keepdim=True)
    at = at / torch.norm(at, dim=1, keepdim=True)
    up = up / torch.norm(up, dim=1, keepdim=True)

    # Calculate the x, y, and z axes
    x = torch.cross(up, at)
    y = torch.cross(at, x)
    z = at

    # Handle cases where the x-axis is close to zero
    if torch.isclose(x, torch.zeros_like(x)).all():
        x = torch.tensor([[1, 0, 0]], device=device_type)

    # Concatenate the axes and transpose to form the rotation matrix
    R = torch.cat((x, y, z), dim=1).T

    return R

if __name__ == "__main__":
    # Create sample input values
    camera_position = torch.tensor([[1, 2, 3]], device="cuda")
    at = torch.tensor([[4, 5, 6]], device="cuda")
    up = torch.tensor([[7, 8, 9]], device="cuda")
    device_type = "cuda"

    # Call the function and print the results
    R = look_at_rotation(camera_position, at, up, device_type)
    print(R)