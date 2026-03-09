import torch

def look_at_rotation(camera_position, at, up, device):
  camera_position = torch.tensor(camera_position, device=device)
  at = torch.tensor(at, device=device)
  up = torch.tensor(up, device=device)
  
  camera_position = camera_position / torch.norm(camera_position, dim=-1, keepdim=True)
  at = at / torch.norm(at, dim=-1, keepdim=True)
  up = up / torch.norm(up, dim=-1, keepdim=True)

  z_axis = camera_position - at
  x_axis = torch.cross(up, z_axis, dim=-1)
  x_axis = x_axis / torch.norm(x_axis, dim=-1, keepdim=True)

  y_axis = torch.cross(z_axis, x_axis, dim=-1)
  
  R = torch.stack((x_axis, y_axis, z_axis), dim=-1)
  return R.transpose(-1, -2)

if __name__ == "__main__":
  camera_position = [0, 0, 1]
  at = [0, 0, 0]
  up = [0, 1, 0]
  device = "cpu" 
  R = look_at_rotation(camera_position, at, up, device)
  print(R)