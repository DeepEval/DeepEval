import torch

def change_basis_real_to_complex(k, dtype=None, device=None):
  if dtype is None:
    dtype = torch.float32
  if device is None:
    device = torch.device('cpu')
  
  n = 2*k + 1
  Q = torch.zeros((n, n), dtype=dtype, device=device)
  for l in range(k+1):
    for m in range(-l, l+1):
      Q[l*2 + m, l*2 + m] = 1
      Q[l*2 + m + 1, l*2 + m + 1] = 1
      
  return Q

if __name__ == "__main__":
  k = 2
  Q = change_basis_real_to_complex(k)
  print(Q)