import torch

def _get_perspective_coeffs(startpoints, endpoints):
  if len(startpoints) != 4 or len(endpoints) != 4:
    raise ValueError("Both inputs must have exactly four coordinate pairs.")
  startpoints = torch.tensor(startpoints, dtype=torch.double)
  endpoints = torch.tensor(endpoints, dtype=torch.double)
  A = torch.hstack((startpoints[:, 0].reshape(-1, 1), startpoints[:, 1].reshape(-1, 1), torch.ones_like(startpoints[:, 0]), torch.zeros_like(startpoints[:, 0])))
  B = endpoints[:, 0].reshape(-1, 1)
  C = torch.hstack((startpoints[:, 0].reshape(-1, 1), startpoints[:, 1].reshape(-1, 1), torch.zeros_like(startpoints[:, 0]), torch.ones_like(startpoints[:, 0])))
  D = endpoints[:, 1].reshape(-1, 1)
  coeffs = torch.linalg.solve(torch.cat((A, C), dim=1), torch.cat((B, D), dim=1))
  return coeffs.float()

if __name__ == "__main__":
  startpoints = [(0, 0), (1, 0), (0, 1), (1, 1)]
  endpoints = [(0.5, 0.2), (0.8, 0.1), (0.2, 0.9), (0.9, 0.8)]
  coeffs = _get_perspective_coeffs(startpoints, endpoints)
  print(coeffs)