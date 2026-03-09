import torch

def _safe_det_3x3(t):
  a = t[:, 0, 0] * (t[:, 1, 1] * t[:, 2, 2] - t[:, 1, 2] * t[:, 2, 1])
  b = t[:, 0, 1] * (t[:, 1, 0] * t[:, 2, 2] - t[:, 1, 2] * t[:, 2, 0])
  c = t[:, 0, 2] * (t[:, 1, 0] * t[:, 2, 1] - t[:, 1, 1] * t[:, 2, 0])
  return a - b + c

if __name__ == "__main__":
  t = torch.tensor([[[1, 2, 3],
                    [4, 5, 6],
                    [7, 8, 9]],

                   [[9, 8, 7],
                    [6, 5, 4],
                    [3, 2, 1]]], dtype=torch.float32)
  result = _safe_det_3x3(t)
  print(result)