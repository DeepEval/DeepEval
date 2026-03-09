import torch
import torch.nn as nn

def binary_focal_loss_with_logits(pred, target, alpha=0.25, gamma=2.0, reduction='mean', pos_weight=None):
  log_p = torch.nn.functional.log_softmax(pred, dim=1)
  p = torch.exp(log_p)
  pt = p * target + (1 - p) * (1 - target)
  focal_loss = -alpha * (1 - pt)**gamma * log_p
  if pos_weight is not None:
    focal_loss = focal_loss * pos_weight
  if reduction == 'none':
    return focal_loss
  elif reduction == 'mean':
    return torch.mean(focal_loss)
  elif reduction == 'sum':
    return torch.sum(focal_loss)

if __name__ == "__main__":
    pred = torch.randn(10, 2, 3)
    target = torch.rand(10, 2, 3)
    loss = binary_focal_loss_with_logits(pred, target, alpha=0.5, gamma=1.0)
    print(loss)