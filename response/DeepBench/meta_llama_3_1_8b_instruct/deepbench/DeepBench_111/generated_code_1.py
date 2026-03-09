import torch
import torch.nn as nn

def binary_focal_loss_with_logits(pred, target, alpha, gamma, reduction='mean',
  pos_weight=None):
    pred = torch.clamp(pred, 0, 1)
    logpt = -pred * torch.log(1 - pred) - (1 - pred) * torch.log(pred)
    logpt *= target
    logpt *= (1 - target) * alpha + target * (1 - alpha)
    logpt *= torch.pow((1 - pred), gamma)

    loss = logpt.sum() if reduction =='sum' else logpt.mean() if reduction =='mean' else logpt

    if pos_weight is not None:
        loss *= pos_weight

    return loss

if __name__ == "__main__":
    # Create sample input values
    pred = torch.tensor([[[0.5, 0.5], [0.8, 0.2]], [[0.2, 0.8], [0.1, 0.9]]])
    target = torch.tensor([[[1, 0], [0, 1]], [[0, 1], [1, 0]]])
    alpha = 0.5
    gamma = 2
    reduction ='mean'
    pos_weight = 1

    # Call the function
    loss = binary_focal_loss_with_logits(pred, target, alpha, gamma, reduction,
  pos_weight)

    # Print the results
    print(loss)