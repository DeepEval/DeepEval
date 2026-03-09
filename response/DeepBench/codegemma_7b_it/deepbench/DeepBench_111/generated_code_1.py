import torch
import torch.nn as nn

def binary_focal_loss_with_logits(pred, target, alpha, gamma, reduction="none", pos_weight=None):
    """
    Compute Binary Focal loss.
    """
    pt = torch.sigmoid(pred)
    ce_loss = nn.functional.binary_cross_entropy_with_logits(pred, target, reduction="none", pos_weight=pos_weight)
    loss = ce_loss * (1 - pt) ** gamma * target + ce_loss * pt ** gamma * (1 - target)

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()

    return loss

if __name__ == "__main__":
    # Create sample input values
    pred = torch.tensor([0.5, 0.2, 0.7, 0.8])
    target = torch.tensor([1, 0, 1, 0])

    # Call the function
    alpha = 0.5
    gamma = 2
    reduction = "mean"
    pos_weight = None
    loss = binary_focal_loss_with_logits(pred, target, alpha, gamma, reduction, pos_weight)

    # Print the results
    print(loss)