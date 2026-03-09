import torch
from torch import nn

def binary_focal_loss_with_logits(pred, target, alpha=None, gamma=2.0, reduction='none'):
    """
    Computes Binary Focal Loss.
    Args:
        pred: Logits from model (B, C, *) where C = number of classes.
        target: Labels of the respective classes (B, C, *) where each value is between 0 and 1.
        alpha: Weighting factor for each class (C,), alpha[i] is the weight for class i.
        gamma: Focusing parameter (default = 2.0).
        reduction: Specifies the reduction to apply to the output:
            - 'none': no reduction will be applied,
            - 'mean': the sum of the output will be divided by the number of elements in the output,
            - 'sum': the output will be summed.
    Returns:
        Loss tensor according to the formula.
    """
    # Calculate the probability estimates
    pred_prob = torch.sigmoid(pred)

    # Create a tensor for the positive cases only
    if target.dtype == torch.float32:
        positive_mask = target > 0.5
    else:
        positive_mask = target == 1

    # Compute the actual focal loss
    if alpha is not None:
        # Adjust alpha and beta for each class
        alpha = alpha[target]
        pred_prob = pred_prob * positive_mask + alpha[None, :] * (1 - positive_mask)

    # Compute the actual focal loss
    pt = (1 - pred_prob) ** gamma * positive_mask + pred_prob ** gamma * (1 - positive_mask)
    fl = -torch.where(target == 1, alpha * (torch.log(pt)) ** gamma, torch.log(1 - pt))

    if reduction == 'mean':
        return fl.mean()
    elif reduction == 'sum':
        return fl.sum()
    else:
        return fl

if __name__ == "__main__":
    # Example usage
    import numpy as np

    # Sample input values
    pred = torch.randn(3, 2)  # (B, C)
    target = torch.tensor([0, 1, 0], dtype=torch.float32)  # (B)
    alpha = torch.tensor([0.5, 0.5])  # (C)
    gamma = 3.0
    reduction = 'mean'

    # Call the function and print the results
    loss = binary_focal_loss_with_logits(pred, target, alpha, gamma, reduction)
    print("Binary Focal Loss:", loss.item())