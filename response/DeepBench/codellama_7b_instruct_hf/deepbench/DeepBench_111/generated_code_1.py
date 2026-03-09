import torch
import torch.nn.functional as F

def binary_focal_loss_with_logits(pred, target, alpha=0.25, gamma=2, reduction='none', pos_weight=None):
    # Compute the logits
    logits = pred

    # Compute the probability of each class
    probabilities = F.sigmoid(logits)

    # Compute the focal loss
    focal_loss = -alpha * (1 - probabilities)**gamma * torch.log(probabilities)

    # Apply the weighting factor
    weighted_focal_loss = focal_loss * pos_weight

    # Reduce the loss
    if reduction == 'none':
        return weighted_focal_loss
    elif reduction == 'mean':
        return weighted_focal_loss.mean()
    elif reduction == 'sum':
        return weighted_focal_loss.sum()
    else:
        raise ValueError(f'Invalid reduction mode: {reduction}')

if __name__ == "__main__":
    # Create sample input values
    pred = torch.randn(3, 2)
    target = torch.randint(0, 2, (3, 2))
    alpha = 0.25
    gamma = 2
    reduction = 'none'
    pos_weight = None

    # Call the function
    output = binary_focal_loss_with_logits(pred, target, alpha, gamma, reduction, pos_weight)

    # Print the output
    print(output)