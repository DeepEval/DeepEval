import torch
import torch.nn.functional as F

def binary_focal_loss_with_logits(pred, target, alpha=0.25, gamma=2.0, reduction='mean', pos_weight=None):
    bce_loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none', pos_weight=pos_weight)
    p_t = torch.exp(-bce_loss)
    focal_loss = alpha * (1 - p_t) ** gamma * bce_loss

    if reduction == 'mean':
        return focal_loss.mean()
    elif reduction == 'sum':
        return focal_loss.sum()
    else:  # 'none'
        return focal_loss

if __name__ == "__main__":
    # Create sample input values
    pred = torch.tensor([[0.2, -1.5], [0.7, 2.0]], requires_grad=True)
    target = torch.tensor([[0.0, 1.0], [1.0, 0.0]])

    # Call the function
    loss = binary_focal_loss_with_logits(pred, target, alpha=0.25, gamma=2.0, reduction='mean')

    # Print the results
    print("Focal Loss:", loss.item())