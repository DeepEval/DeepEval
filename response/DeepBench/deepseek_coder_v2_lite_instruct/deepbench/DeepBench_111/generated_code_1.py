import torch
import torch.nn as nn
import torch.nn.functional as F

def binary_focal_loss_with_logits(pred, target, alpha, gamma, reduction='none'):
    bce_loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
    pt = torch.exp(-bce_loss)
    focal_loss = alpha * (1 - pt) ** gamma * bce_loss
    
    if reduction == 'mean':
        return torch.mean(focal_loss)
    elif reduction == 'sum':
        return torch.sum(focal_loss)
    else:
        return focal_loss

if __name__ == "__main__":
    # Create sample input values
    pred = torch.randn(3, 2, requires_grad=True)  # Predicted logits
    target = torch.empty(3, 2).random_(2)  # Target labels (0 or 1)
    alpha = 0.5
    gamma = 2.0
    
    # Call the function
    loss = binary_focal_loss_with_logits(pred, target, alpha, gamma, reduction='mean')
    
    # Print the results
    print(f"Binary Focal Loss: {loss.item()}")