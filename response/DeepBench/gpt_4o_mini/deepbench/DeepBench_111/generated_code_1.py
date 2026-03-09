import torch
import torch.nn.functional as F

def binary_focal_loss_with_logits(pred, target, alpha=0.25, gamma=2.0, reduction='mean', pos_weight=None):
    if pos_weight is not None:
        pred = pred * pos_weight
    
    pred_sigmoid = torch.sigmoid(pred)
    p_t = pred_sigmoid * target + (1 - pred_sigmoid) * (1 - target)
    
    focal_loss = -alpha * (1 - p_t) ** gamma * torch.log(p_t + 1e-6)  # Add epsilon to prevent log(0)
    
    if reduction == 'mean':
        return focal_loss.mean()
    elif reduction == 'sum':
        return focal_loss.sum()
    else:
        return focal_loss

if __name__ == "__main__":
    pred = torch.tensor([[0.5, 0.2], [0.9, 0.1]], dtype=torch.float32)  # Sample logits
    target = torch.tensor([[1.0, 0.0], [1.0, 0.0]], dtype=torch.float32)  # Sample targets
    alpha = 0.25
    gamma = 2.0
    loss = binary_focal_loss_with_logits(pred, target, alpha=alpha, gamma=gamma, reduction='mean')
    print(f"Computed Binary Focal Loss: {loss.item()}")