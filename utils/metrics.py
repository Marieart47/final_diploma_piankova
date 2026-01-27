import torch

def accuracy(preds, targets):
    preds = torch.argmax(preds, dim=1)
    return (preds == targets).float().mean().item()

def iou(preds, targets, num_classes):
    preds = torch.argmax(preds, dim=1)
    ious = []

    for cls in range(num_classes):
        p = preds == cls
        t = targets == cls
        inter = (p & t).sum().item()
        union = (p | t).sum().item()
        if union == 0:
            ious.append(1.0)
        else:
            ious.append(inter / union)
    return sum(ious) / len(ious)
