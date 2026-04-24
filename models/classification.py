import torch.nn as nn
import torchvision.models as models


def resnet18(num_classes: int = 10) -> nn.Module:
    m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m


def swin_t(num_classes: int = 10) -> nn.Module:
    m = models.swin_t(weights=models.Swin_T_Weights.IMAGENET1K_V1)
    m.head = nn.Linear(m.head.in_features, num_classes)
    return m
