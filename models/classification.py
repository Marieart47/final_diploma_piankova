import torchvision.models as models
import torch.nn as nn

def resnet18(num_classes):
    m = models.resnet18(pretrained=True)
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m

def efficientnet_b0(num_classes):
    m = models.efficientnet_b0(pretrained=True)
    m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
    return m
