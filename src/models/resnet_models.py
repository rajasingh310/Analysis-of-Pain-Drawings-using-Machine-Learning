import torch
import torch.nn as nn
import torchvision.models as models


def get_resnet18(num_classes, pretrained=False):
    """Return a ResNet18 model"""
    model = models.resnet18(pretrained=pretrained)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def get_resnet34(num_classes, pretrained=False):
    model = models.resnet34(pretrained=pretrained)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model