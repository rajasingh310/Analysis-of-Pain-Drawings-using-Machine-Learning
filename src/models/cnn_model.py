import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()

        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)

        self.fc1 = nn.Linear(32 * 32 * 32, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # (128 → 64)
        x = self.pool(F.relu(self.conv2(x)))  # (64 → 32)

        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x

import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNNSmall(nn.Module):
    def __init__(self, num_classes, input_size=(128, 128)):
        super(SimpleCNNSmall, self).__init__()

        # Conv layers
        self.conv1 = nn.Conv2d(3, 8, kernel_size=7, padding=1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=7, padding=1)

        # Max pooling
        self.pool = nn.MaxPool2d(2, 2)

        # Dropout
        self.dropout = nn.Dropout(0.3)

        # Compute fc1 input dynamically
        with torch.no_grad():
            dummy = torch.zeros(1, 3, *input_size)
            x = self.pool(F.relu(self.conv1(dummy)))
            x = self.pool(F.relu(self.conv2(x)))
            self.flattened_size = x.numel()  # total features after conv+pool

        # Fully connected layers
        self.fc1 = nn.Linear(self.flattened_size, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN2DNet(nn.Module):
    def __init__(self, num_classes=3, input_channels=3, input_size=128):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, 8, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(16)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.3)

        # compute flattened size
        dummy = torch.zeros(1, input_channels, input_size, input_size)
        dummy = self._forward_conv(dummy)
        self.flat_size = dummy.view(1, -1).size(1)

        # Fully connected layers
        self.fc1 = nn.Linear(self.flat_size, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def _forward_conv(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.dropout(x)
        return x

    def forward(self, x):
        x = self._forward_conv(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x