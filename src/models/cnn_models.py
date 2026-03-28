import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    """Basic 2-layer CNN"""
    def __init__(self, num_classes, input_size=(128, 128), input_channels=3):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        # Compute fc1 input size dynamically
        with torch.no_grad():
            dummy = torch.zeros(1, input_channels, *input_size)
            x = self.pool(F.relu(self.conv1(dummy)))
            x = self.pool(F.relu(self.conv2(x)))
            self.flattened_size = x.numel()

        self.fc1 = nn.Linear(self.flattened_size, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class SimpleCNNSmall(nn.Module):
    """Smaller CNN for lightweight experiments"""
    def __init__(self, num_classes, input_size=(128, 128), input_channels=3):
        super(SimpleCNNSmall, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 8, 7, padding=1)
        self.conv2 = nn.Conv2d(8, 16, 7, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.3)

        with torch.no_grad():
            dummy = torch.zeros(1, input_channels, *input_size)
            x = self.pool(F.relu(self.conv1(dummy)))
            x = self.pool(F.relu(self.conv2(x)))
            self.flattened_size = x.numel()

        self.fc1 = nn.Linear(self.flattened_size, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x