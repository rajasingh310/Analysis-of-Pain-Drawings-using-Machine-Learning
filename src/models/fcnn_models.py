import torch
import torch.nn as nn
import torch.nn.functional as F


class FCNNSmall(nn.Module):
    """Simple fully connected network for small datasets"""
    def __init__(self, num_classes, input_size=(128,128), input_channels=3):
        super(FCNNSmall, self).__init__()
        # Use actual input size
        self.input_dim = input_channels * input_size[0] * input_size[1]
        self.fc1 = nn.Linear(self.input_dim, 128)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class FCNNNet(nn.Module):
    """Deeper fully connected network"""
    def __init__(self, num_classes, input_size=128, input_channels=3):
        super(FCNNNet, self).__init__()
        self.flattened_size = input_size * input_size * input_channels

        self.fc1 = nn.Linear(self.flattened_size, 200)
        self.bn1 = nn.BatchNorm1d(200)

        self.fc2 = nn.Linear(200, 250)
        self.bn2 = nn.BatchNorm1d(250)

        self.fc3 = nn.Linear(250, 300)
        self.bn3 = nn.BatchNorm1d(300)

        self.fc4 = nn.Linear(300, 250)
        self.bn4 = nn.BatchNorm1d(250)
        self.dropout = nn.Dropout(0.3)

        self.fc5 = nn.Linear(250, 200)
        self.bn5 = nn.BatchNorm1d(200)

        self.fc6 = nn.Linear(200, 100)
        self.bn6 = nn.BatchNorm1d(100)

        self.fc7 = nn.Linear(100, 50)
        self.bn7 = nn.BatchNorm1d(50)

        self.fc8 = nn.Linear(50, 25)
        self.bn8 = nn.BatchNorm1d(25)

        self.fc9 = nn.Linear(25, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = F.relu(self.bn3(self.fc3(x)))
        x = self.dropout(F.relu(self.bn4(self.fc4(x))))
        x = F.relu(self.bn5(self.fc5(x)))
        x = F.relu(self.bn6(self.fc6(x)))
        x = F.relu(self.bn7(self.fc7(x)))
        x = F.relu(self.bn8(self.fc8(x)))
        x = self.fc9(x)
        return x