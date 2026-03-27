import sys

from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
from pathlib import Path
import yaml
import torch
import torch.nn as nn
import torch.optim as optim

from src.utils.preprocess import get_transforms
from src.utils.data_loader import get_dataloaders
from src.models.cnn_model import CNN2DNet
from src.models.cnn_model import SimpleCNN, SimpleCNNSmall
from src.models.fcnn_model import FCNNSmall, FCNNNet

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    accuracy = correct / total
    return total_loss / len(loader), accuracy


def validate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total
    return total_loss / len(loader), accuracy


def main():
    # Load config
    config_path = "configs/aachen.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    project_root = Path(__file__).resolve().parent.parent
    data_path = project_root / config["data"]["base_dir"] / config["data"]["dataset"]
    data_path = data_path.resolve()
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset path not found: {data_path}")
    print("Dataset path:", data_path)

    transform = get_transforms(config["data"]["img_size"])

    train_loader, val_loader, num_classes, train_dataset = get_dataloaders(
        data_path=str(data_path),
        transform=transform,
        batch_size=config["data"]["batch_size"],
        train_split=config["data"]["train_split"],
        exclude_dirs=config["data"].get("exclude_dirs", []),
        undersample=config["data"].get("undersample", False),
        weighted_sampler=config["data"].get("weighted_sampler", False)
    )

    # Choose model
    model_name = config["model"]["type"]
    if model_name == "CNN2DNet":
        model = CNN2DNet(
            input_channels=3,
            num_classes=num_classes,
            input_size=config["data"]["img_size"]
        ).to(device)
    elif model_name == "SimpleCNN":
        model = SimpleCNN(num_classes).to(device)
    elif model_name == "SimpleCNNSmall":
        model = SimpleCNNSmall(num_classes).to(device)
    elif model_name == "FCNNSmall":
        model = FCNNSmall(num_classes).to(device)
    elif model_name == "FCNNNet":
        model = FCNNNet(num_classes).to(device)
    else:
        raise ValueError(f"Unknown model type: {model_name}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config["training"]["learning_rate"])

    for epoch in range(config["training"]["epochs"]):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        print(f"Epoch {epoch+1}/{config['training']['epochs']}: "
              f"Train Loss={train_loss:.4f}, Train Acc={train_acc*100:.2f}%, "
              f"Val Loss={val_loss:.4f}, Val Acc={val_acc*100:.2f}%")

    print("Training complete!")


if __name__ == "__main__":
    main()



