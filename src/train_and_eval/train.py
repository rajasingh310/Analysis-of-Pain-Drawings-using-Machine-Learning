# ==========================================
# train.py
# Training script
# ==========================================

import torch
import sys
import torch.nn as nn
import torch.optim as optim
from pathlib import Path

# --------------------------
# Project path
# --------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

# Import evaluate function from separate file
from src.train_and_eval.evaluate import evaluate_model  # make sure this path is correct


def train_model(model, train_loader, val_loader, config, device):
    """
    Trains the model using the given data loaders and configuration.

    Args:
        model: PyTorch model
        train_loader: training DataLoader
        val_loader: validation DataLoader
        config: configuration dictionary
        device: torch device (cpu / cuda)
    Returns:
        trained model
    """

    epochs = config["training"]["epochs"]
    lr = config["training"]["learning_rate"]

    model = model.to(device)

    # Loss function
    criterion = nn.CrossEntropyLoss()

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)

    print("\n🚀 Starting Training...\n")

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        # Validation
        val_acc = evaluate_model(model, val_loader, device)

        print(f"Epoch [{epoch+1}/{epochs}] | Loss: {avg_loss:.4f} | Val Acc: {val_acc:.4f}")

    return model