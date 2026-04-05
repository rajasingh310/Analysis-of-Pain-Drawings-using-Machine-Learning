import torch
import torch.nn as nn  # ✅ REQUIRED
import torch.optim as optim
import sys
from pathlib import Path

# --------------------------
# Project path
# --------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

# Import evaluate function from separate file
from src.train_and_eval.evaluate import evaluate_model  # make sure this path is correct
def train_model(model, train_loader, val_loader, config, device):

    epochs = config["training"]["epochs"]
    lr = config["training"]["learning_rate"]

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    print("\n🚀 Starting Training...\n")

    # 🔥 STORE METRICS
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": []
    }

    for epoch in range(epochs):
        model.train()

        total_loss = 0
        correct = 0
        total = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            outputs = model(x)

            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()

        train_loss = total_loss / len(train_loader)
        train_acc = (correct / total) * 100

        # 🔥 VALIDATION
        val_acc, val_loss = evaluate_model(model, val_loader, device)

        # STORE
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"Epoch [{epoch+1}/{epochs}] "
              f"| Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% "
              f"| Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

    return model, history