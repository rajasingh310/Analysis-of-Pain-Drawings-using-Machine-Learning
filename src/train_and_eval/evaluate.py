# ==========================================
# evaluate.py
# Evaluation script
# ==========================================

import torch

def evaluate_model(model, val_loader, device):
    """
    Evaluate the model on validation set.

    Args:
        model: PyTorch model
        val_loader: validation DataLoader
        device: torch device (cpu / cuda)
    Returns:
        val_accuracy: float
    """

    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            _, predicted = torch.max(outputs, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()

    val_accuracy = correct / total
    return val_accuracy * 100