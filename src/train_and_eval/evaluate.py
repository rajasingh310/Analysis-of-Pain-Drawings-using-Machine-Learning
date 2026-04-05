import torch
import torch.nn as nn

def evaluate_model(model, val_loader, device, return_preds=False):
    model.eval()

    correct = 0
    total = 0
    total_loss = 0

    criterion = nn.CrossEntropyLoss()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)

            outputs = model(x)
            loss = criterion(outputs, y)

            total_loss += loss.item()

            _, predicted = torch.max(outputs, 1)

            total += y.size(0)
            correct += (predicted == y).sum().item()

            if return_preds:
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(y.cpu().numpy())

    acc = (correct / total) * 100
    avg_loss = total_loss / len(val_loader)

    if return_preds:
        return acc, avg_loss, all_preds, all_labels

    return acc, avg_loss