import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from sklearn.metrics import confusion_matrix


def plot_training_results(history, config, train_size, val_size,
                          preds, labels, num_classes, save_dir):

    dataset_name = config["data"]["dataset_name"]
    model_name = config["model"]["name"]

    base_name = f"{dataset_name}_num_classes_{num_classes}_{model_name}"

    Path(save_dir).mkdir(parents=True, exist_ok=True)

    epochs = range(1, len(history["train_loss"]) + 1)

    # --------------------------
    # 🔥 COMMON TEXT (TOP LEFT)
    # --------------------------
    hyper_text = (
        f"Epochs: {config['training']['epochs']}\n"
        f"LR: {config['training']['learning_rate']}\n"
        f"Batch: {config['data']['batch_size']}\n"
        f"Split: {config['data']['train_split']}\n"
        f"Train: {train_size}\n"
        f"Val: {val_size}\n"
        f"Model: {config['model']['name']}\n"
    )

    # ==========================
    # 1️⃣ LOSS CURVE
    # ==========================
    plt.figure()

    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["val_loss"], label="Val Loss")

    plt.xlabel("Epochs")
    plt.ylabel("Loss")

    plt.legend(loc="upper right")
    plt.text(0.02, 0.98, hyper_text,
             transform=plt.gca().transAxes,
             verticalalignment='top')

    loss_path = Path(save_dir) / f"{base_name}_loss.svg"
    plt.savefig(loss_path, format="svg", bbox_inches="tight")
    plt.close()

    print(f"✅ Loss plot saved: {loss_path}")

    # ==========================
    # 2️⃣ ACCURACY CURVE
    # ==========================
    plt.figure()

    plt.plot(epochs, history["train_acc"], label="Train Acc")
    plt.plot(epochs, history["val_acc"], label="Val Acc")

    plt.xlabel("Epochs")
    plt.ylabel("Accuracy (%)")

    plt.legend(loc="upper right")
    plt.text(0.02, 0.98, hyper_text,
             transform=plt.gca().transAxes,
             verticalalignment='top')

    acc_path = Path(save_dir) / f"{base_name}_accuracy.svg"
    plt.savefig(acc_path, format="svg", bbox_inches="tight")
    plt.close()

    print(f"✅ Accuracy plot saved: {acc_path}")

    # ==========================
    # 3️⃣ CONFUSION MATRIX (PERCENT, GRAYSCALE)
    # ==========================
    cm = confusion_matrix(labels, preds)
    cm_percent = cm.astype(float) / cm.sum(axis=1, keepdims=True)  # normalize row-wise

    plt.figure()
    im = plt.imshow(cm_percent, cmap="Greys")  # grayscale
    plt.colorbar(im)

    # numeric ticks: 1,2,...,num_classes
    ticks = np.arange(num_classes)
    plt.xticks(ticks, ticks + 1)
    plt.yticks(ticks, ticks + 1)

    # labels inside boxes
    for i in range(cm_percent.shape[0]):
        for j in range(cm_percent.shape[1]):
            value = cm_percent[i, j]
            plt.text(j, i, f"{value:.2f}", ha="center", va="center")

    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()

    cm_path = Path(save_dir) / f"{base_name}_confusion_matrix.svg"
    plt.savefig(cm_path, format="svg", bbox_inches="tight")
    plt.close()
    print(f"✅ Confusion matrix saved: {cm_path}")