# ==========================================
# run_pipeline.py
# Complete pipeline: preprocess → load → model → training → evaluation + plotting
# ==========================================

import sys
import yaml
from pathlib import Path
import torch

# --------------------------
# Step 1: Define project root
# --------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

# --------------------------
# Step 2: Imports from src
# --------------------------
from src.utils.data_preprocess import process_dataset
from src.utils.load_processed_dataset import get_dataloaders

# Models
from src.models.cnn_models import SimpleCNN, SimpleCNNSmall
from src.models.fcnn_models import FCNNSmall, FCNNNet

# Training & Evaluation
from src.train_and_eval.train import train_model
from src.train_and_eval.evaluate import evaluate_model

# 🔥 NEW: Plotting
from src.utils.plot_results import plot_training_results


# ===========================
# Step 3: Helper function to pick device
# ===========================
def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ===========================
# Step 4: Main pipeline
# ===========================
if __name__ == "__main__":

    # --------------------------
    # Step 4.1: Load configuration file
    # --------------------------
    config_path = PROJECT_ROOT / "experiments/configs/aachen.yaml"
    print("Config path:", config_path)

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # --------------------------
    # Step 4.2: Preprocessing
    # --------------------------
    use_processed_tensors = config["data"].get("use_processed_tensors", False)

    if use_processed_tensors:
        print("\n⚠️ Using preprocessed tensors. Skipping preprocessing.")
    else:
        print("\n🔄 Running preprocessing pipeline...")
        process_dataset(config)

    # --------------------------
    # Step 4.3: Load dataset
    # --------------------------
    print("\n📦 Loading dataset...")
    train_loader, val_loader, num_classes = get_dataloaders(config)

    train_size = len(train_loader.dataset)
    val_size = len(val_loader.dataset)

    print(f"✅ Train samples: {train_size}")
    print(f"✅ Validation samples: {val_size}")
    print(f"✅ Number of classes: {num_classes}")

    # --------------------------
    # Step 4.4: Detect input shape
    # --------------------------
    x_sample, _ = next(iter(train_loader))
    input_channels = x_sample.shape[1]
    input_size = (x_sample.shape[2], x_sample.shape[3])

    print(f"Detected input channels: {input_channels}")
    print(f"Detected input size: {input_size}")

    # --------------------------
    # Step 4.5: Initialize model
    # --------------------------
    model_type = config["model"].get("type", "cnn")
    model_name = config["model"].get("name", "")

    print(f"\n⚙️ Model type: {model_type}, name: {model_name}")

    if model_type == "cnn":
        if model_name == "SimpleCNNSmall":
            model = SimpleCNNSmall(
                num_classes=num_classes,
                input_size=input_size,
                input_channels=input_channels
            )
        else:
            # 🔥 Pass input_size here too
            model = SimpleCNN(
                num_classes=num_classes,
                input_size=input_size,
                input_channels=input_channels
            )

    elif model_type == "fcnn":
        if model_name == "FCNNSmall":
            model = FCNNSmall(
                num_classes=num_classes,
                input_size=input_size,
                input_channels=input_channels
            )
        else:
            model = FCNNNet(
                input_size=input_size[0],
                input_channels=input_channels,
                num_classes=num_classes
            )

    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # --------------------------
    # Step 4.6: Move to device
    # --------------------------
    device = get_device()
    model = model.to(device)

    print(f"\n✅ Model moved to device: {device}")
    print(model)

    # --------------------------
    # Step 4.7: Sanity check
    # --------------------------
    with torch.no_grad():
        x_sample = x_sample.to(device)
        out = model(x_sample)
        print(f"\n✅ Forward pass OK, output shape: {out.shape}")

    # --------------------------
    # Step 4.8: Training
    # --------------------------
    print("\n🏋️ Training started...")
    trained_model, history = train_model(
        model,
        train_loader,
        val_loader,
        config,
        device
    )

    # --------------------------
    # Step 4.9: Save model
    # --------------------------
    model_save_path = PROJECT_ROOT / "experiments" / "saved_models" / f"{model_type}_{model_name}.pt"
    model_save_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(trained_model.state_dict(), model_save_path)
    print(f"✅ Model saved at: {model_save_path}")

    # --------------------------
    # Step 4.10: Final evaluation
    # --------------------------
    print("\n🔍 Final evaluation...")
    final_acc, final_loss = evaluate_model(trained_model, val_loader, device)

    print(f"🎯 Final Validation Accuracy: {final_acc:.2f}%")
    print(f"🎯 Final Validation Loss: {final_loss:.4f}")

    # --------------------------
    # Step 4.11: Plot results (SVG)
    # --------------------------
    # 🔥 GET predictions for confusion matrix
    final_acc, final_loss, preds, labels = evaluate_model(
        trained_model,
        val_loader,
        device,
        return_preds=True
    )

    # 🔥 SAVE ALL PLOTS
    plot_training_results(
        history=history,
        config=config,
        train_size=train_size,
        val_size=val_size,
        preds=preds,
        labels=labels,
        num_classes=num_classes,
        save_dir=PROJECT_ROOT / "results"
    )

    print("\n🎉 Pipeline completed successfully!")