# ==========================================
# run_pipeline.py
# Complete pipeline: preprocess → load → model → training → evaluation
# ==========================================

import sys
import yaml
from pathlib import Path
import torch

# --------------------------
# Step 1: Define project root
# --------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))  # Ensure src modules are importable

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
from src.train_and_eval.evaluate import evaluate_model  # your evaluate.py script

# ===========================
# Step 3: Helper function to pick device
# ===========================
def get_device():
    """
    Returns torch.device('cuda') if GPU is available, else CPU.
    """
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
    # Step 4.2: Check if preprocessed tensors should be used
    # --------------------------
    use_processed_tensors = config["data"].get("use_processed_tensors", False)

    if use_processed_tensors:
        print("\n⚠️ Using preprocessed tensors. Skipping processing pipeline.")
    else:
        print("\n🔄 Running preprocessing pipeline...")
        process_dataset(config)

    # --------------------------
    # Step 4.3: Load dataset
    # --------------------------
    print("\n📦 Loading dataset for training...")
    train_loader, val_loader, num_classes = get_dataloaders(config)
    print(f"✅ Data loaded: {len(train_loader.dataset)} train, {len(val_loader.dataset)} val")
    print(f"Number of classes: {num_classes}")

    # --------------------------
    # Step 4.4: Detect input channels dynamically
    # --------------------------
    x_sample, _ = next(iter(train_loader))
    input_channels = x_sample.shape[1]  # 1 for grayscale, 3 for RGB
    input_size = (x_sample.shape[2], x_sample.shape[3])
    print(f"Detected input channels: {input_channels}, input size: {input_size}")

    # --------------------------
    # Step 4.5: Initialize model
    # --------------------------
    model_type = config["model"].get("type", "cnn")  # cnn / fcnn / resnet
    print(f"\n⚙️ Using model type: {model_type}")

    if model_type == "cnn":
        model_name = config["model"].get("name", "SimpleCNNSmall")
        if model_name == "SimpleCNNSmall":
            model = SimpleCNNSmall(num_classes=num_classes,
                                   input_size=input_size,
                                   input_channels=input_channels)
        else:
            model = CNN2DNet(num_classes=num_classes,
                             input_channels=input_channels,
                             input_size=input_size[0])

    elif model_type == "fcnn":
        model_name = config["model"].get("name", "FCNNSmall")
        if model_name == "FCNNSmall":
            model = FCNNSmall(num_classes=num_classes,
                              input_size=input_size)
        else:
            model = FCNNNet(input_size=input_size[0],
                            input_channels=input_channels,
                            num_classes=num_classes)

    elif model_type == "resnet":
        model = ResNetCustom(num_classes=num_classes,
                             input_channels=input_channels)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # --------------------------
    # Step 4.6: Move model to device
    # --------------------------
    device = get_device()
    model = model.to(device)
    print(f"\n✅ Model initialized on device: {device}")
    print(model)

    # --------------------------
    # Step 4.7: Forward pass sanity check
    # --------------------------
    with torch.no_grad():
        x_sample = x_sample.to(device)
        out = model(x_sample)
        print(f"\n✅ Forward pass successful, output shape: {out.shape}")

    # --------------------------
    # Step 4.8: Training
    # --------------------------
    print("\n🏋️ Starting training...")
    trained_model = train_model(model, train_loader, val_loader, config, device)

    # --------------------------
    # Step 4.9: Save trained model
    # --------------------------
    model_save_path = PROJECT_ROOT / "experiments" / "saved_models" / f"{model_type}_{model_name}.pt"
    model_save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(trained_model.state_dict(), model_save_path)
    print(f"✅ Model saved at: {model_save_path}")

    # --------------------------
    # Step 4.10: Evaluation
    # --------------------------
    print("\n🔍 Evaluating model on validation set...")
    evaluate_model(trained_model, val_loader, device)