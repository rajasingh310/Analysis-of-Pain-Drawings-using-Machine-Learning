import os
from pathlib import Path
import yaml
import torch
from torch.utils.data import Dataset, DataLoader, random_split, WeightedRandomSampler
from collections import Counter


# ==============================
# 🔹 DATASET (LOAD .pt FILES)
# ==============================
class ProcessedTensorDataset(Dataset):
    def __init__(self, root_dir):
        """
        root_dir structure:
        processed_dir/
            0/
                0.pt, 1.pt, ...
            1/
                0.pt, 1.pt, ...
        """
        self.samples = []

        class_dirs = sorted(os.listdir(root_dir))

        for label, class_name in enumerate(class_dirs):
            class_path = os.path.join(root_dir, class_name)

            for f in os.listdir(class_path):
                if f.endswith(".pt"):
                    self.samples.append((os.path.join(class_path, f), label))

        if not self.samples:
            raise ValueError(f"No .pt files found in {root_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        tensor = torch.load(path)
        return tensor, label


# ==============================
# 🔹 DATALOADER FUNCTION
# ==============================
def get_dataloaders(config):

    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

    processed_dir = PROJECT_ROOT / config["data"]["processed_dir"]
    print(f"📂 Loading processed data from: {processed_dir}")

    # -------------------------
    # Load dataset
    # -------------------------
    dataset = ProcessedTensorDataset(processed_dir)

    num_classes = len(set(label for _, label in dataset.samples))
    print(f"📊 Total samples: {len(dataset)} | Classes: {num_classes}")

    # -------------------------
    # Train / Validation Split
    # -------------------------
    train_size = int(len(dataset) * config["data"]["train_split"])
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    print(f"Train: {train_size}, Validation: {val_size}")

    # -------------------------
    # Weighted Sampler (optional)
    # -------------------------
    if config["data"].get("use_weighted_sampler", False):

        labels = [dataset.samples[i][1] for i in train_dataset.indices]
        class_counts = Counter(labels)

        print("⚖️ Train class distribution:", class_counts)

        weights = [1.0 / class_counts[label] for label in labels]

        sampler = WeightedRandomSampler(
            weights,
            num_samples=len(weights),
            replacement=True
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=config["data"]["batch_size"],
            sampler=sampler
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=config["data"]["batch_size"],
            shuffle=True
        )

    # -------------------------
    # Validation Loader
    # -------------------------
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["data"]["batch_size"],
        shuffle=False
    )

    return train_loader, val_loader, num_classes


# ==============================
# 🚀 STANDALONE RUN
# ==============================
if __name__ == "__main__":

    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    config_path = PROJECT_ROOT / "experiments/configs/aachen.yaml"

    print("Config path:", config_path)

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    train_loader, val_loader, num_classes = get_dataloaders(config)

    print("\n🚀 READY FOR TRAINING")

    # Debug: check batch shape
    for x, y in train_loader:
        print("Batch shape:", x.shape)
        print("Labels shape:", y.shape)
        break