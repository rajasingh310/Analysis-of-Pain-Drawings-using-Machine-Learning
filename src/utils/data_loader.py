import os
from pathlib import Path
import yaml


def load_raw_dataset(config):
    """
    Loads raw image dataset from the specified path in the configuration file.
    Returns a dictionary mapping class indices to lists of image file paths.
    """

    # ---------------------------
    # STEP 1: Determine project root
    # ---------------------------
    # __file__ is the current file path; .parent.parent.parent moves 3 levels up
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    print("Project path: ", PROJECT_ROOT)

    # ---------------------------
    # STEP 2: Build dataset path
    # ---------------------------
    # base_dir and dataset name are read from config
    base_dir = Path(config["data"]["base_dir"])  # e.g., "data/raw"
    dataset_name = config["data"]["dataset"]     # e.g., "aachen"
    dataset_path = PROJECT_ROOT / base_dir / dataset_name
    print("Dataset path: ", dataset_path)

    # Check if dataset exists
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path not found: {dataset_path}")

    print(f"\n📂 Loaded dataset from: {dataset_path}")

    # ---------------------------
    # STEP 3: Identify class directories
    # ---------------------------
    # Optionally exclude certain directories
    exclude_dirs = config["data"].get("exclude_dirs", [])

    # List all directories in dataset_path that are not in exclude_dirs
    class_dirs = sorted([
        d for d in os.listdir(dataset_path)
        if (dataset_path / d).is_dir() and d not in exclude_dirs
    ])

    if not class_dirs:
        raise ValueError(f"No class folders found in {dataset_path}")

    # ---------------------------
    # STEP 4: Map class indices to image files
    # ---------------------------
    class_to_files = {}

    for idx, class_name in enumerate(class_dirs):
        class_path = dataset_path / class_name

        # Gather all image files in this class folder
        files = [
            str(class_path / f)
            for f in os.listdir(class_path)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]

        # Only include classes with images
        if files:
            class_to_files[idx] = files
            print(f"  Class [{idx}] '{class_name}': {len(files)} images")
        else:
            print(f"  ⚠️ Skipping empty class '{class_name}'")

    if not class_to_files:
        raise ValueError(f"No valid images found in {dataset_path}")

    print(f"\n✅ Total classes loaded: {len(class_to_files)}")

    return class_to_files


# ---------------------------
# 🚀 Standalone execution (debug mode)
# ---------------------------
if __name__ == "__main__":

    # Determine project root
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    print("Project path: ", PROJECT_ROOT)

    # Load config YAML
    base_dir = "experiments/configs/"
    config_name = "aachen.yaml"
    config_path = PROJECT_ROOT / base_dir / config_name
    print("Config path: ", config_path)

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Load dataset using the function
    dataset = load_raw_dataset(config)

    print("Dataset Loaded Successfully")