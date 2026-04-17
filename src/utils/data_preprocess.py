import os
import shutil
import random
from pathlib import Path
import yaml
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, random_split, WeightedRandomSampler
from torchvision import transforms
from collections import Counter
import matplotlib.pyplot as plt
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.utils.data_loader import load_raw_dataset

# ==============================
# 🔹 FAST DATASET (LOAD .pt)
# ==============================
class FastTensorDataset(Dataset):
    def __init__(self, root_dir):
        self.samples = []
        class_dirs = sorted(os.listdir(root_dir))
        for label, class_name in enumerate(class_dirs):
            class_path = os.path.join(root_dir, class_name)
            for f in os.listdir(class_path):
                if f.endswith(".pt"):
                    self.samples.append((os.path.join(class_path, f), label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        tensor = torch.load(path)
        return tensor, label


def get_crop_settings(config):
    crop_config = config["data"].get("crop", {})
    enabled = crop_config.get("enabled", False)
    position = crop_config.get("position")
    percent = crop_config.get("percent", 0)
    preview_enabled = crop_config.get("preview_enabled", False)

    if not enabled:
        return {
            "enabled": False,
            "position": None,
            "percent": 0,
            "preview_enabled": False,
        }

    if position not in {"top", "bottom"}:
        raise ValueError("data.crop.position must be either 'top' or 'bottom'")

    if not isinstance(percent, (int, float)):
        raise ValueError("data.crop.percent must be a number between 0 and 100")

    if percent < 0 or percent >= 100:
        raise ValueError("data.crop.percent must be in the range [0, 100)")

    return {
        "enabled": True,
        "position": position,
        "percent": float(percent),
        "preview_enabled": bool(preview_enabled),
    }


def crop_image_by_config(img, crop_settings):
    if not crop_settings["enabled"] or crop_settings["percent"] == 0:
        return img

    width, height = img.size
    rows_to_remove = int(height * (crop_settings["percent"] / 100.0))

    if rows_to_remove <= 0:
        return img

    if rows_to_remove >= height:
        raise ValueError("Crop removes the full image height. Reduce data.crop.percent.")

    if crop_settings["position"] == "top":
        crop_box = (0, rows_to_remove, width, height)
    else:
        crop_box = (0, 0, width, height - rows_to_remove)

    return img.crop(crop_box)


def show_crop_preview(img, crop_settings, image_path):
    if not crop_settings["preview_enabled"]:
        return

    plt.ion()
    fig, ax = plt.subplots(figsize=(5, 8))
    preview_img = img.convert("RGB") if img.mode != "RGB" else img
    ax.imshow(preview_img)
    ax.set_title(
        f"Cropped preview: {crop_settings['position']} {crop_settings['percent']:.1f}%\n"
        f"{Path(image_path).name}"
    )
    ax.axis("off")
    plt.show(block=False)
    plt.pause(0.001)


# ==============================
# 🔹 MAIN PROCESSING PIPELINE
# ==============================
def process_dataset(config):

    # =========================
    # STEP 1: LOAD DATA
    # =========================
    class_to_files = load_raw_dataset(config)
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    processed_root = PROJECT_ROOT / Path(config["data"]["processed_dir"])

    # =========================
    # STEP 2: CLEAN DIRECTORY
    # =========================
    if processed_root.exists():
        print(f"\n🧹 Cleaning processed dir: {processed_root}")
        shutil.rmtree(processed_root)
    processed_root.mkdir(parents=True, exist_ok=True)

    # =========================
    # STEP 3: CONFIG FLAGS
    # =========================
    to_grayscale = config["data"].get("to_grayscale", False)
    to_rgb_after_grayscale = config["data"].get("to_rgb_after_grayscale", True)
    use_undersample = config["data"].get("use_undersample", False)
    use_transforms = config["data"].get("use_transforms", True)
    show_unique_res = config["data"].get("show_unique_res", False)
    img_size = config["data"]["img_size"]
    crop_settings = get_crop_settings(config)

    if crop_settings["enabled"]:
        print(
            f"✂️ Cropping {crop_settings['percent']:.1f}% from the "
            f"{crop_settings['position']} of each image before resizing"
        )

    # =========================
    # STEP 4: UNDERSAMPLING
    # =========================
    if use_undersample:
        min_count = min(len(files) for files in class_to_files.values())
        print(f"🔽 Undersampling to {min_count} samples per class")

    # =========================
    # STEP 5: TRANSFORM
    # =========================
    transform_list = []
    if use_transforms:
        transform_list.append(transforms.Resize((img_size, img_size)))
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)

    # =========================
    # STEP 6: PROCESS + SAVE
    # =========================
    unique_res = {}  # store original images for unique resolutions
    crop_preview_shown = False
    for label, files in class_to_files.items():

        class_dir = processed_root / str(label)
        class_dir.mkdir(parents=True, exist_ok=True)

        if use_undersample:
            files = random.sample(files, min_count)

        for i, fpath in enumerate(files):
            img = Image.open(fpath)

            # Record original resolution
            res = img.size + (len(img.getbands()),)  # (width, height, channels)
            if res not in unique_res:
                unique_res[res] = img

            img = crop_image_by_config(img, crop_settings)

            if crop_settings["preview_enabled"] and not crop_preview_shown:
                show_crop_preview(img, crop_settings, fpath)
                crop_preview_shown = True

            # Grayscale conversion
            if to_grayscale:
                img = img.convert("L")  # single channel
                if to_rgb_after_grayscale:
                    img = img.convert("RGB")
            else:
                img = img.convert("RGB")

            # Apply transform and save tensor
            tensor = transform(img)
            save_path = class_dir / f"{i}.pt"
            torch.save(tensor, save_path)

        print(f"Class {label}: {len(files)} saved")

    print(f"\n✅ Preprocessing complete: {processed_root}")

    # =========================
    # STEP 6.1: SHOW ORIGINAL & PROCESSED UNIQUE IMAGES
    # =========================
    if show_unique_res:
        n = len(unique_res)

        # ----- Original images -----
        print("\n🖼 Unique original resolutions:")
        fig, axes = plt.subplots(1, n, figsize=(4*n, 4))
        if n == 1:
            axes = [axes]
        for ax, (res, img) in zip(axes, unique_res.items()):
            ax.imshow(img if img.mode == "RGB" else img.convert("L"),
                      cmap='gray' if img.mode == "L" else None)
            ax.set_title(f"{res}")
            ax.axis("off")
        plt.suptitle("Original Images of Unique Resolutions")
        plt.show()

        # ----- Processed images -----
        print("\n🖼 Processed images (resized & converted):")
        processed_images = []
        processed_res = []
        for res, orig_img in unique_res.items():
            img = orig_img

            img = crop_image_by_config(img, crop_settings)

            # Process conversion same as preprocessing
            if to_grayscale:
                img = img.convert("L")
                if to_rgb_after_grayscale:
                    img = img.convert("RGB")
            else:
                img = img.convert("RGB")

            img = img.resize((img_size, img_size))
            processed_images.append(img)
            processed_res.append(img.size + (len(img.getbands()),))  # width, height, channels

        fig, axes = plt.subplots(1, n, figsize=(4*n, 4))
        if n == 1:
            axes = [axes]
        for ax, img, res in zip(axes, processed_images, processed_res):
            ax.imshow(img if img.mode == "RGB" else img.convert("L"),
                      cmap='gray' if img.mode == "L" else None)
            ax.set_title(f"{res}")
            ax.axis("off")
        plt.suptitle("Processed Images of Unique Original Resolutions")
        plt.show()

    # =========================
    # STEP 7: LOAD FAST DATASET
    # =========================
    dataset = FastTensorDataset(processed_root)
    print(f"📊 Total samples: {len(dataset)}")

    # =========================
    # STEP 8: TRAIN / VAL SPLIT
    # =========================
    train_size = int(len(dataset) * config["data"]["train_split"])
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    print(f"Train: {train_size}, Val: {val_size}")

    # =========================
    # STEP 9: WEIGHTED SAMPLER
    # =========================
    if config["data"].get("use_weighted_sampler", False):
        labels = [dataset.samples[i][1] for i in train_dataset.indices]
        class_counts = Counter(labels)
        print("⚖️ Class distribution:", class_counts)
        weights = [1.0 / class_counts[l] for l in labels]
        sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
        train_loader = DataLoader(train_dataset, batch_size=config["data"]["batch_size"], sampler=sampler)
    else:
        train_loader = DataLoader(train_dataset, batch_size=config["data"]["batch_size"], shuffle=True)

    val_loader = DataLoader(val_dataset, batch_size=config["data"]["batch_size"], shuffle=False)
    return train_loader, val_loader


# ==============================
# 🚀 STANDALONE RUN
# ==============================
if __name__ == "__main__":

    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    config_path = PROJECT_ROOT / "experiments/configs/aachen.yaml"
    print("Config path:", config_path)

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    train_loader, val_loader = process_dataset(config)

    print("\n🚀 READY FOR TRAINING")

    # Debug: show first batch shape
    for x, y in train_loader:
        print("Batch shape:", x.shape)
        break
