import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from pathlib import Path
from pdf2image import convert_from_path
from PIL import Image
from shutil import copy2

# Resolve project root dynamically
project_root = Path(__file__).resolve().parent.parent

# Paths
RAW_DIR = project_root / "data" / "raw" / "aachen" / "old datasets from Pain2D - PROMM and FSHD"

PROCESSED_DIR = project_root / "data" / "processed" / "aachen" / "old datasets from Pain2D - PROMM and FSHD"

# Create processed directory if it doesn't exist
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

def process_dir(raw_subdir: Path, processed_subdir: Path):
    """
    Convert PDFs to images and save in processed directory
    """
    processed_subdir.mkdir(parents=True, exist_ok=True)

    for file_path in sorted(raw_subdir.iterdir()):
        if file_path.is_dir():
            # Recursive call for subdirectories
            process_dir(file_path, processed_subdir / file_path.name)
        elif file_path.suffix.lower() == ".pdf":
            try:
                pages = convert_from_path(file_path, dpi=200)
                img_name = file_path.stem + ".png"  # preserve original stem
                target_path = processed_subdir / img_name
                if not target_path.exists():
                    pages[0].save(target_path, format="PNG")
                print(f"Converted PDF: {file_path} -> {target_path}")
            except Exception as e:
                print(f"Error converting {file_path}: {e}")
        elif file_path.suffix.lower() in [".png", ".jpg", ".jpeg"]:
            target_path = processed_subdir / file_path.name
            if not target_path.exists():
                copy2(file_path, target_path)
            print(f"Copied image: {file_path} -> {target_path}")
        else:
            print(f"Skipping unsupported file: {file_path}")

# Process all top-level directories in RAW_DIR
for class_dir in sorted(RAW_DIR.iterdir()):
    if class_dir.is_dir():
        target_dir = PROCESSED_DIR / class_dir.name
        process_dir(class_dir, target_dir)

print("All PDFs processed and images saved in processed directory!")