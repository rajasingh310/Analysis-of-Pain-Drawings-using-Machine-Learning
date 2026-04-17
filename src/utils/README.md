# Utils Directory

This directory contains utility modules for dataset loading, preprocessing, and preparation for model training.

## Files

- `data_loader.py`  
  Loads the raw image dataset from the configured directory. Maps class folders to image file paths, supports excluding specific class folders, and returns a dictionary mapping class indices to image file lists.

- `data_preprocess.py`  
  Implements the main preprocessing pipeline. Uses `data_loader.py` to load raw data, applies transformations (crop, resize, grayscale, etc.), balances classes if needed, saves processed tensors, and splits the dataset into training and validation sets.

- `load_processed_dataset.py`  
  Loads the preprocessed tensor datasets and creates PyTorch DataLoaders for training and validation, ready for use in model training scripts.

## Typical Pipeline

1. **Load raw data:**  
   `data_loader.py` loads and organizes raw image files from the dataset directory.

2. **Preprocess and save tensors:**  
   `data_preprocess.py` processes the raw images (resizing, grayscale conversion, balancing), saves them as tensors, and splits them into train/validation sets.

3. **Load processed data for training:**  
   `load_processed_dataset.py` loads the saved tensors and creates DataLoaders for model training and validation.

## Configuration

- All scripts use a YAML configuration file (e.g., `experiments/configs/aachen.yaml`) to specify dataset paths, preprocessing options, and training parameters.
- Update the config file to change dataset location, image size, batch size, preprocessing flags, crop settings, and other options.

## Usage

These modules are imported and used in the main experiment and training scripts (e.g., `experiments/run_pd_pipeline.py`) to automate data handling and preparation for deep learning workflows.
