# Guide: Using Custom Datasets with CE-VAE

This guide explains how to use your own datasets with the CE-VAE (Capsule Enhanced Variational AutoEncoder) codebase for underwater image enhancement.

## Table of Contents

1. [Overview](#overview)
2. [Dataset Preparation](#dataset-preparation)
3. [Configuration Files](#configuration-files)
4. [Running Locally (VS Code / Terminal)](#running-locally-vs-code--terminal)
5. [Running on Google Colab](#running-on-google-colab)
6. [Evaluation and Metrics](#evaluation-and-metrics)
7. [Common Issues and Troubleshooting](#common-issues-and-troubleshooting)

---

## Overview

CE-VAE supports working with paired underwater image datasets. This means you need:
- **Input images**: Degraded/underwater images that need enhancement
- **Ground Truth (GT) images**: Clean/enhanced reference images for training/evaluation

The system can be used for:
- **Training**: Train the model on your custom dataset
- **Testing/Inference**: Enhance images using a pre-trained model
- **Evaluation**: Compute metrics like PSNR, SSIM, LPIPS, UIQM, UCIQE, NIQE

---

## Dataset Preparation

### Required Folder Structure

Organize your dataset with the following structure:

```
/path/to/your/dataset/
├── train/
│   ├── GT/           # Ground truth (clean/enhanced) images
│   │   ├── image001.png
│   │   ├── image002.png
│   │   └── ...
│   └── input/        # Degraded/underwater images
│       ├── image001.png
│       ├── image002.png
│       └── ...
└── val/
    ├── GT/
    │   ├── image001.png
    │   └── ...
    └── input/
        ├── image001.png
        └── ...
```

### Important Notes:
- **Supported formats**: PNG, JPG, JPEG, BMP, TIFF, TIF
- **Paired images**: Ensure corresponding input and GT images have the **same filename**
- **Image naming**: Files are sorted alphabetically, so matching names ensures proper pairing

### Generate Dataset Text Files

After organizing your dataset, generate the text files that list all image paths:

```bash
# Replace with your actual dataset path
bash scripts/generate_dataset_txt.sh /path/to/your/dataset/
```

This script creates four files in the `./data/` folder:
- `LSUI_train_input.txt` - List of training input image paths
- `LSUI_train_target.txt` - List of training GT image paths  
- `LSUI_val_input.txt` - List of validation input image paths
- `LSUI_val_target.txt` - List of validation GT image paths

### For Multiple Datasets

If you have 3 different datasets, repeat the process for each:

```bash
# Dataset 1
bash scripts/generate_dataset_txt.sh /path/to/dataset1/

# Rename the generated files
mv data/LSUI_train_input.txt data/Dataset1_train_input.txt
mv data/LSUI_train_target.txt data/Dataset1_train_target.txt
mv data/LSUI_val_input.txt data/Dataset1_val_input.txt
mv data/LSUI_val_target.txt data/Dataset1_val_target.txt

# Repeat for Dataset 2 and 3
```

Alternatively, create the text files manually:
```bash
# List all input images
find /path/to/dataset/train/input -type f \( -name "*.png" -o -name "*.jpg" \) | sort > data/MyDataset_train_input.txt

# List all GT images
find /path/to/dataset/train/GT -type f \( -name "*.png" -o -name "*.jpg" \) | sort > data/MyDataset_train_target.txt
```

---

## Configuration Files

### Creating a Custom Config for Your Dataset

Copy an existing config and modify it for your dataset:

```bash
cp configs/cevae_E2E_lsui.yaml configs/cevae_E2E_mydataset.yaml
```

Edit the config file (`configs/cevae_E2E_mydataset.yaml`):

```yaml
model:
  target: src.models.cevae.CEVAE
  params:
    discriminator: False
    ckpt_path: data/imagenet-pre-trained-cevae.ckpt  # Pre-trained weights
    embed_dim: 256
    ddconfig:
      double_z: False
      z_channels: 256
      resolution: 256
      in_channels: 3
      out_ch: 3
      ch: 128
      ch_mult: [1, 1, 2, 2, 4]
      num_res_blocks: 2
      attn_resolutions: [16]
      dropout: 0.0

    lossconfig:
      target: src.modules.losses.combined.ReconstructionLossWithDiscriminator
      params:
        pixelloss_weight: 10.0
        perceptual_weight: 1.0
        gdl_loss_weight: 0.0
        color_loss_weight: 0.0
        ssim_loss_weight: 1.0
        disc_enabled: False

    optimizer:
      base_learning_rate: 4.5e-6

lightning:
  trainer:
    max_epochs: 600
    accelerator: gpu
    devices: 1  # Number of GPUs
    check_val_every_n_epoch: 10

data:
  target: src.data.dataset_wrapper.DataModuleFromConfig
  params:
    dataset_name: "MyDataset"  # Change to your dataset name
    train_batch_size: 8
    val_batch_size: 32
    num_workers: 8
    train:
      target: src.data.image_enhancement.DatasetTrainFromImageFileList
      params:
        training_images_list_file: data/MyDataset_train_input.txt  # Your input file
        target_images_list_file: data/MyDataset_train_target.txt   # Your GT file
        random_crop: True
        random_flip: True
        color_jitter:
          brightness: [0.9, 1.1]
          contrast: [0.9, 1.1]
          saturation: [0.9, 1.1]
          hue: [-0.02, 0.02]
        max_size: 288
        size: 256
    validation:
      target: src.data.image_enhancement.DatasetTestFromImageFileList
      params:
        test_images_list_file: data/MyDataset_val_input.txt       # Your validation input
        test_target_images_list_file: data/MyDataset_val_target.txt  # Your validation GT
        size: 256
    test:
      target: src.data.image_enhancement.DatasetTestFromImageFileList
      params:
        test_images_list_file: data/MyDataset_val_input.txt
        test_target_images_list_file: data/MyDataset_val_target.txt
        size: 256
```

### Key Parameters to Adjust:

| Parameter | Description |
|-----------|-------------|
| `devices` | Number of GPUs (set to 1 for single GPU) |
| `train_batch_size` | Batch size (reduce if OOM errors occur) |
| `max_epochs` | Number of training epochs |
| `size` | Image size for training (256 recommended) |
| `training_images_list_file` | Path to your training input list |
| `target_images_list_file` | Path to your training GT list |

---

## Running Locally (VS Code / Terminal)

### Prerequisites

1. **Environment Setup**:
```bash
# Create conda environment
conda create -n cevae python=3.11
conda activate cevae

# Install PyTorch
conda install pytorch==2.2.0 torchvision==0.17.0 pytorch-cuda=12.1 -c pytorch -c nvidia

# Install dependencies
pip install -r requirements.txt
```

2. **Download Pre-trained Model** (for testing or fine-tuning):
   - ImageNet pre-trained: [Download Link](https://uniudamce-my.sharepoint.com/:u:/g/personal/niki_martinel_uniud_it/ESe3q_vE9EtJur7Ioda8UMoBS-P8jCZdlXbLO3gp-XUKQg?e=RBpa8x)
   - LSUI pre-trained: [Download Link](https://uniudamce-my.sharepoint.com/:u:/g/personal/niki_martinel_uniud_it/ERyTb1QQvARBuU6GL_-M2egBlndUS6Xi0LLPxEP2AI8qxg?e=RMQMQu)
   
   Save to `./data/` folder.

### Training on Your Dataset

```bash
# Train without discriminator (Phase 1)
python main.py --config configs/cevae_E2E_mydataset.yaml

# Fine-tune with discriminator (Phase 2) - after Phase 1
python main.py --config configs/cevae_GAN_mydataset.yaml
```

Training logs and checkpoints are saved to `./training_logs/`.

### Testing/Inference on Your Dataset

Run inference on a folder of images:

```bash
python test.py \
    --config configs/cevae_E2E_lsui.yaml \
    --checkpoint path/to/your/checkpoint.pth \
    --data-path /path/to/your/test/images/ \
    --output-path ./output_enhanced/ \
    --batch-size 8 \
    --device cuda:0
```

**Arguments:**
- `--checkpoint`: Path to trained model checkpoint
- `--data-path`: Folder containing images to enhance
- `--output-path`: Where to save enhanced images
- `--batch-size`: Number of images to process at once
- `--device`: Use `cuda:0` for GPU, `cpu` for CPU

---

## Running on Google Colab

### Option 1: Quick Start Notebook

Create a new Colab notebook and run these cells:

```python
# Cell 1: Clone repository and install dependencies
# Replace with the actual repository URL or your fork
!git clone https://github.com/priyanshuharshbodhi1/ce-vae-underwater-image-enhancement.git
%cd ce-vae-underwater-image-enhancement

!pip install -r requirements.txt
```

```python
# Cell 2: Upload your dataset
from google.colab import drive
drive.mount('/content/drive')

# If your dataset is in Google Drive, create symlinks
!ln -s "/content/drive/MyDrive/my_dataset" "/content/dataset"
```

```python
# Cell 3: Generate dataset text files
!bash scripts/generate_dataset_txt.sh /content/dataset/
```

```python
# Cell 4: Download pre-trained weights
# Option A: Download from SharePoint (see README for links)
# You can manually download from the links in README and upload to Colab
!mkdir -p data

# Option B: If you've uploaded the checkpoint to Google Drive, copy it:
# !cp "/content/drive/MyDrive/your_checkpoint.ckpt" data/imagenet-pre-trained-cevae.ckpt

# Option C: If using gdown, replace FILE_ID with actual Google Drive file ID
# To get FILE_ID: share the file, copy link, extract ID from URL
# !pip install gdown
# !gdown "FILE_ID" -O data/imagenet-pre-trained-cevae.ckpt
```

```python
# Cell 5: Run training
!python main.py --config configs/cevae_E2E_mydataset.yaml
```

```python
# Cell 6: Run inference
!python test.py \
    --config configs/cevae_E2E_lsui.yaml \
    --checkpoint training_logs/YOUR_EXPERIMENT/checkpoints/last.ckpt \
    --data-path /content/dataset/val/input \
    --output-path ./output
```

### Option 2: Upload Dataset Directly

```python
# Upload zip file from local machine
from google.colab import files
uploaded = files.upload()  # Upload your dataset.zip

# Extract
!unzip dataset.zip -d /content/dataset
```

---

## Evaluation and Metrics

### Available Metrics

CE-VAE supports the following quality metrics:

| Metric | Type | Description |
|--------|------|-------------|
| **PSNR** | Reference | Peak Signal-to-Noise Ratio (higher is better) |
| **SSIM** | Reference | Structural Similarity Index (higher is better) |
| **LPIPS** | Reference | Learned Perceptual Image Patch Similarity (lower is better) |
| **UIQM** | No-Reference | Underwater Image Quality Measure |
| **UCIQE** | No-Reference | Underwater Color Image Quality Evaluation |
| **NIQE** | No-Reference | Natural Image Quality Evaluator |

### Running Evaluation

After generating enhanced images, you can compute metrics by comparing them with ground truth images. The metrics are automatically computed during training and shown in logs.

For custom evaluation, you can use the metrics module:

```python
import numpy as np
from PIL import Image
from src.metrics import compute

# Load images
enhanced_img = np.array(Image.open('enhanced.png'))
gt_img = np.array(Image.open('gt.png'))

# Compute metrics
metrics = compute(enhanced_img, gt_img, gt_metrics=True)
print(f"PSNR: {metrics['psnr']:.2f}")
print(f"SSIM: {metrics['ssim']:.4f}")
print(f"UIQM: {metrics['uiqm']:.4f}")
print(f"UCIQE: {metrics['uciqe']:.4f}")
```

---

## Common Issues and Troubleshooting

### 1. CUDA Out of Memory

**Solution**: Reduce batch size in config:
```yaml
train_batch_size: 4  # or 2
val_batch_size: 8
```

### 2. FileNotFoundError for dataset files

**Check**:
- Ensure paths in config point to existing `.txt` files
- Run `generate_dataset_txt.sh` before training
- Use absolute paths if relative paths don't work

### 3. Image size mismatch

**Solution**: Ensure all images can be resized to the training size (256x256). Very small images may cause issues.

### 4. Slow training on CPU

**Note**: This model is designed for GPU training. CPU training will be very slow. Use Colab with GPU runtime for free GPU access.

### 5. wandb login issues

The code uses Weights & Biases for logging. You can either:
- Login: `wandb login` and enter your API key
- Run offline: `wandb offline`

---

## Quick Reference Commands

```bash
# Generate dataset files
bash scripts/generate_dataset_txt.sh /path/to/dataset/

# Train model
python main.py --config configs/cevae_E2E_mydataset.yaml

# Test/Inference
python test.py --config configs/cevae_E2E_lsui.yaml \
    --checkpoint path/to/checkpoint.pth \
    --data-path /path/to/images \
    --output-path ./output

# Count model parameters and FLOPs
python test.py --config configs/cevae_E2E_lsui.yaml \
    --checkpoint path/to/checkpoint.pth \
    --data-path /path/to/images \
    --count-flops-params
```

---

## Summary: Which Environment to Use?

| Environment | Best For | Pros | Cons |
|-------------|----------|------|------|
| **VS Code (Local)** | Development, debugging, large datasets | Full control, persistent storage | Requires GPU hardware |
| **Google Colab** | Quick experiments, no GPU hardware | Free GPU access, easy setup | Session limits, storage limits |
| **Server/Cluster** | Full training, production | Best performance | Requires setup |

**Recommendation**: 
- Start with **Google Colab** for quick testing and experiments
- Move to **VS Code/Local** for serious development and large-scale training
