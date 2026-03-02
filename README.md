# Brain Tumor Segmentation - BRISC 2025

## 📋 Table of Contents

- [Brain Tumor Segmentation - BRISC 2025](#brain-tumor-segmentation---brisc-2025)
  - [📋 Table of Contents](#-table-of-contents)
  - [🎯 Overview](#-overview)
  - [🏗️ Architecture](#️-architecture)
    - [Key Components:](#key-components)
  - [🔧 Installation](#-installation)
    - [Prerequisites](#prerequisites)
    - [Setup](#setup)
    - [Dependencies](#dependencies)
  - [📊 Dataset](#-dataset)
  - [🚀 Usage](#-usage)
    - [Training](#training)
    - [Testing/Inference](#testinginference)
    - [Jupyter Notebook](#jupyter-notebook)
  - [📄 License](#-license)
  - [🤝 Contributing](#-contributing)

## 🎯 Overview

This project tackles the `BRISC 2025 brain tumor segmentation challenge` by developing a robust deep learning pipeline that segments brain tumors from medical imaging data. Our solution tries to modify `TransNet` using `Spatial Attention`, `Combo loss` and `Data Augmentation` to achieve high accuracy in tumor segmentation.

---

## 🏗️ Architecture

![Architecture Diagram](./docs/img/1.png)

### Key Components:

1. **Encoder Blocks**: Multi-scale feature extraction with DoubleConv layers
2. **Spatial Attention**: Learns to focus on relevant spatial regions
3. **Transformer Decoder**: Refines predictions using attention-based mechanisms
4. **Position Embedding**: Sine-based positional embeddings for spatial awareness
5. **Decoder Pathway**: Progressive upsampling with feature concatenation

---

## 🔧 Installation

### Prerequisites
- Python 3.8+
- CUDA 11.0+ (for GPU support)
- pip or conda package manager

### Setup

```bash
# Clone the repository
git clone https://github.com/mohassan5286/Brain-Tumor-Segmentation---BRISC-2025.git
cd Brain-Tumor-Segmentation---BRISC-2025

# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Dependencies
- PyTorch 2.0+
- torchvision
- torchmetrics
- Pillow
- NumPy
- SciPy

---

## 📊 Dataset

The dataset structure should follow this format:

```
segmentation_task/
├── train/
│   ├── images/
│   │   ├── image_1.jpg
│   │   ├── image_2.jpg
│   │   └── ...
│   └── masks/
│       ├── image_1.png
│       ├── image_2.png
│       └── ...
├── val/
│   └── (same structure as train)
└── test/
    └── (test images only)
```

**Note**: Images are processed in grayscale (single channel) with 256×256 resolution.

---

## 🚀 Usage

### Training

```bash
python train_seg.py \
  --data_dir /path/to/segmentation_task \
  --bsz 32 \
  --niter 100 \
  --lr 0.0001
```

**Training Arguments:**
- `--data_dir`: Path to the dataset directory
- `--bsz`: Batch size (default: 32)
- `--niter`: Number of training iterations/epochs (default: 100)
- `--lr`: Learning rate (default: 0.0001)
- `--model_save_path`: Path to save model checkpoints (optional)
- `--device`: GPU device ID (default: 0)

### Testing/Inference

```bash
python test_seg.py \
  --data_dir /path/to/test/images \
  --model_path /path/to/trained_model.pth \
  --bsz 32 \
  --output_dir /path/to/output/masks
```

**Testing Arguments:**
- `--data_dir`: Path to test images directory
- `--model_path`: Path to trained model weights
- `--bsz`: Batch size for inference (default: 32)
- `--output_dir`: Directory to save predicted masks (optional)

**Output:**
- Binary segmentation masks (PNG format)
- Threshold: 0.5 (configurable)
- Naming convention: `{image_name}_mask.png`

### Jupyter Notebook

An interactive notebook (`brain-tumor-segmentation.ipynb`) is available to use in google colab or kaggle.

---

## 📄 License

This project is open source and available under the MIT License.

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request with improvements, bug fixes, or new features.

---