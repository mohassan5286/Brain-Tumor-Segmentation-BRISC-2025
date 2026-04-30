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
    - [Test-Time Augmentation (TTA)](#test-time-augmentation-inference)
  - [💾 Checkpoint](#-checkpoint)
  - [🤝 Contributing](#-contributing)

## 🎯 Overview

This project tackles the `BRISC 2025 brain tumor segmentation challenge` by developing a robust deep learning pipeline that segments brain tumors from medical imaging data. Our solution tries to modify `The Lighweight TransNet` using `Spatial Attention`, `Combo loss` and `Data Augmentation`, `Test-Time Augmentation` to achieve high accuracy in tumor segmentation.

---

## 🏗️ Architecture
<picture>
  <!-- Dark mode -->
  <source srcset="docs/img/SA-TransUNet-Dark.svg" media="(prefers-color-scheme: dark)">
  
  <!-- Light mode -->
  <source srcset="docs/img/SA-TransUNet-Light.svg" media="(prefers-color-scheme: light)">
  
  <!-- REQUIRED fallback -->
  <img src="docs/img/SA-TransUNet-Light.svg" alt="Diagram">
</picture>

### Key Components:

1. **CNN Encoder:** Extracts fine, local features using standard DoubleConv layers.
2. **Spatial Attention:** Actively focuses the network on relevant tumor regions.
3. **Transformer Bottleneck:** Captures long-range global dependencies using multi-head self-attention.
4. **Multi-Scale Transformers:** Dynamically weights skip connection features before fusion.
5. **CNN Decoder:** Progressively upsamples and concatenates features for the final mask.
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
root_directory/
├── train/
│   ├── images/
│   │   ├── image_1.jpg
│   │   ├── image_2.jpg
│   │   └── ...
│   └── masks/
│       ├── image_1.png
│       ├── image_2.png
│       └── ...

└── test/
│   ├── images/
│   │   ├── image_1.jpg
│   │   ├── image_2.jpg
│   │   └── ...
│   └── masks/
│       ├── image_1.png
│       ├── image_2.png
│       └── ...
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

### Testing/Inference

```bash
python test_seg.py \
  --data_dir /path/to/test/images \
  --model_path /path/to/trained_model.pth \
  --bsz 32 \
  --output_dir /path/to/output/masks
```

### Test-Time Augmentation/Inference

```bash
python tta_inference.py \
  --data_dir /path/to/test/images \
  --model_path /path/to/trained_model.pth \
  --bsz 32 \
  --output_dir /path/to/output/masks
```

---

## Checkpoint
You can download the best pre-trained SA-TransUNet model weights for evaluation here:  
🔗 **[Best Model Checkpoint (Google Drive)](https://drive.google.com/drive/folders/1DY_CwsfGjO4pQg4zMCpyX0HsYmNi8yj3?usp=sharing)**

---

## 📄 License

This project is open source and available under the MIT License.

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request with improvements, bug fixes, or new features.

---