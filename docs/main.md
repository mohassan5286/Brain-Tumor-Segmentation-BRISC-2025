

## ✨ Features

- **Advanced UNet Architecture**: Modified U-Net with spatial attention mechanisms for improved feature learning
- **Transformer Integration**: Position-aware transformer layers for capturing long-range dependencies
- **Multiple Loss Functions**: 
  - Dice Loss for region-based optimization
  - Binary Cross-Entropy (BCE) for pixel-level accuracy
  - Combined Combo Loss for balanced learning
- **Data Augmentation**: Random rotations, resizing, and contrast stretching
- **GPU Support**: CUDA-enabled training and inference
- **Comprehensive Metrics**: Dice coefficient, Jaccard index (IoU), and more
- **Checkpoint Management**: Model saving and loading with training resumption

## 🧠 Model Details

### UNet Architecture

```
Encoder Path:
- Input: (1, 256, 256)
- Conv Block 1 → (64, 256, 256)
- MaxPool + Conv Block 2 → (128, 128, 128)
- MaxPool + Conv Block 3 → (256, 64, 64) + Spatial Attention
- MaxPool + Conv Block 4 → (512, 32, 32) + Spatial Attention
- MaxPool + Transformer Bottleneck → (1024, 16, 16)

Decoder Path:
- UpSample + Conv Block → Feature maps with skip connections
- Progressive refinement through 4 decoder levels
- Output: (1, 256, 256) with sigmoid activation
```

### Spatial Attention Mechanism

```python
SpatialAttention:
- Computes attention scores using max pooling
- Applies learned convolution filters
- Element-wise multiplication with input features
- Reduces spatial redundancy and amplifies informative regions
```

### Transformer Decoder

- **d_model**: 1024 (feature dimension)
- **nhead**: 4 (attention heads)
- **num_decoder_layers**: 1
- **dropout**: 0.1
- Enables modeling of long-range dependencies in feature maps
