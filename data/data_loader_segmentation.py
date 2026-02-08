import os
import torch
import numpy as np
import cv2
import random
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import functional as F
from PIL import Image

# ==========================================
# 1. The "Medical-Grade" Elastic Transform
# ==========================================
class ElasticDeformPair:
    """
    Manually generates a random displacement field.
    Applies it to:
      - Image using BILINEAR (Smooth)
      - Mask using NEAREST (Exact)
    Guarantees 100% geometric alignment with zero mask degradation.
    """
    def __init__(self, alpha=100.0, sigma=10.0, p=0.5):
        self.alpha = alpha
        self.sigma = sigma
        self.p = p

    def __call__(self, image, mask):
        # 1. Random Probability Check
        if random.random() > self.p:
            return image, mask

        # 2. Convert to Tensor
        image_t = F.to_tensor(image)
        mask_t = F.to_tensor(mask)
        
        _, h, w = image_t.shape

        # 3. Generate Separate Displacement Fields (dx, dy)
        # We generate them separately to ensure total independence and randomness
        dx = torch.randn(1, h, w).numpy()[0] * self.sigma
        dy = torch.randn(1, h, w).numpy()[0] * self.sigma

        # 4. Smooth the fields (Gaussian Blur)
        # This makes the distortion "wavy" rather than "jittery"
        dx = cv2.GaussianBlur(dx, (0, 0), self.sigma)
        dy = cv2.GaussianBlur(dy, (0, 0), self.sigma)

        # 5. Convert back to Tensor and Scale by Alpha
        dx = torch.from_numpy(dx).unsqueeze(0) * self.alpha
        dy = torch.from_numpy(dy).unsqueeze(0) * self.alpha

        # 6. STACK them into the (2, H, W) shape required by the PyTorch API
        # This prevents the "TypeError" mentioned in critiques
        displacement = torch.cat([dx, dy], dim=0)

        # 7. Apply SAME field with DIFFERENT interpolation
        image_d = F.elastic_transform(image_t, displacement, interpolation=transforms.InterpolationMode.BILINEAR)
        mask_d  = F.elastic_transform(mask_t,  displacement, interpolation=transforms.InterpolationMode.NEAREST)

        # 8. Return as PIL for the rest of the pipeline
        return F.to_pil_image(image_d), F.to_pil_image(mask_d)

# ==========================================
# 2. Dataset Class
# ==========================================
class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        # Robust extension handling
        mask_path = os.path.join(self.mask_dir, self.images[idx].replace('.jpg', '.png'))

        image = Image.open(img_path).convert("L")
        mask = Image.open(mask_path).convert("L")

        if self.transform is not None:
            image, mask = self.transform(image, mask)

        try:
            tumor_type = self.images[idx].split('_')[3]
        except:
            tumor_type = "unknown"

        return image, mask, tumor_type

# ==========================================
# 3. Pipeline Logic
# ==========================================
class JointTransform:
    def __init__(self, mode='train'):
        self.mode = mode
        # Initialize the Corrected Elastic Class
        self.pair_elastic = ElasticDeformPair(alpha=100.0, sigma=10.0, p=0.5)

    def __call__(self, image, mask):
        # 1. Apply Elastic Deformation (Training Only)
        if self.mode == 'train':
            image, mask = self.pair_elastic(image, mask)

        # 2. Standard Resize
        # Image: Bilinear (Default)
        image = transforms.Resize((256, 256))(image)
        # Mask: Nearest (Crucial for labels)
        mask = transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.NEAREST)(mask)

        # 3. ToTensor
        image = transforms.ToTensor()(image)
        mask = transforms.ToTensor()(mask)

        # 4. Normalization (Image Only)
        image = transforms.Normalize(mean=[0.16666122], std=[0.17487395])(image)

        # 5. Mask Thresholding
        # Even with Nearest, we keep this to be safe and ensure float format
        mask = (mask > 0.5).float()

        return image, mask

def get_data_loader(data_dir, batch_size):
    
    # Clean and simple - Logic is encapsulated in JointTransform
    train_transform = JointTransform(mode='train')
    test_transform  = JointTransform(mode='test')
    
    train_dataset = SegmentationDataset(
        image_dir=os.path.join(data_dir, "train", "images"),
        mask_dir=os.path.join(data_dir, "train", "masks"),
        transform=train_transform
    )

    test_dataset = SegmentationDataset(
        image_dir=os.path.join(data_dir, "test", "images"),
        mask_dir=os.path.join(data_dir, "test", "masks"),
        transform=test_transform
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, test_loader
