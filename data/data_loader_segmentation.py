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
# 1. The "Medical-Grade" Elastic Transform (PLATINUM)
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

        # 3. Generate Random Noise (Unit Gaussian)
        # Note: We do NOT multiply by sigma here. Sigma is only for the Blur.
        dx = torch.randn(h, w).numpy()
        dy = torch.randn(h, w).numpy()

        # 4. Smooth the fields (Gaussian Blur) with Sigma
        # This controls the "smoothness" or "frequency" of the distortion
        dx = cv2.GaussianBlur(dx, (0, 0), self.sigma)
        dy = cv2.GaussianBlur(dy, (0, 0), self.sigma)

        # 5. Scale by Alpha (Magnitude) and Cast to Float32
        # This controls the "intensity" or "amplitude" of the distortion
        dx = torch.from_numpy(dx).float() * self.alpha
        dy = torch.from_numpy(dy).float() * self.alpha

        # 6. STACK them into the (H, W, 2) shape
        displacement = torch.stack([dx, dy], dim=-1) # (H, W, 2)
        displacement = displacement.unsqueeze(0)     # (1, H, W, 2) for Batch Dim
        
        # -----------------------------------------------------------
        #  CRITICAL FIX: NORMALIZE DISPLACEMENT
        # Torchvision expects normalized [-1, 1] coordinates.
        # A shift of 'w/2' pixels equals a shift of '1.0' in normalized space.
        # -----------------------------------------------------------
        displacement[..., 0] /= (w / 2.0)  # Normalize X
        displacement[..., 1] /= (h / 2.0)  # Normalize Y

        # 7. Apply SAME field with DIFFERENT interpolation
        # image_t shape is (1, H, W)
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
        # PRO TIP: Sorting ensures reproducibility across different machines
        self.images = sorted(os.listdir(image_dir)) 

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
