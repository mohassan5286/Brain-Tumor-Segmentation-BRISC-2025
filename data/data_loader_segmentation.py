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
# 1. Elastic Deformation Class (The "Platinum" Version)
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
        dx = torch.randn(h, w).numpy()
        dy = torch.randn(h, w).numpy()

        # 4. Smooth the fields (Gaussian Blur)
        dx = cv2.GaussianBlur(dx, (0, 0), self.sigma)
        dy = cv2.GaussianBlur(dy, (0, 0), self.sigma)

        # 5. Scale by Alpha
        dx = torch.from_numpy(dx).float() * self.alpha
        dy = torch.from_numpy(dy).float() * self.alpha

        # 6. STACK & NORMALIZE
        displacement = torch.stack([dx, dy], dim=-1).unsqueeze(0) # (1, H, W, 2)
        
        # Normalize to [-1, 1] for grid_sample
        displacement[..., 0] /= (w / 2.0)
        displacement[..., 1] /= (h / 2.0)

        # 7. Apply Transform
        image_d = F.elastic_transform(image_t, displacement, interpolation=transforms.InterpolationMode.BILINEAR)
        mask_d  = F.elastic_transform(mask_t,  displacement, interpolation=transforms.InterpolationMode.NEAREST)

        # 8. Return as PIL
        return F.to_pil_image(image_d), F.to_pil_image(mask_d)

# ==========================================
# 2. Contrast Stretch Class
# ==========================================
class ContrastStretch:
    def __call__(self, img):
        # Convert PIL -> Numpy
        img_np = np.array(img).astype(np.float32)
        mn, mx = img_np.min(), img_np.max()
        # Stretch to 0-255
        img_np = (img_np - mn) / (mx - mn + 1e-8) * 255.0
        # Convert Numpy -> PIL
        return Image.fromarray(img_np.astype(np.uint8))

# ==========================================
# 3. Dataset Class (MoE Filtered)
# ==========================================
class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None, target_tumor_type=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = sorted(os.listdir(image_dir))
        
        # --- MoE Filtering Logic ---
        if target_tumor_type:
            mapping = {
                'glioma': 'gl',
                'meningioma': 'me',
                'pituitary': 'pi'
            }
            target_code = mapping.get(target_tumor_type.lower())
            
            if target_code:
                # Keep ONLY images that match the target code
                self.images = [
                    img for img in self.images 
                    if len(img.split('_')) > 3 and img.split('_')[3] == target_code
                ]
                print(f"[{target_tumor_type.upper()} DATASET] Filtered: Keeping {len(self.images)} pure {target_tumor_type} images.")
            else:
                raise ValueError(f"Unknown tumor type: {target_tumor_type}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
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
# 4. Joint Transform (Merges Everything)
# ==========================================
class JointTransform:
    def __init__(self, base_transform, mask_transform=None, use_elastic=False):
        self.base_transform = base_transform
        self.mask_transform = mask_transform if mask_transform else base_transform
        self.use_elastic = use_elastic
        
        if self.use_elastic:
            self.elastic = ElasticDeformPair(alpha=100.0, sigma=10.0, p=0.5)

    def __call__(self, image, mask):
        # 1. Apply Elastic Deformation First (Training Only)
        if self.use_elastic:
            image, mask = self.elastic(image, mask)

        # 2. Apply Synchronized Geometric Transforms (Seed Trick)
        seed = random.randint(0, 2**32)
        
        # Apply to Image
        random.seed(seed)
        torch.manual_seed(seed)
        image = self.base_transform(image)

        # Apply to Mask
        random.seed(seed)
        torch.manual_seed(seed)
        mask = self.mask_transform(mask)

        return image, mask

# ==========================================
# 5. Data Loader Factory
# ==========================================
def get_data_loader(data_dir, batch_size, target_tumor_type=None):
    
    base_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomRotation(degrees=15),
        transforms.RandomHorizontalFlip(p=0.5),
        ContrastStretch(), 
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.16666122], std = [0.17487395])
    ])

    mask_transform = transforms.Compose([
        transforms.Resize((256, 256), interpolation = transforms.InterpolationMode.NEAREST),
        transforms.RandomRotation(degrees=15),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: (x > 0.5).float())
    ])

    # Enable elastic only for training
    train_transform = JointTransform(base_transform, mask_transform, use_elastic=True)

    # --- TEST TRANSFORMS ---
    test_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        ContrastStretch(),
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.16666122], std = [0.17487395])
    ])

    test_mask_transform = transforms.Compose([
        transforms.Resize((256, 256), interpolation = transforms.InterpolationMode.NEAREST),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: (x > 0.5).float())
    ])
    
    # Disable elastic for testing
    val_joint_transform = JointTransform(test_transform, test_mask_transform, use_elastic=False)

    # --- DATASETS ---
    train_dataset = SegmentationDataset(
        image_dir=os.path.join(data_dir, "train", "images"),
        mask_dir=os.path.join(data_dir, "train", "masks"),
        transform=train_transform,
        target_tumor_type=target_tumor_type
    )

    test_dataset = SegmentationDataset(
        image_dir=os.path.join(data_dir, "test", "images"),
        mask_dir=os.path.join(data_dir, "test", "masks"),
        transform=val_joint_transform,
        target_tumor_type=target_tumor_type
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, test_loader
