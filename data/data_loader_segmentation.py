import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import random

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
        mask_path = os.path.join(self.mask_dir, self.images[idx].replace('.jpg', '.png'))

        image = Image.open(img_path).convert("L")
        mask = Image.open(mask_path).convert("L")

        if self.transform is not None:
            image, mask = self.transform(image, mask)

        tumor_type = self.images[idx].split('_')[3]

        return image, mask, tumor_type

class ContrastStretch:
    def __call__(self, img):
        img_np = np.array(img).astype(np.float32)
        mn, mx = img_np.min(), img_np.max()
        img_np = (img_np - mn) / (mx - mn + 1e-8) * 255.0
        return Image.fromarray(img_np.astype(np.uint8))


class JointTransform:
    def __init__(self, base_transform, mask_transform=None):
        self.base_transform = base_transform
        self.mask_transform = mask_transform if mask_transform else base_transform

    def __call__(self, image, mask):
        seed = random.randint(0, 2**32)
        random.seed(seed)
        torch.manual_seed(seed)
        image = self.base_transform(image)

        random.seed(seed)
        torch.manual_seed(seed)
        mask = self.mask_transform(mask)

        return image, mask

def get_data_loader(data_dir, batch_size):
    base_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        ContrastStretch(),
        transforms.ToTensor(),
    ])

    mask_transform = transforms.Compose([
        transforms.Resize((256, 256), interpolation = transforms.InterpolationMode.NEAREST),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: (x > 0.5).float())
    ])

    train_transform = JointTransform(base_transform, mask_transform)

    test_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        ContrastStretch(),
        transforms.ToTensor(),
    ])

    test_mask_transform = transforms.Compose([
        transforms.Resize((256, 256), interpolation = transforms.InterpolationMode.NEAREST),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: (x > 0.5).float())
    ])
    
    train_dataset = SegmentationDataset(
        image_dir=os.path.join(data_dir, "train", "images"),
        mask_dir=os.path.join(data_dir, "train", "masks"),
        transform=train_transform
    )

    test_dataset = SegmentationDataset(
        image_dir=os.path.join(data_dir, "test", "images"),
        mask_dir=os.path.join(data_dir, "test", "masks"),
        transform=JointTransform(test_transform, test_mask_transform)
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, test_loader
