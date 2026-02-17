import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
import torchvision.transforms.functional as TF
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
    def __init__(self, size=(256, 256), scale=(0.8, 1.0), ratio=(0.75, 1.33), is_train=True):
        self.size = size
        self.scale = scale
        self.ratio = ratio
        self.is_train = is_train
        self.mean = [0.16666122]
        self.std = [0.17487395]

    def __call__(self, image, mask):
        if self.is_train:
            i, j, h, w = T.RandomResizedCrop.get_params(
                image, scale=self.scale, ratio=self.ratio
            )
            image = TF.resized_crop(image, i, j, h, w, self.size, interpolation=T.InterpolationMode.BILINEAR)
            mask = TF.resized_crop(mask, i, j, h, w, self.size, interpolation=T.InterpolationMode.NEAREST)

            if random.random() > 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)

            if random.random() > 0.5:
                angle = random.uniform(-15, 15)
                image = TF.rotate(image, angle)
                mask = TF.rotate(mask, angle, interpolation=T.InterpolationMode.NEAREST)
        else:
            image = TF.resize(image, self.size, interpolation=T.InterpolationMode.BILINEAR)
            mask = TF.resize(mask, self.size, interpolation=T.InterpolationMode.NEAREST)

        image = TF.to_tensor(image)
        mask = TF.to_tensor(mask)

        image = TF.normalize(image, mean=self.mean, std=self.std)

        mask = (mask > 0.5).float()

        return image, mask

def get_data_loader(data_dir, batch_size):

    train_transform = JointTransform(scale=(0.5, 1.0), ratio=(0.75, 1.33), is_train=True)
    test_transform = JointTransform(is_train=False)
    
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
