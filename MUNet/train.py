import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from model import MUNet  # 导入 MUNet 模型
from loss import total_loss  # 导入自定义损失函数
from dataset import BRISC2025Dataset  # 假设有一个自定义数据集模块
import os
import pandas as pd
from sklearn.model_selection import train_test_split


# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from config import logging

# if logging:
print(f"Using device: {device}")

# 参数设置
EPOCHS = 50
# EPOCHS = 1
BATCH_SIZE = 2
LEARNING_RATE = 1e-4
NUM_WORKERS = 4
IN_CHANNELS = 3
OUT_CHANNELS = 1
IMAGE_SIZE = 256
EMBED_DIM = 32
NUM_LAYERS = 4
CHECKPOINT_DIR = "./checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# 数据增强
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.22])
])

image_dir = '/kaggle/input/brisc2025/brisc2025/segmentation_task/train/images'
mask_dir = '/kaggle/input/brisc2025/brisc2025/segmentation_task/train/masks'

# if logging:
print("Sample image files:", os.listdir(image_dir)[:5])
print("Sample mask files:", os.listdir(mask_dir)[:5])

image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]
mask_files = [f for f in os.listdir(mask_dir) if f.endswith('.png')]

image_basenames = {os.path.splitext(f)[0]: f for f in image_files}
mask_basenames = {os.path.splitext(f)[0]: f for f in mask_files}

common_keys = set(image_basenames.keys()) & set(mask_basenames.keys()) # make them set of tuples

data = []
for key in sorted(common_keys):
    image_path = os.path.join(image_dir, image_basenames[key])
    mask_path = os.path.join(mask_dir, mask_basenames[key])
    data.append({'image_path': image_path, 'mask_path': mask_path})

df = pd.DataFrame(data)

full_dataset = BRISC2025Dataset(df)

train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size

torch.manual_seed(42)
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])


train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

# 初始化模型
model = MUNet(in_channels=IN_CHANNELS, out_channels=OUT_CHANNELS, embed_dim=EMBED_DIM, num_layers=NUM_LAYERS).to(device)

# 定义优化器和学习率调度器
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)  # 每 10 个 epoch 学习率减半

# 定义损失函数
def compute_loss(preds, targets):
    return total_loss(preds, targets, alpha=1.0, beta=1.0, gamma=1.0)  # 调用自定义的加权损失函数

# 训练函数
def train_fn(loader, model, optimizer):
    model.train()
    epoch_loss = 0.0
    for batch_idx, (images, masks) in enumerate(loader):
        images, masks = images.to(device), masks.to(device)

        # 前向传播
        preds = model(images)
        loss = compute_loss(preds, masks)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        if batch_idx % 10 == 0:  # 每 10 个 batch 打印一次日志
            # if logging:
            print(f"Batch {batch_idx}/{len(loader)}, Loss: {loss.item():.4f}")
            break
    return epoch_loss / len(loader)

# 验证函数
def eval_fn(loader, model):
    model.eval()
    epoch_loss = 0.0
    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(loader):
            images, masks = images.to(device), masks.to(device)

            # 前向传播
            preds = model(images)
            loss = compute_loss(preds, masks)

            epoch_loss += loss.item()

    return epoch_loss / len(loader)

# 保存模型
def save_checkpoint(model, optimizer, epoch, checkpoint_dir):
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pth")
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, checkpoint_path)
    # if logging:
    print(f"Checkpoint saved at {checkpoint_path}")

# 训练过程
def train():
    best_val_loss = float("inf")
    for epoch in range(EPOCHS):
        # if logging:
        print(f"Epoch [{epoch+1}/{EPOCHS}]")

        # 训练
        train_loss = train_fn(train_loader, model, optimizer)
        # if logging:
        print(f"Train Loss: {train_loss:.4f}")

        # 验证
        val_loss = eval_fn(val_loader, model)
        # if logging:
        print(f"Validation Loss: {val_loss:.4f}")

        # 学习率调度
        scheduler.step()

        # 保存最优模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, optimizer, epoch, CHECKPOINT_DIR)

if __name__ == "__main__":
    train()
