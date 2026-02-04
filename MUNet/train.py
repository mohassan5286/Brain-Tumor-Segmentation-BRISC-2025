import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchmetrics.classification import BinaryJaccardIndex
from model import MUNet  # 导入 MUNet 模型
from loss import total_loss  # 导入自定义损失函数
from dataset import BRISC2025Dataset  # 假设有一个自定义数据集模块
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from data_loader_segmentation import get_data_loader
import numpy as np
from torchvision.utils import save_image

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
IN_CHANNELS = 1
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

data_dir = '/kaggle/input/brisc2025/brisc2025/segmentation_task/'

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
train_loader, val_loader = get_data_loader(data_dir, BATCH_SIZE)

# 初始化模型
model = MUNet(in_channels=IN_CHANNELS, out_channels=OUT_CHANNELS, embed_dim=EMBED_DIM, num_layers=NUM_LAYERS).to(device)

# 定义优化器和学习率调度器
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)  # 每 10 个 epoch 学习率减半
iou_metric = BinaryJaccardIndex().to(device)

# 定义损失函数
def compute_loss(preds, targets):
    return total_loss(preds, targets, alpha=1.0, beta=1.0, gamma=1.0)  # 调用自定义的加权损失函数

def train_fn(model, train_loader, optimizer):
    model.train()
    running_loss = 0.0
    gl_train_iou, me_train_iou, pi_train_iou = [], [], []

    for inputs, targets, tumors_type in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)

        loss = compute_loss(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        preds = (torch.sigmoid(outputs) > 0.5).float()
        
        for i in range(inputs.size(0)):
            train_iou = iou_metric(preds[i].unsqueeze(0).int(), targets[i].unsqueeze(0).int()).item()
            if tumors_type[i] == 'gl':
                gl_train_iou.append(train_iou)
            elif tumors_type[i] == 'me':
                me_train_iou.append(train_iou)
            elif tumors_type[i] == 'pi':
                pi_train_iou.append(train_iou)

    # Calculate weighted mIoU
    len_gl = len(gl_train_iou)
    len_me = len(me_train_iou)
    len_pi = len(pi_train_iou)
    total_samples = len_gl + len_me + len_pi
    
    gl_miou = np.mean(gl_train_iou) if len_gl > 0 else 0
    me_miou = np.mean(me_train_iou) if len_me > 0 else 0
    pi_miou = np.mean(pi_train_iou) if len_pi > 0 else 0
    weighted_miou = (gl_miou * len_gl + me_miou * len_me + pi_miou * len_pi) / total_samples if total_samples > 0 else 0

    train_loss = running_loss / len(train_loader)

    print(f'\n[TRAIN] Loss: {train_loss:.4f}')
    print(f'GL: {gl_miou:.4f} ({len_gl}) | ME: {me_miou:.4f} ({len_me}) | PI: {pi_miou:.4f} ({len_pi}) | Weighted: {weighted_miou:.4f}')

    return train_loss

def eval_fn(model, val_loader, epoch):
    model.eval()
    running_loss = 0.0
    val_outputs = []
    gl_val_iou, me_val_iou, pi_val_iou = [], [], []

    with torch.no_grad():
        for inputs, targets, tumors_type in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            val_outputs.append(outputs)
            loss = compute_loss(outputs, targets)
            running_loss += loss.item()

            preds = (torch.sigmoid(outputs) > 0.5).float()

            for i in range(inputs.size(0)):
                val_iou = iou_metric(preds[i].unsqueeze(0).int(), targets[i].unsqueeze(0).int()).item()
                if tumors_type[i] == 'gl':
                    gl_val_iou.append(val_iou)
                elif tumors_type[i] == 'me':
                    me_val_iou.append(val_iou)
                elif tumors_type[i] == 'pi':
                    pi_val_iou.append(val_iou)

    # Save validation output masks
    save_dir = 'validation_output_masks'
    os.makedirs(save_dir, exist_ok=True)
    for i, output in enumerate(val_outputs):
        save_image(output, os.path.join(save_dir, f'epoch_{epoch}_batch_{i}_output.png'))

    # Calculate weighted mIoU
    len_gl = len(gl_val_iou)
    len_me = len(me_val_iou)
    len_pi = len(pi_val_iou)
    total_samples = len_gl + len_me + len_pi
    
    gl_miou = np.mean(gl_val_iou) if len_gl > 0 else 0
    me_miou = np.mean(me_val_iou) if len_me > 0 else 0
    pi_miou = np.mean(pi_val_iou) if len_pi > 0 else 0
    weighted_miou = (gl_miou * len_gl + me_miou * len_me + pi_miou * len_pi) / total_samples if total_samples > 0 else 0

    val_loss = running_loss / len(val_loader)

    print(f'\n[VAL] Loss: {val_loss:.4f}')
    print(f'GL: {gl_miou:.4f} ({len_gl}) | ME: {me_miou:.4f} ({len_me}) | PI: {pi_miou:.4f} ({len_pi}) | Weighted: {weighted_miou:.4f}')

    return val_loss

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
        train_loss = train_fn(model, train_loader, optimizer)
        # if logging:
        print(f"Train Loss: {train_loss}")

        # 验证
        val_loss = eval_fn(model, val_loader, epoch)
        # if logging:
        print(f"Validation Loss: {val_loss}")

        # 学习率调度
        scheduler.step()

        # 保存最优模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, optimizer, epoch, CHECKPOINT_DIR)

if __name__ == "__main__":
    train()
