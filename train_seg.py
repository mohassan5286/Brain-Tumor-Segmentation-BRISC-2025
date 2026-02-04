import argparse
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics.classification import BinaryJaccardIndex
from model.UNet import UNet  # Ensure UNet class is defined in model/unet.py
from data.data_loader_segmentation import get_data_loader
from torchvision.utils import save_image

def dice_coefficient(pred, target, threshold=0.5):
    pred = (pred > threshold).float()
    target = target.float()
    
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    
    if union.item() == 0:
        return 1.0
    
    return 2.0 * intersection / union

# FIX 1: Updated ComboLoss to calculate Dice PER IMAGE
class ComboLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(ComboLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        # Flatten pixels but KEEP batch dimension (Batch, H*W)
        inputs = inputs.view(inputs.size(0), -1)
        targets = targets.view(targets.size(0), -1)
        
        # Calculate Dice per image
        intersection = (inputs * targets).sum(dim=1)
        dice = (2. * intersection + smooth) / (inputs.sum(dim=1) + targets.sum(dim=1) + smooth)
        dice_loss = 1 - dice.mean() # Average dice loss over batch
        
        # BCE is calculated per pixel then averaged, which is standard
        BCE = nn.functional.binary_cross_entropy(inputs, targets, reduction='mean')
        
        return BCE + dice_loss
        
def count_parameters(model, trainable_only=False):
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())

def train(model, train_loader, optimizer, criterion, iou_metric, device, tumor_type_name):
    model.train()
    running_loss = 0.0
    ious = []

    for inputs, targets, tumors_type in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)

        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        preds = (outputs > 0.5).float()
        
        # --- VECTORIZED MACRO IOU (CORRECTED) ---
        # 1. Sum Intersection & Union per image
        intersection = (preds * targets).sum(dim=(1, 2, 3))
        union = preds.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3)) - intersection
        
        # 2. Calculate IoU per image
        batch_ious = (intersection + 1e-6) / (union + 1e-6)
        
        # 3. FIX: Extend the list with individual scores (move to CPU first) instead of append
        ious.extend(batch_ious.detach().cpu().tolist())

    mean_iou = np.mean(ious) if len(ious) > 0 else 0
    train_loss = running_loss / len(train_loader)

    print(f'\n[TRAIN] Loss: {train_loss:.4f} | {tumor_type_name.upper()} mIoU: {mean_iou:.4f}')
    
def validate(model, val_loader, criterion, iou_metric, device, epoch, tumor_type_name):
    model.eval()
    running_loss = 0.0
    val_outputs = []
    ious = []

    with torch.no_grad():
        for inputs, targets, tumors_type in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            val_outputs.append(outputs)
            loss = criterion(outputs, targets)
            running_loss += loss.item()

            preds = (outputs > 0.5).float()

            # --- VECTORIZED MACRO IOU (CORRECTED) ---
            # 1. Sum Intersection & Union per image
            intersection = (preds * targets).sum(dim=(1, 2, 3))
            union = preds.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3)) - intersection
            
            # 2. Calculate IoU per image
            batch_ious = (intersection + 1e-6) / (union + 1e-6)
            
            # 3. FIX: Extend the list with individual scores (move to CPU first) instead of append
            ious.extend(batch_ious.detach().cpu().tolist())
    
    # Create a specific folder for this expert
    save_dir = f'validation_output_masks_{tumor_type_name}'
    os.makedirs(save_dir, exist_ok=True)
    
    for i, output in enumerate(val_outputs):
        save_image((output > 0.5).float(), os.path.join(save_dir, f'epoch_{epoch}_batch_{i}_output.png'))

    mean_iou = np.mean(ious) if len(ious) > 0 else 0
    val_loss = running_loss / len(val_loader)

    print(f'\n[VAL] Loss: {val_loss:.4f} | {tumor_type_name.upper()} mIoU: {mean_iou:.4f}')

    return val_loss, mean_iou


def main(args):
    print(":=========== Brain Tumor Segmentation ===========")
    print(f"|             datapath: {args.data_dir}")
    print(f"|              logpath: {args.logpath}")
    print(f"|                  bsz: {args.bsz}")
    print(f"|                   lr: {args.lr}")
    print(f"|                niter: {args.niter}")
    print(":==========================================================")

    # Get the data loaders for the Brisc 2025 dataset
    train_loader, val_loader = get_data_loader(args.data_dir, args.bsz, target_tumor_type=args.tumor_type)

    # Instantiate the UNet model
    model = UNet(in_channels=1, out_channels=1, feature_dims=64)  # UNet for binary segmentation

    # Print the number of parameters
    total_params = count_parameters(model)
    trainable_params = count_parameters(model, trainable_only=True)
    print(f"\nTotal # param.: {total_params}")
    print(f"Learnable # param.: {trainable_params}")

    # Move the model to the GPU if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = torch.nn.DataParallel(model)
        
    # Define loss function and optimizer
    criterion = ComboLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    iou_metric = BinaryJaccardIndex().to(device)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
    
    best_val_mIoU = float('-inf')
    save_path = ""
    
    # Train the model
    for epoch in range(args.niter):  # Train for the specified number of epochs
        print(f'\nEpoch {epoch+1}/{args.niter}')
        start_time = time.time()
        
        train(model, train_loader, optimizer, criterion, iou_metric, device, args.tumor_type)
        val_loss, mean_iou = validate(model, val_loader, criterion, iou_metric, device, epoch, args.tumor_type)
        
        scheduler.step(val_loss)
        
        end_time = time.time()
        epoch_time = end_time - start_time
        print(f'\nEpoch {epoch+1} completed in {epoch_time:.2f} seconds\n')
        
        # Save the best model
        if mean_iou > best_val_mIoU:
            best_val_mIoU = mean_iou
            os.makedirs(args.logpath, exist_ok=True)
            # Save with a specific name for the expert
            save_path = os.path.join(args.logpath, f'best_model_{args.tumor_type}.pth')
            # FIX 2: Correctly save weights regardless of DataParallel
            if isinstance(model, nn.DataParallel):
                torch.save(model.module.state_dict(), save_path)
            else:
                torch.save(model.state_dict(), save_path)
                
            print(f'New Best {args.tumor_type} model saved with mean IoU: {best_val_mIoU:.4f}')
    
    print(f'Best {args.tumor_type} model saved with mean IoU: {best_val_mIoU:.4f} Location: {save_path}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train UNet on brain tumor segmentation using BRISC 2025 dataset")
    parser.add_argument('--bsz', type=int, default=8, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate for optimizer')
    parser.add_argument('--niter', type=int, default=30, help='Number of training iterations (epochs)')
    parser.add_argument('--data_dir', type=str, default='segmentation', help='Directory where dataset is located')

    # NEW ARGUMENT
    parser.add_argument('--tumor_type', type=str, required=True, 
                        choices=['glioma', 'meningioma', 'pituitary'],
                        help='Which expert to train? (Filters dataset)')
    
    parser.add_argument('--logpath', type=str, default='log', help='Directory to save the best model checkpoint')

    
    args = parser.parse_args()
    main(args)
