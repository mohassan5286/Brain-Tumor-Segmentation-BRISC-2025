import argparse
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
# Removed torchmetrics dependency to use faster custom vectorization
from model.UNet import UNet  
from data.data_loader_segmentation import get_data_loader
from torchvision.utils import save_image


# ============================================
# LOSS FUNCTIONS 
# ============================================

class LovaszHingeLoss(nn.Module):
    """Binary Lovász hinge loss"""
    def __init__(self):
        super(LovaszHingeLoss, self).__init__()
    
    def forward(self, logits, labels):
        labels = labels.float()
        if logits.dim() == 4:
            logits = logits.squeeze(1)
        if labels.dim() == 4:
            labels = labels.squeeze(1)
            
        logits = torch.clamp(logits, min=-20, max=20)
        return self.lovasz_hinge_flat(logits, labels)
    
    def lovasz_hinge_flat(self, logits, labels):
        logits = logits.view(-1)
        labels = labels.view(-1)
        signs = 2.0 * labels - 1.0
        errors = 1.0 - logits * signs
        errors_sorted, perm = torch.sort(errors, descending=True)
        labels_sorted = labels[perm]
        grad = self.lovasz_grad(labels_sorted)
        loss = torch.dot(F.relu(errors_sorted), grad)
        return loss
    
    def lovasz_grad(self, gt_sorted):
        p = len(gt_sorted)
        gts = gt_sorted.sum()
        intersection = gts - gt_sorted.float().cumsum(0)
        union = gts + (1 - gt_sorted).float().cumsum(0)
        jaccard = 1.0 - intersection / union
        if p > 1:
            jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
        return jaccard


class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.7, beta=0.3, smooth=1.0):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
    
    def forward(self, inputs, targets):
        targets = targets.float()
        inputs = torch.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        TP = (inputs * targets).sum()
        FP = ((1 - targets) * inputs).sum()
        FN = (targets * (1 - inputs)).sum()
        
        Tversky = (TP + self.smooth) / (TP + self.alpha * FN + self.beta * FP + self.smooth)
        return 1 - Tversky


class FocalTverskyLoss(nn.Module):
    def __init__(self, alpha=0.7, beta=0.3, gamma=1.33, smooth=1.0):
        super(FocalTverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth
    
    def forward(self, inputs, targets):
        targets = targets.float()
        inputs = torch.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        TP = (inputs * targets).sum()
        FP = ((1 - targets) * inputs).sum()
        FN = (targets * (1 - inputs)).sum()
        
        Tversky = (TP + self.smooth) / (TP + self.alpha * FN + self.beta * FP + self.smooth)
        
        # Corrected Formula
        FocalTversky = torch.pow((1 - Tversky), self.gamma)
        return FocalTversky


def count_parameters(model, trainable_only=False):
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())

# ============================================
# OPTIMIZED TRAINING LOOP
# ============================================

def train(model, train_loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    gl_train_iou, me_train_iou, pi_train_iou = [], [], []

    for inputs, targets, tumors_type in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)

        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # --- OPTIMIZED VECTORIZED IOU (Replaces the slow loop) ---
        preds = (torch.sigmoid(outputs) > 0.5).float()
        
        # Calculate intersection and union for the whole batch at once on GPU
        # Dims: (Batch, Channel, H, W) -> sum over (1,2,3) gives (Batch,)
        intersection = (preds * targets).sum(dim=(1, 2, 3))
        union = preds.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3)) - intersection
        
        # Add epsilon to avoid divide by zero
        batch_ious = (intersection + 1e-6) / (union + 1e-6)
        
        # Convert to list to iterate efficiently in Python
        batch_ious_list = batch_ious.detach().cpu().tolist()

        # Sort into categories
        for i, iou_val in enumerate(batch_ious_list):
            t_type = tumors_type[i]
            if t_type == 'gl':
                gl_train_iou.append(iou_val)
            elif t_type == 'me':
                me_train_iou.append(iou_val)
            elif t_type == 'pi':
                pi_train_iou.append(iou_val)

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

def validate(model, val_loader, criterion, device, epoch):
    model.eval()
    running_loss = 0.0
    val_outputs = []
    gl_val_iou, me_val_iou, pi_val_iou = [], [], []

    with torch.no_grad():
        for inputs, targets, tumors_type in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            
            # Save for visualization (limit memory usage by saving only 1 batch per epoch if needed)
            if len(val_outputs) < 1: 
                val_outputs.append(outputs)
            
            loss = criterion(outputs, targets)
            running_loss += loss.item()

            # --- OPTIMIZED VECTORIZED IOU ---
            preds = (torch.sigmoid(outputs) > 0.5).float()
            intersection = (preds * targets).sum(dim=(1, 2, 3))
            union = preds.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3)) - intersection
            batch_ious = (intersection + 1e-6) / (union + 1e-6)
            batch_ious_list = batch_ious.cpu().tolist()

            for i, iou_val in enumerate(batch_ious_list):
                t_type = tumors_type[i]
                if t_type == 'gl':
                    gl_val_iou.append(iou_val)
                elif t_type == 'me':
                    me_val_iou.append(iou_val)
                elif t_type == 'pi':
                    pi_val_iou.append(iou_val)
    
    # Save validation output masks
    save_dir = 'validation_output_masks'
    os.makedirs(save_dir, exist_ok=True)
    # Only save the first batch we collected to save time/space
    for i, output in enumerate(val_outputs):
        save_image(torch.sigmoid(output), os.path.join(save_dir, f'epoch_{epoch}_batch_{i}_output.png'))

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

    return val_loss, weighted_miou


def main(args):
    print(":=========== Brain Tumor Segmentation ===========")
    print(f"|             datapath: {args.data_dir}")
    print(f"|              logpath: {args.logpath}")
    print(f"|                  bsz: {args.bsz}")
    print(f"|                   lr: {args.lr}")
    print(f"|                niter: {args.niter}")
    print(":==========================================================")

    # Get the data loaders for the Brisc 2025 dataset
    train_loader, val_loader = get_data_loader(args.data_dir, args.bsz)

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
    
    # Handle DataParallel
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = torch.nn.DataParallel(model)
    
    # Define loss function
    if args.loss_type == 'bce':
        criterion = nn.BCEWithLogitsLoss()
        print("Using BCE Loss")
    elif args.loss_type == 'lovasz':
        criterion = LovaszHingeLoss()
        print("Using Lovász-Hinge Loss")
    elif args.loss_type == 'tversky':
        criterion = TverskyLoss(alpha=args.alpha, beta=args.beta)
        print(f"Using Tversky Loss (α={args.alpha}, β={args.beta})")
    elif args.loss_type == 'focal_tversky':
        criterion = FocalTverskyLoss(alpha=args.alpha, beta=args.beta, gamma=args.gamma)
        print(f"Using Focal Tversky Loss (α={args.alpha}, β={args.beta}, γ={args.gamma})")
    else:
        raise ValueError(f"Unknown loss type: {args.loss_type}")
        
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
    
    best_weighted_miou = float('-inf')

    # Train the model
    for epoch in range(args.niter):  # Train for the specified number of epochs
        print(f'\nEpoch {epoch+1}/{args.niter}')
        start_time = time.time()
        
        # Note: I removed iou_metric from args as we calculate it manually now
        train(model, train_loader, optimizer, criterion, device)
        val_loss, weighted_miou = validate(model, val_loader, criterion, device, epoch)
        
        scheduler.step(val_loss)
        
        end_time = time.time()
        epoch_time = end_time - start_time
        print(f'\nEpoch {epoch+1} completed in {epoch_time:.2f} seconds\n')
        
        # Save the best model
        if weighted_miou > best_weighted_miou:
            best_weighted_miou = weighted_miou
            os.makedirs(args.logpath, exist_ok=True)
            
            # Safe saving for DataParallel (saves model.module instead of wrapper)
            save_path = os.path.join(args.logpath, 'best_model.pth')
            if isinstance(model, nn.DataParallel):
                torch.save(model.module.state_dict(), save_path)
            else:
                torch.save(model.state_dict(), save_path)
                
            # FIX: Correct variable name in print statement
            print(f'Best model saved with weighted miou: {best_weighted_miou:.4f}')
            
    print(f'\n Training Complete!')
    print(f'Best Validation Weighted mIoU: {best_weighted_miou:.4f}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train UNet on brain tumor segmentation using BRISC 2025 dataset")
    parser.add_argument('--bsz', type=int, default=8, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate for optimizer')
    parser.add_argument('--niter', type=int, default=30, help='Number of training iterations (epochs)')
    parser.add_argument('--data_dir', type=str, default='segmentation', help='Directory where dataset is located')
    parser.add_argument('--logpath', type=str, default='log', help='Directory to save the best model checkpoint')

    # Loss selection arguments
    parser.add_argument('--loss_type', type=str, default='focal_tversky', 
                        choices=['bce', 'lovasz', 'tversky', 'focal_tversky'],
                        help='Loss function to use')
    parser.add_argument('--alpha', type=float, default=0.7, help='Tversky alpha (FN weight)')
    parser.add_argument('--beta', type=float, default=0.3, help='Tversky beta (FP weight)')
    parser.add_argument('--gamma', type=float, default=1.33, help='Focal Tversky gamma')
    
    args = parser.parse_args()
    main(args)
