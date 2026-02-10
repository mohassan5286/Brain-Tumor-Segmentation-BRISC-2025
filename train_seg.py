import argparse
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics.classification import BinaryJaccardIndex
from model.UNet import UNet  
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

class ComboLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(ComboLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice_loss = 1 - (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        BCE = nn.functional.binary_cross_entropy(inputs, targets, reduction='mean')
        combo_loss = BCE + dice_loss
        
        return combo_loss

def count_parameters(model, trainable_only=False):
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())

def train(model, train_loader, optimizer, criterion, iou_metric, device):
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

        preds = (outputs > 0.5).float()
        
        for i in range(inputs.size(0)):
            train_iou = iou_metric(preds[i].unsqueeze(0).int(), targets[i].unsqueeze(0).int()).item()
            if tumors_type[i] == 'gl':
                gl_train_iou.append(train_iou)
            elif tumors_type[i] == 'me':
                me_train_iou.append(train_iou)
            elif tumors_type[i] == 'pi':
                pi_train_iou.append(train_iou)

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

def validate(model, val_loader, criterion, iou_metric, device, epoch):
    model.eval()
    running_loss = 0.0
    gl_val_iou, me_val_iou, pi_val_iou = [], [], []

    with torch.no_grad():
        for inputs, targets, tumors_type in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            
            loss = criterion(outputs, targets)
            running_loss += loss.item()

            preds = (outputs > 0.5).float()

            for i in range(inputs.size(0)):
                val_iou = iou_metric(preds[i].unsqueeze(0).int(), targets[i].unsqueeze(0).int()).item()
                if tumors_type[i] == 'gl':
                    gl_val_iou.append(val_iou)
                elif tumors_type[i] == 'me':
                    me_val_iou.append(val_iou)
                elif tumors_type[i] == 'pi':
                    pi_val_iou.append(val_iou)
    
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

    train_loader, val_loader = get_data_loader(args.data_dir, args.bsz)

    model = UNet(in_channels=1, out_channels=1, feature_dims=64) 

    total_params = count_parameters(model)
    trainable_params = count_parameters(model, trainable_only=True)
    print(f"\nTotal # param.: {total_params}")
    print(f"Learnable # param.: {trainable_params}")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model = torch.nn.DataParallel(model, device_ids = [0, 1])
    
    criterion = ComboLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    iou_metric = BinaryJaccardIndex().to(device)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
    
    best_weighted_miou = 0.0

    # Train the model
    for epoch in range(args.niter): 
        print(f'\nEpoch {epoch+1}/{args.niter}')
        start_time = time.time()
        
        train(model, train_loader, optimizer, criterion, iou_metric, device)
        val_loss, weighted_miou = validate(model, val_loader, criterion, iou_metric, device, epoch)
        
        scheduler.step(val_loss)
        
        end_time = time.time()
        epoch_time = end_time - start_time
        print(f'\nEpoch {epoch+1} completed in {epoch_time:.2f} seconds\n')
        
        # Save the best model based on Weighted Mean IOU
        if weighted_miou > best_weighted_miou:
            best_weighted_miou = weighted_miou
            os.makedirs(args.logpath, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(args.logpath, 'best_model.pth'))
            print(f'Best model saved with Weighted IOU: {best_weighted_miou:.4f}')

    # --- Post Training: Filter and Save Images based on Thresholds ---
    print("\nTraining Finished. Starting post-training evaluation and saving...")
    
    # Load the best model
    best_model_path = os.path.join(args.logpath, 'best_model.pth')
    model.load_state_dict(torch.load(best_model_path))
    model.eval()

    # Create directories
    os.makedirs('glioma', exist_ok=True)
    os.makedirs('meningioma', exist_ok=True)
    os.makedirs('pituitary', exist_ok=True)

    with torch.no_grad():
        batch_idx = 0
        for inputs, targets, tumors_type in val_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            preds = (outputs > 0.5).float()

            for i in range(inputs.size(0)):
                iou = iou_metric(preds[i].unsqueeze(0).int(), targets[i].unsqueeze(0).int()).item()
                t_type = tumors_type[i]
                
                should_save = False
                save_folder = ""
                
                # Check thresholds
                if t_type == 'gl' and iou < 0.6:
                    should_save = True
                    save_folder = 'glioma'
                elif t_type == 'me' and iou < 0.8:
                    should_save = True
                    save_folder = 'meningioma'
                elif t_type == 'pi' and iou < 0.7:
                    should_save = True
                    save_folder = 'pituitary'
                
                if should_save:
                    filename_base = f"{iou:.4f}_batch{batch_idx}_idx{i}"
                    
                    # Save Image
                    save_image(inputs[i], os.path.join(save_folder, f"{filename_base}_image.png"))
                    
                    # Save Predicted Mask
                    save_image(preds[i], os.path.join(save_folder, f"{filename_base}_mask.png"))

                    # Save Target (Ground Truth) Mask
                    save_image(targets[i], os.path.join(save_folder, f"{filename_base}_target.png"))
            
            batch_idx += 1
    
    print("Post-training evaluation and saving completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train UNet model on Pneumonia chest X-ray segmentation dataset")
    parser.add_argument('--bsz', type=int, default=8, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate for optimizer')
    parser.add_argument('--niter', type=int, default=30, help='Number of training iterations (epochs)')
    parser.add_argument('--data_dir', type=str, default='segmentation', help='Directory where dataset is located')
    parser.add_argument('--logpath', type=str, default='log', help='Directory to save the best model checkpoint')
    
    args = parser.parse_args()
    main(args)