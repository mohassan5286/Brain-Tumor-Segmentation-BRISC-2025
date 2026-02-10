import argparse
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics.classification import BinaryJaccardIndex
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedKFold
from model.UNet import UNet
from data.data_loader_segmentation import get_data_loader

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
            if tumors_type[i] == 'gl': gl_train_iou.append(train_iou)
            elif tumors_type[i] == 'me': me_train_iou.append(train_iou)
            elif tumors_type[i] == 'pi': pi_train_iou.append(train_iou)

    train_loss = running_loss / len(train_loader)
    return train_loss

def validate_fold(model, val_loader, criterion, iou_metric, device):
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
                if tumors_type[i] == 'gl': gl_val_iou.append(val_iou)
                elif tumors_type[i] == 'me': me_val_iou.append(val_iou)
                elif tumors_type[i] == 'pi': pi_val_iou.append(val_iou)
    
    len_gl, len_me, len_pi = len(gl_val_iou), len(me_val_iou), len(pi_val_iou)
    total = len_gl + len_me + len_pi
    
    gl_miou = np.mean(gl_val_iou) if len_gl > 0 else 0
    me_miou = np.mean(me_val_iou) if len_me > 0 else 0
    pi_miou = np.mean(pi_val_iou) if len_pi > 0 else 0
    weighted_miou = (gl_miou * len_gl + me_miou * len_me + pi_miou * len_pi) / total if total > 0 else 0

    val_loss = running_loss / len(val_loader)
    print(f'   [Fold Val] Loss: {val_loss:.4f} | W-IoU: {weighted_miou:.4f}')
    return val_loss, weighted_miou

def evaluate_ensemble(test_loader, model_paths, weights, device, iou_metric):
    print("\n:=========== Final Ensemble Evaluation on HELD-OUT TEST Data ===========")
    
    models = []
    for path in model_paths:
        model = UNet(in_channels=1, out_channels=1, feature_dims=64)
        model.load_state_dict(torch.load(path))
        model.to(device)
        model = torch.nn.DataParallel(model, device_ids=[0, 1])
        model.eval()
        models.append(model)
    
    weights = torch.tensor(weights, device=device)
    weights = weights / weights.sum()
    
    gl_ens_iou, me_ens_iou, pi_ens_iou = [], [], []

    with torch.no_grad():
        for inputs, targets, tumors_type in test_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            ensemble_outputs = torch.zeros_like(inputs)
            for i, model in enumerate(models):
                outputs = model(inputs)
                ensemble_outputs += outputs * weights[i]
            
            preds = (ensemble_outputs > 0.5).float()
            
            for i in range(inputs.size(0)):
                iou = iou_metric(preds[i].unsqueeze(0).int(), targets[i].unsqueeze(0).int()).item()
                t_type = tumors_type[i]
                
                if t_type == 'gl': gl_ens_iou.append(iou)
                elif t_type == 'me': me_ens_iou.append(iou)
                elif t_type == 'pi': pi_ens_iou.append(iou)

    len_gl, len_me, len_pi = len(gl_ens_iou), len(me_ens_iou), len(pi_ens_iou)
    total = len_gl + len_me + len_pi
    
    gl_miou = np.mean(gl_ens_iou) if len_gl > 0 else 0
    me_miou = np.mean(me_ens_iou) if len_me > 0 else 0
    pi_miou = np.mean(pi_ens_iou) if len_pi > 0 else 0
    weighted_miou = (gl_miou * len_gl + me_miou * len_me + pi_miou * len_pi) / total if total > 0 else 0

    print(f'\n:=========== FINAL TEST SET RESULTS ===========')
    print(f'| Test Samples: {total}')
    print(f'| GL IoU: {gl_miou:.4f} ({len_gl})')
    print(f'| ME IoU: {me_miou:.4f} ({len_me})')
    print(f'| PI IoU: {pi_miou:.4f} ({len_pi})')
    print(f'| --------------------------------')
    print(f'| ENSEMBLE WEIGHTED IoU: {weighted_miou:.4f}')
    print(f':==============================================')

def main(args):
    print(":=========== Brain Tumor Segmentation (K-Fold + External Test) ===========")
    
    train_loader_full, test_loader_heldout = get_data_loader(args.data_dir, args.bsz)
    train_dataset_full = train_loader_full.dataset
    
    print(f"| Train Data (for K-Fold): {len(train_dataset_full)}")
    print(f"| Test Data (Held-out):    {len(test_loader_heldout.dataset)}")
    print(":=======================================================================")

    labels = []
    for i in range(len(train_dataset_full)):
        labels.append(train_dataset_full[i][2])

    skf = StratifiedKFold(n_splits=args.k_folds, shuffle=True, random_state=42)
    
    fold_best_models = []
    fold_weights = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(labels)), labels)):
        print(f"\n:=========== Starting Fold {fold + 1} / {args.k_folds} ===========\n")

        train_subset = Subset(train_dataset_full, train_idx)
        val_subset = Subset(train_dataset_full, val_idx)

        fold_train_loader = DataLoader(train_subset, batch_size=args.bsz, shuffle=True)
        fold_val_loader = DataLoader(val_subset, batch_size=args.bsz, shuffle=False)

        model = UNet(in_channels=1, out_channels=1, feature_dims=64) 
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model = torch.nn.DataParallel(model, device_ids=[0, 1])
        
        criterion = ComboLoss()
        optimizer = optim.AdamW(model.parameters(), lr=args.lr)
        iou_metric = BinaryJaccardIndex().to(device)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
        
        best_fold_iou = 0.0

        for epoch in range(args.niter): 
            print(f'Fold {fold + 1} | Epoch {epoch+1}', end=' ')
            
            train(model, fold_train_loader, optimizer, criterion, iou_metric, device)
            val_loss, weighted_miou = validate_fold(model, fold_val_loader, criterion, iou_metric, device)
            
            scheduler.step(val_loss)
            
            if weighted_miou > best_fold_iou:
                best_fold_iou = weighted_miou
                os.makedirs(args.logpath, exist_ok=True)
                save_path = os.path.join(args.logpath, f'best_model_fold{fold+1}.pth')
                torch.save(model.state_dict(), save_path)
        
        print(f"--> Fold {fold + 1} Finished. Best IoU: {best_fold_iou:.4f}")
        fold_best_models.append(os.path.join(args.logpath, f'best_model_fold{fold+1}.pth'))
        fold_weights.append(best_fold_iou)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    iou_metric = BinaryJaccardIndex().to(device)
    
    evaluate_ensemble(test_loader_heldout, fold_best_models, fold_weights, device, iou_metric)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train UNet model on Pneumonia chest X-ray segmentation dataset")
    parser.add_argument('--bsz', type=int, default=8, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate for optimizer')
    parser.add_argument('--niter', type=int, default=30, help='Number of training iterations (epochs)')
    parser.add_argument('--k_folds', type=int, default=5, help='Number of folds for Cross Validation')
    parser.add_argument('--data_dir', type=str, default='segmentation', help='Directory where dataset is located')
    parser.add_argument('--logpath', type=str, default='log', help='Directory to save the best model checkpoint')
    
    args = parser.parse_args()
    main(args)