# ==========================================
# TTA Inference and Evaluation Script - BRISC 2025
# ==========================================
import os
import argparse
import random
import torch
import numpy as np
import torch.nn as nn
from torchmetrics.classification import BinaryJaccardIndex
from model.UNet import UNet  
from data.data_loader_segmentation import get_data_loader

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ==========================================
# 5-View TTA Function
# ==========================================
def predict_with_heavy_tta(model, inputs):
    # 1. Original
    pred_1 = model(inputs)
    
    # 2. Horizontal Flip
    inputs_h = torch.flip(inputs, dims=[3]) 
    pred_2 = torch.flip(model(inputs_h), dims=[3]) 
    
    # 3. Vertical Flip
    inputs_v = torch.flip(inputs, dims=[2]) 
    pred_3 = torch.flip(model(inputs_v), dims=[2]) 

    # 4. Double Flip (Horizontal + Vertical)
    inputs_hv = torch.flip(inputs, dims=[2, 3])
    pred_4 = torch.flip(model(inputs_hv), dims=[2, 3]) 

    # 5. Intensity Scaling (Brightness)
    # Note: No clamping here because inputs are normalized via z-score
    inputs_bright = inputs * 1.1
    pred_5 = model(inputs_bright)

    return (pred_1 + pred_2 + pred_3 + pred_4 + pred_5) / 5.0

# ==========================================
# Evaluation Loop
# ==========================================
def evaluate_and_save_tta(model, val_loader, iou_metric, device):
    model.eval()
    gl_ious, me_ious, pi_ious = [], [], []
    
    # Check device count for dynamic printing
    device_count = torch.cuda.device_count()
    if device.type == 'cuda' and device_count > 1:
        device_display = f"{device_count} GPUs"
    else:
        device_display = str(device)
        
    print(f"Starting 5-VIEW Modified TTA Evaluation on {device_display}...")
    
    with torch.no_grad():
        for batch_idx, (inputs, targets, tumors_type) in enumerate(val_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            probs = predict_with_heavy_tta(model, inputs)
            preds = (probs > 0.5).float()
            
            for i in range(inputs.size(0)):
                iou = iou_metric(preds[i].unsqueeze(0).int(), targets[i].unsqueeze(0).int()).item()
                t_type = tumors_type[i]
                
                if t_type == 'gl':
                    gl_ious.append(iou)
                elif t_type == 'me':
                    me_ious.append(iou)
                elif t_type == 'pi':
                    pi_ious.append(iou)
            
            if (batch_idx + 1) % 10 == 0: 
                print(f"Processed {batch_idx + 1} batches...")

    def get_mean(lst): return np.mean(lst) if len(lst) > 0 else 0
    gl_mean, me_mean, pi_mean = get_mean(gl_ious), get_mean(me_ious), get_mean(pi_ious)
    total = len(gl_ious) + len(me_ious) + len(pi_ious)
    weighted_miou = (gl_mean * len(gl_ious) + me_mean * len(me_ious) + pi_mean * len(pi_ious)) / total if total > 0 else 0
    
    print("\n:=========== 5-VIEW TTA RESULTS ===========")
    print(f"| Glioma (GL):      {gl_mean:.4f}")
    print(f"| Meningioma (ME):  {me_mean:.4f}")
    print(f"| Pituitary (PI):   {pi_mean:.4f}")
    print("-------------------------------------------")
    print(f"| WEIGHTED MEAN IoU: {weighted_miou:.4f}")
    print(":==========================================")

# ==========================================
# Main Setup
# ==========================================
def main(args):
    set_seed(42)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    _, val_loader = get_data_loader(args.data_dir, args.bsz)
    
    model = UNet(in_channels=1, out_channels=1, feature_dims=64)
    model.to(device)
    
    # Keeping the state_dict fix to prevent DataParallel loading errors
    if os.path.exists(args.model_path):
        print(f"Loading checkpoint: {args.model_path}")
        checkpoint = torch.load(args.model_path, map_location=device)
        new_state_dict = {k.replace('module.', '') if k.startswith('module.') else k: v for k, v in checkpoint.items()}
        model.load_state_dict(new_state_dict)
        
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        
        iou_metric = BinaryJaccardIndex().to(device)
        evaluate_and_save_tta(model, val_loader, iou_metric, device)
    else:
        print(f"Error: Checkpoint not found at {args.model_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TTA Evaluation")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing dataset")
    parser.add_argument("--model_path", type=str, required=True, help="Path to best_model.pth")
    parser.add_argument("--bsz", type=int, default=32, help="Batch size for evaluation")
    
    args = parser.parse_args()
    main(args)
    