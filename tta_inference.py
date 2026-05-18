# ==========================================
# TTA Evaluation Script
# ==========================================
import os
import argparse
import random
import torch
import numpy as np
import torch.nn as nn
from torchmetrics.classification import BinaryJaccardIndex
from torchvision.utils import save_image
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

# --- TTA Function ---
def predict_with_tta(model, inputs):
    pred_1 = model(inputs)
    
    inputs_h = torch.flip(inputs, dims=[3])
    pred_2 = torch.flip(model(inputs_h), dims=[3])
    
    inputs_v = torch.flip(inputs, dims=[2])
    pred_3 = torch.flip(model(inputs_v), dims=[2])

    inputs_hv = torch.flip(inputs, dims=[2, 3])
    pred_4 = torch.flip(model(inputs_hv), dims=[2, 3])

    return (pred_1 + pred_2 + pred_3 + pred_4) / 4.0

# --- Evaluation Loop ---
def evaluate_and_save_tta(model, val_loader, iou_metric, device, output_dir):
    model.eval()
    gl_ious, me_ious, pi_ious = [], [], []
    
    dirs = ['glioma_tta', 'meningioma_tta', 'pituitary_tta']
    for d in dirs: 
        os.makedirs(os.path.join(output_dir, d), exist_ok=True)
        
    print(f"Starting 4-VIEW TTA Evaluation on {device}...")
    
    with torch.no_grad():
        for batch_idx, (inputs, targets, tumors_type) in enumerate(val_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            probs = predict_with_tta(model, inputs)
            preds = (probs > 0.5).float()
            
            for i in range(inputs.size(0)):
                iou = iou_metric(preds[i].unsqueeze(0).int(), targets[i].unsqueeze(0).int()).item()
                t_type = tumors_type[i]
                save_folder = ""
                should_save = False
                
                if t_type == 'gl':
                    gl_ious.append(iou)
                    if iou < 0.6: save_folder = dirs[0]; should_save = True
                elif t_type == 'me':
                    me_ious.append(iou)
                    if iou < 0.8: save_folder = dirs[1]; should_save = True
                elif t_type == 'pi':
                    pi_ious.append(iou)
                    if iou < 0.7: save_folder = dirs[2]; should_save = True
                
                if should_save:
                    filename = f"{iou:.4f}_b{batch_idx}_i{i}"
                    save_path = os.path.join(output_dir, save_folder)
                    save_image(inputs[i], os.path.join(save_path, f"{filename}_img.png"))
                    save_image(preds[i], os.path.join(save_path, f"{filename}_pred.png"))
            
            if (batch_idx + 1) % 10 == 0: 
                print(f"Processed {batch_idx + 1} batches...")

    def get_mean(lst): return np.mean(lst) if len(lst) > 0 else 0
    gl_mean, me_mean, pi_mean = get_mean(gl_ious), get_mean(me_ious), get_mean(pi_ious)
    total = len(gl_ious) + len(me_ious) + len(pi_ious)
    weighted_miou = (gl_mean * len(gl_ious) + me_mean * len(me_ious) + pi_mean * len(pi_ious)) / total if total > 0 else 0
    
    print("\n:=========== 4-VIEW TTA RESULTS ===========")
    print(f"| Glioma (GL):      {gl_mean:.4f}")
    print(f"| Meningioma (ME):  {me_mean:.4f}")
    print(f"| Pituitary (PI):   {pi_mean:.4f}")
    print("-------------------------------------------")
    print(f"| WEIGHTED MEAN IoU: {weighted_miou:.4f}")
    print(":==========================================")

# --- Main Setup ---
def main(args):
    set_seed(42)
    device = torch.device("cuda")
    _, val_loader = get_data_loader(args.data_dir, args.bsz)
    
    model = UNet(in_channels=1, out_channels=1, feature_dims=64)
    
    if os.path.exists(args.model_path):
        print(f"Loading checkpoint: {args.model_path}")
        checkpoint = torch.load(args.model_path, map_location=device)
        model.load_state_dict(checkpoint)
        
        model.to(device)
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        
        iou_metric = BinaryJaccardIndex().to(device)
        evaluate_and_save_tta(model, val_loader, iou_metric, device, args.output_dir)
    else:
        print(f"Error: Checkpoint not found at {args.model_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TTA Evaluation")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing dataset")
    parser.add_argument("--model_path", type=str, required=True, help="Path to best_model.pth")
    parser.add_argument("--output_dir", type=str, default="tta_analysis", help="Base directory to save bad predictions")
    parser.add_argument("--bsz", type=int, default=32, help="Batch size for evaluation")
    
    args = parser.parse_args()
    main(args)