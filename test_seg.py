# ============================================================================
# Brain Tumor Segmentation Testing/Inference Script - BRISC 2025
# ============================================================================
# This script performs inference on test images using a trained UNet model
# and evaluates the binary segmentation masks for brain tumors.
# ============================================================================

import os
import argparse
import random
import torch
from torchvision import transforms
import numpy as np
from torchmetrics.classification import BinaryJaccardIndex
from data.data_loader_segmentation import get_test_loader
from model.UNet import UNet

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ============================================================================
# Mask Evaluation Function
# ============================================================================


def evaluate_masks(test_loader, model, iou_metric, device):
    model.eval()
    
    gl_ious, me_ious, pi_ious = [], [], []

    print(f"Starting Evaluation on {device}...")

    with torch.no_grad():
        # Unpack targets and tumor_types alongside inputs
        for i, (inputs, targets, tumors_type) in enumerate(test_loader):
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            outputs = model(inputs)
            predictions = (outputs > 0.5).float()  

            # Compute IoU per instance in the batch
            for j in range(inputs.size(0)):
                iou = iou_metric(
                    predictions[j].unsqueeze(0).int(), 
                    targets[j].unsqueeze(0).int()
                ).item()
                
                t_type = tumors_type[j]
                if t_type == 'gl':
                    gl_ious.append(iou)
                elif t_type == 'me':
                    me_ious.append(iou)
                elif t_type == 'pi':
                    pi_ious.append(iou)

    # Calculate mean per class
    def get_mean(lst): return np.mean(lst) if len(lst) > 0 else 0
    gl_mean = get_mean(gl_ious)
    me_mean = get_mean(me_ious)
    pi_mean = get_mean(pi_ious)
    
    # Calculate weighted mean IoU based on instance counts
    total_samples = len(gl_ious) + len(me_ious) + len(pi_ious)
    weighted_miou = (
        (gl_mean * len(gl_ious) + me_mean * len(me_ious) + pi_mean * len(pi_ious)) / total_samples 
        if total_samples > 0 
        else 0
    )

    print("\n:=========== TEST EVALUATION RESULTS ===========")
    print(f"| Glioma (GL):      {gl_mean:.4f} ({len(gl_ious)} instances)")
    print(f"| Meningioma (ME):  {me_mean:.4f} ({len(me_ious)} instances)")
    print(f"| Pituitary (PI):   {pi_mean:.4f} ({len(pi_ious)} instances)")
    print("------------------------------------------------")
    print(f"| WEIGHTED MEAN IoU: {weighted_miou:.4f}")
    print(":===============================================")


# ============================================================================
# Main Inference Function
# ============================================================================


def main(args):
    set_seed(42)
    test_loader = get_test_loader(args.data_dir, args.bsz)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(in_channels=1, out_channels=1, feature_dims=64)
    
    if os.path.exists(args.model_path):
        print(f"Loading checkpoint: {args.model_path}")
        checkpoint = torch.load(args.model_path, map_location=device)
        new_state_dict = {k.replace('module.', '') if k.startswith('module.') else k: v for k, v in checkpoint.items()}
        model.load_state_dict(new_state_dict)
        
        model.to(device)
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)

        iou_metric = BinaryJaccardIndex().to(device)
        evaluate_masks(test_loader, model, iou_metric, device)
    else:
        print(f"Error: Checkpoint not found at {args.model_path}")


# ============================================================================
# Command-line Arguments and Entry Point
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate trained UNet model on test dataset"
    )
    parser.add_argument(
        "--data_dir", type=str, required=True, help="Directory containing test images"
    )
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to trained model weights"
    )
    parser.add_argument("--bsz", type=int, default=32, help="Batch size for testing")

    args = parser.parse_args()
    main(args)