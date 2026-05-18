# ============================================================================
# Brain Tumor Segmentation Testing/Inference Script - BRISC 2025
# ============================================================================
# This script performs inference on test images using a trained UNet model
# and generates binary segmentation masks for brain tumors.
# ============================================================================

import os
import argparse
import random
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
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
# Mask Generation Function
# ============================================================================


def generate_masks(test_loader, model, output_dir):
    model.eval()

    device = torch.device("cuda")
    model.to(device)

    with torch.no_grad():
        for i, (inputs, _, _) in enumerate(test_loader):
            inputs = inputs.to(device)
            outputs = model(inputs)
            predictions = (
                outputs > 0.5
            ).float()  

            for j in range(inputs.size(0)):
                pred_mask = predictions[j].cpu().numpy().squeeze()
                image_name = test_loader.dataset.images[i * test_loader.batch_size + j]
                save_path = os.path.join(
                    output_dir, image_name.replace(".jpg", "_mask.png")
                )

                pred_mask = (pred_mask * 255).astype(np.uint8)
                Image.fromarray(pred_mask).save(save_path)

                print(f"Saved {save_path}")


# ============================================================================
# Main Inference Function
# ============================================================================


def main(args):
    set_seed(42)
    test_loader = get_test_loader(args.data_dir, args.bsz)
    device = torch.device("cuda")

    model = UNet(in_channels=1, out_channels=1, feature_dims=64)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    
    model.to(device)
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    os.makedirs(args.output_dir, exist_ok=True)

    generate_masks(test_loader, model, args.output_dir)


# ============================================================================
# Command-line Arguments and Entry Point
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate masks using trained UNet model"
    )
    parser.add_argument(
        "--data_dir", type=str, required=True, help="Directory containing test images"
    )
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to trained model weights"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save predicted masks",
    )
    parser.add_argument("--bsz", type=int, default=32, help="Batch size for testing")

    args = parser.parse_args()
    main(args)