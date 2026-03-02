# ============================================================================
# Brain Tumor Segmentation Testing/Inference Script - BRISC 2025
# ============================================================================
# This script performs inference on test images using a trained UNet model
# and generates binary segmentation masks for brain tumors.
# ============================================================================

import os
import argparse
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
from data.data_loader_segmentation import get_test_loader
from model.UNet import UNet

# ============================================================================
# Mask Generation Function
# ============================================================================


def generate_masks(test_loader, model, output_dir):
    """Generate and save segmentation masks for test images."""
    # Set model to evaluation mode
    model.eval()

    # Move model to GPU if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Inference on test set without gradient computation
    with torch.no_grad():
        for i, (inputs, _) in enumerate(test_loader):
            # Move batch to device
            inputs = inputs.to(device)
            # Get model predictions
            outputs = model(inputs)
            # Apply threshold to get binary predictions
            predictions = (
                outputs > 0.5
            ).float()  # Threshold of 0.5 for binary prediction

            # Process each image in the batch
            for j in range(inputs.size(0)):
                # Extract mask and convert to numpy
                pred_mask = predictions[j].cpu().numpy().squeeze()
                # Get original image name from dataset
                image_name = test_loader.dataset.images[i * test_loader.batch_size + j]
                # Create output path with _mask suffix
                save_path = os.path.join(
                    output_dir, image_name.replace(".jpg", "_mask.png")
                )

                # Convert to uint8 and save as PNG
                pred_mask = (pred_mask * 255).astype(np.uint8)
                Image.fromarray(pred_mask).save(save_path)

                print(f"Saved {save_path}")


# ============================================================================
# Main Inference Function
# ============================================================================


def main(args):
    """Load model and generate masks for all test images."""
    # Load test data
    test_loader = get_test_loader(args.data_dir, args.bsz)

    # Instantiate the UNet model with specified architecture
    model = UNet(in_channels=1, out_channels=1, feature_dims=64)
    # Load pre-trained weights
    model.load_state_dict(torch.load(args.model_path))

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Generate masks using the trained model
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
    parser.add_argument("--bsz", type=int, default=8, help="Batch size for testing")

    args = parser.parse_args()
    main(args)
