import torch
import torch.nn as nn
import torch.nn.functional as F
from config import logging

class SCConv(nn.Module):
    """
    Spatial and Channel Reconstruction Convolution (SCConv)
    - SRU: Spatial Reconstruction Unit
    - CRU: Channel Reconstruction Unit
    """
    def __init__(self, in_channels, out_channels):
        super(SCConv, self).__init__()
        # Spatial Reconstruction Unit (SRU)
        self.spatial_reconstruction = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels)
        # Channel Reconstruction Unit (CRU)
        self.channel_reconstruction = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.SiLU()
        )

    def forward(self, x):
        if logging:
            print(f"  [SCConv] Input: {x.shape}")
        x = self.spatial_reconstruction(x)
        x = self.channel_reconstruction(x)
        if logging:
            print(f"  [SCConv] Output: {x.shape}")
        return x


class DWConv(nn.Module):
    """
    Depthwise Separable Convolution (DWConv)
    - Efficient feature compression
    """
    def __init__(self, in_channels, out_channels):
        super(DWConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        if logging:
            print(f"  [DWConv] Input: {x.shape}")
        x = self.depthwise(x)
        x = self.pointwise(x)
        if logging:
            print(f"  [DWConv] Output: {x.shape}")
        return x


class SDConv(nn.Module):
    """
    SD-Conv: Combination of SCConv and DWConv with dilated convolutions
    """
    def __init__(self, in_channels, out_channels, dilation=1):
        super(SDConv, self).__init__()
        self.scconv = SCConv(in_channels, out_channels)
        self.dwconv = DWConv(out_channels, out_channels)
        self.dilated_conv = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=dilation, dilation=dilation)

    def forward(self, x):
        if logging:
            print(f" [SDConv] Dilation {self.dilated_conv.dilation[0]} - Input: {x.shape}")
        x = self.scconv(x)
        x = self.dwconv(x)
        x = self.dilated_conv(x)
        return x


class SDSSMBlock(nn.Module):
    """
    SD-SSM Block: Combines two branches (X1 and X2) for local and global feature extraction
    """
    def __init__(self, in_channels, out_channels):
        super(SDSSMBlock, self).__init__()
        
        # X1 Branch: Local feature extraction with multi-scale SD-Conv
        self.bn1 = nn.BatchNorm2d(in_channels // 2)
        self.sdconv1 = SDConv(in_channels // 2, out_channels, dilation=1)
        self.sdconv2 = SDConv(out_channels, out_channels, dilation=2)
        self.sdconv3 = SDConv(out_channels, out_channels, dilation=3)

        # X2 Branch: Global feature extraction with LayerNorm, Linear, SD-Conv, and SS2D
        self.layer_norm = nn.LayerNorm(in_channels // 2)  # Normalize along the channel axis
        self.linear1 = nn.Linear(in_channels // 2, in_channels // 2)
        self.silu = nn.SiLU()
        self.sdconv4 = SDConv(in_channels // 2, out_channels, dilation=2)
        self.ss2d = nn.Conv2d(out_channels, out_channels, kernel_size=1)

        # Fusion: Combine features from both branches
        self.fusion_linear = nn.Linear(out_channels, out_channels)
        self.final_layer_norm = nn.LayerNorm(out_channels)
        self.output_conv = nn.Conv2d(out_channels, out_channels, kernel_size=1)

        self.residual_proj = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        if logging:
            print(f"\n[SDSSMBlock] === Start Block (Input: {x.shape}) ===")
        
        # Split input into two branches
        # CRITICAL: x1 and x2 will have in_channels // 2
        x1, x2 = torch.chunk(x, 2, dim=1) 
        if logging:
            print(f"[SDSSMBlock] After Chunk - x1: {x1.shape}, x2: {x2.shape}")

        # X1 Branch
        if logging:
            print(f"[SDSSMBlock] Processing X1 Branch...")
        x1 = self.bn1(x1)
        x1 = self.sdconv1(x1)
        x1 = self.sdconv2(x1)
        x1 = self.sdconv3(x1)
        if logging:
            print(f"[SDSSMBlock] X1 Branch Final: {x1.shape}")

        # X2 Branch
        if logging:
            print(f"[SDSSMBlock] Processing X2 Branch...")
        x2 = self.layer_norm(x2.permute(0, 2, 3, 1))       # layer norm is done on the last dim (this is layer norm pixel wise not layer wise)
        x2 = self.linear1(x2)  # (B, H, W, C)
        x2 = self.silu(x2).permute(0, 3, 1, 2)    # (B, C, H, W)
        x2 = self.sdconv4(x2)
        x2 = self.ss2d(x2)
        if logging:
            print(f"[SDSSMBlock] X2 Branch Final: {x2.shape}")

        # Fusion
        fused = x1 + x2 
        if logging:
            print(f"[SDSSMBlock] Fused (x1+x2): {fused.shape}")
        
        fused = self.fusion_linear(fused.permute(0, 2, 3, 1))
        fused = self.final_layer_norm(fused).permute(0, 3, 1, 2)
        output = self.output_conv(fused)
        if logging:
            print(f"[SDSSMBlock] Before Residual Addition: {output.shape}")

        # Residual connection
        # Check if output channels match original x channels
        if output.shape != x.shape:
            if logging:
                print(f" [WARNING] Shape mismatch for residual! output: {output.shape}, original x: {x.shape}")
        
        output += self.residual_proj(x)
        if logging:
            print(f"[SDSSMBlock] === End Block (Output: {output.shape}) ===\n")
        return output


