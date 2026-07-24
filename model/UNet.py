# ============================================================================
# UNet Architecture with Spatial Attention and Transformer - BRISC 2025
# ============================================================================
# This module implements a modified UNet architecture that combines:
# - Double convolution blocks for feature extraction
# - Spatial attention mechanisms for focus on relevant regions
# - Transformer decoder layers for capturing long-range dependencies
# ============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from .transformer import PositionEmbeddingSine, Transformer

# ============================================================================
# Basic Convolution Block
# ============================================================================


class DoubleConv(nn.Module):
    """Two consecutive 3x3 convolutions with batch norm and ReLU activation."""

    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        # Double 3x3 convolution block: Conv -> BN -> ReLU -> Conv -> BN -> ReLU
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


# ============================================================================
# Attention Mechanism
# ============================================================================


class SpatialAttention(nn.Module):
    """Spatial attention module to emphasize important image regions."""

    def __init__(self, in_channels, out_channel):
        super(SpatialAttention, self).__init__()
        # Compute attention scores via max pooling and convolution
        self.attention_score = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels, out_channel, kernel_size=1),
            nn.Sigmoid(),  # Sigmoid for attention weights in [0, 1]
        )
        # Process input features via convolution and pooling
        self.input = nn.Sequential(
            nn.Conv2d(in_channels, out_channel, kernel_size=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

    def forward(self, x):
        # Compute attention weights
        attention_score = self.attention_score(x)
        # Process input
        input = self.input(x)
        # Apply attention: element-wise multiplication with scores
        return torch.mul(attention_score, input)


# ============================================================================
# Full UNet Architecture
# ============================================================================


class UNet(nn.Module):
    """Modified UNet with spatial attention and transformer refinement for segmentation."""

    def __init__(self, in_channels, out_channels, feature_dims):
        super(UNet, self).__init__()

        # ====== ENCODER PATH ======
        # Progressive feature extraction with increasing channels
        self.encoder1 = DoubleConv(in_channels, feature_dims)
        self.encoder2 = DoubleConv(feature_dims * 2, feature_dims * 2)
        self.encoder3 = DoubleConv(feature_dims * 2, feature_dims * 4)
        self.encoder4 = DoubleConv(feature_dims * 8, feature_dims * 8)

        # Spatial attention modules at intermediate layers
        self.spatial_attention1 = SpatialAttention(feature_dims, feature_dims * 2)
        self.spatial_attention2 = SpatialAttention(feature_dims * 4, feature_dims * 8)

        # Max pooling for downsampling
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # ====== TRANSFORMER REFINEMENT LAYERS ======
        # Fully connected layers to compress transformer outputs
        self.fc_4 = nn.Sequential(
            nn.Linear(feature_dims * 16, feature_dims * 8), nn.ReLU(inplace=True)
        )

        self.fc_3 = nn.Sequential(
            nn.Linear(feature_dims * 8, feature_dims * 4), nn.ReLU(inplace=True)
        )

        self.fc_2 = nn.Sequential(
            nn.Linear(feature_dims * 4, feature_dims * 2), nn.ReLU(inplace=True)
        )

        self.fc_1 = nn.Sequential(
            nn.Linear(feature_dims * 2, feature_dims * 1), nn.ReLU(inplace=True)
        )

        # Bottleneck transformer components
        self.embed = nn.Embedding(feature_dims * 16, 1)
        self.pe_layer = PositionEmbeddingSine(feature_dims * 16 // 2, normalize=True)
        self.transformer = Transformer(
            d_model=feature_dims * 16,
            dropout=0.1,
            nhead=4,
            dim_feedforward=feature_dims * 4,
            num_encoder_layers=0,
            num_decoder_layers=1,
            normalize_before=False,
            return_intermediate_dec=False,
        )

        # Transformer refinement at decoder level 4
        self.embed_4 = nn.Embedding(feature_dims * 8, 1)
        self.pe_layer_4 = PositionEmbeddingSine(feature_dims * 8 // 2, normalize=True)
        self.transformer_4 = Transformer(
            d_model=feature_dims * 8,
            dropout=0.1,
            nhead=4,
            dim_feedforward=feature_dims * 4,
            num_encoder_layers=0,
            num_decoder_layers=1,
            normalize_before=False,
            return_intermediate_dec=False,
        )

        # Transformer refinement at decoder level 3
        self.embed_3 = nn.Embedding(feature_dims * 4, 1)
        self.pe_layer_3 = PositionEmbeddingSine(feature_dims * 4 // 2, normalize=True)
        self.transformer_3 = Transformer(
            d_model=feature_dims * 4,
            dropout=0.1,
            nhead=4,
            dim_feedforward=feature_dims * 4,
            num_encoder_layers=0,
            num_decoder_layers=1,
            normalize_before=False,
            return_intermediate_dec=False,
        )

        # Transformer refinement at decoder level 2
        self.embed_2 = nn.Embedding(feature_dims * 2, 1)
        self.pe_layer_2 = PositionEmbeddingSine(feature_dims * 2 // 2, normalize=True)
        self.transformer_2 = Transformer(
            d_model=feature_dims * 2,
            dropout=0.1,
            nhead=4,
            dim_feedforward=feature_dims * 4,
            num_encoder_layers=0,
            num_decoder_layers=1,
            normalize_before=False,
            return_intermediate_dec=False,
        )

        # Transformer refinement at decoder level 1
        self.embed_1 = nn.Embedding(feature_dims * 1, 1)
        self.pe_layer_1 = PositionEmbeddingSine(feature_dims * 1 // 2, normalize=True)
        self.transformer_1 = Transformer(
            d_model=feature_dims * 1,
            dropout=0.1,
            nhead=4,
            dim_feedforward=feature_dims * 4,
            num_encoder_layers=0,
            num_decoder_layers=1,
            normalize_before=False,
            return_intermediate_dec=False,
        )

        # ====== DECODER PATH ======
        # Bottleneck processing
        self.bottleneck_conv = DoubleConv(feature_dims * 8, feature_dims * 16)

        # Decoder blocks with upsampling and skip connections
        self.upconv4 = nn.ConvTranspose2d(
            feature_dims * 16, feature_dims * 8, kernel_size=2, stride=2
        )
        self.decoder4 = DoubleConv(feature_dims * 16, feature_dims * 8)
        self.upconv3 = nn.ConvTranspose2d(
            feature_dims * 8, feature_dims * 4, kernel_size=2, stride=2
        )
        self.decoder3 = DoubleConv(feature_dims * 8, feature_dims * 4)
        self.upconv2 = nn.ConvTranspose2d(
            feature_dims * 4, feature_dims * 2, kernel_size=2, stride=2
        )
        self.decoder2 = DoubleConv(feature_dims * 4, feature_dims * 2)
        self.upconv1 = nn.ConvTranspose2d(
            feature_dims * 2, feature_dims, kernel_size=2, stride=2
        )
        self.decoder1 = DoubleConv(feature_dims * 2, feature_dims)

        # Final output layer
        self.final_conv = nn.Conv2d(feature_dims, out_channels, kernel_size=1)

    def forward(self, x):
        # ====== ENCODER ======
        # Level 1: extract features
        enc1 = self.encoder1(x)
        # Apply spatial attention to encoder output
        attentioned_layer1 = self.spatial_attention1(enc1)
        # Level 2: downsample and extract features
        enc2 = self.encoder2(attentioned_layer1)
        # Level 3: with attention
        enc3 = self.encoder3(self.pool(enc2))
        attentioned_layer2 = self.spatial_attention2(enc3)
        # Level 4: deepest encoder level
        enc4 = self.encoder4(attentioned_layer2)

        # ====== BOTTLENECK ======
        # Downsample to bottleneck
        bottleneck = self.pool(enc4)
        bottleneck = self.bottleneck_conv(bottleneck)

        # Transformer refinement at bottleneck
        b, c, h, w = bottleneck.shape
        bottleneck_avg = bottleneck.mean(dim=(2, 3), keepdim=True).squeeze(-1)
        bottleneck_pe = self.pe_layer(bottleneck)

        transformer_output = self.transformer(
            bottleneck, None, bottleneck_avg, self.embed.weight, bottleneck_pe
        )
        transformer_output_expanded = transformer_output.unsqueeze(-1)
        bottleneck = bottleneck * transformer_output_expanded

        # ====== DECODER ======
        # Decoder level 4
        dec4 = self.upconv4(bottleneck)

        enc4_avg = self.fc_4(transformer_output.squeeze(-1)).unsqueeze(2)
        enc4_pe = self.pe_layer_4(enc4)
        transformer_output_4 = self.transformer_4(
            enc4, None, enc4_avg, self.embed_4.weight, enc4_pe
        )
        transformer_output_4_expanded = transformer_output_4.unsqueeze(-1)
        enc4 = enc4 * transformer_output_4_expanded

        # Concatenate upsampled features with encoder features
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)

        # Decoder level 3
        dec3 = self.upconv3(dec4)

        enc3_avg = self.fc_3(transformer_output_4.squeeze(-1)).unsqueeze(2)
        enc3_pe = self.pe_layer_3(enc3)
        transformer_output_3 = self.transformer_3(
            enc3, None, enc3_avg, self.embed_3.weight, enc3_pe
        )
        transformer_output_3_expanded = transformer_output_3.unsqueeze(-1)

        enc3 = enc3 * transformer_output_3_expanded

        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)

        # Decoder level 2
        dec2 = self.upconv2(dec3)

        enc2_avg = self.fc_2(transformer_output_3.squeeze(-1)).unsqueeze(2)
        enc2_pe = self.pe_layer_2(enc2)
        transformer_output_2 = self.transformer_2(
            enc2, None, enc2_avg, self.embed_2.weight, enc2_pe
        )
        transformer_output_2_expanded = transformer_output_2.unsqueeze(-1)
        enc2 = enc2 * transformer_output_2_expanded

        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)

        # Decoder level 1
        dec1 = self.upconv1(dec2)

        enc1_avg = self.fc_1(transformer_output_2.squeeze(-1)).unsqueeze(2)
        enc1_pe = self.pe_layer_1(enc1)
        transformer_output_1 = self.transformer_1(
            enc1, None, enc1_avg, self.embed_1.weight, enc1_pe
        )
        transformer_output_1_expanded = transformer_output_1.unsqueeze(-1)
        enc1 = enc1 * transformer_output_1_expanded

        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)

        # Final output with sigmoid activation for binary segmentation
        return torch.sigmoid(self.final_conv(dec1))
