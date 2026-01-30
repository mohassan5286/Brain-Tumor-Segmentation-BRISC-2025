import torch
import torch.nn as nn
from sd_ssm_block import SDSSMBlock  # 引入 SDSSMBlock 模块

class LinearEmbedding(nn.Module):
    def __init__(self, in_channels, embed_dim):
        super(LinearEmbedding, self).__init__()
        self.linear_proj = nn.Conv2d(in_channels, embed_dim, kernel_size=1)

    def forward(self, x):
        out = self.linear_proj(x)
        print(f"[LinearEmbedding] Input: {x.shape} -> Output: {out.shape}")
        return out

class PatchMerging(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(PatchMerging, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
        self.norm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        print(f"[PatchMerging] Input: {x.shape}")
        
        x = self.conv(x)
        print(f"[PatchMerging] After Conv (stride 2): {x.shape}")
        
        x = self.norm(x)
        return x

class PatchExpanding(nn.Module):
    """
    Patch Expanding: Increases spatial resolution and halves the feature channels.
    """
    def __init__(self, in_channels, out_channels):
        super(PatchExpanding, self).__init__()
        # Using a stride of 2 to upscale the image
        self.deconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x):
        print(f"[PatchExpanding] Input shape: {x.shape}")
        x = self.deconv(x)
        print(f"[PatchExpanding] Output shape: {x.shape}")
        return x

class MUNet(nn.Module):
    """
    MUNet: Combines encoder-decoder architecture with SD-SSM Blocks.
    """
    def __init__(self, in_channels=1, out_channels=1, embed_dim=64, num_layers=4):
        """
        Args:
            in_channels: Number of input channels (e.g., 1 for grayscale images).
            out_channels: Number of output channels (e.g., 1 for binary segmentation).
            embed_dim: Initial embedding dimension (number of features after LinearEmbedding).
            num_layers: Number of encoder and decoder layers.
        """
        super(MUNet, self).__init__()

        # Linear Embedding
        self.linear_embedding = LinearEmbedding(in_channels, embed_dim)

        # Encoder: SD-SSM Blocks + Patch Merging
        self.encoder_blocks = nn.ModuleList()
        self.patch_merging_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.encoder_blocks.append(SDSSMBlock(embed_dim, embed_dim))
            if _ != num_layers - 1:  # No patch merging after the last layer
                self.patch_merging_layers.append(PatchMerging(embed_dim, embed_dim * 2))
                embed_dim *= 2

        # Bottleneck: SD-SSM Block
        self.bottleneck = SDSSMBlock(embed_dim, embed_dim)
        
        # embed_dim *= 2
        
        # Decoder: Patch Expanding + SD-SSM Blocks
        self.decoder_blocks = nn.ModuleList()
        self.patch_expanding_layers = nn.ModuleList()
        for _ in range(num_layers - 1, -1, -1):
            if _ != num_layers - 1:  # No patch expanding before the first decoder layer
                embed_dim //= 2
                self.patch_expanding_layers.append(PatchExpanding(embed_dim * 2, embed_dim))
            self.decoder_blocks.append(SDSSMBlock(embed_dim, embed_dim))

        # Final Conv Layer
        self.final_conv = nn.Conv2d(embed_dim, out_channels, kernel_size=1)

    def forward(self, x):
        print(f"\n>>> [MUNet] Starting Forward Pass")
        print(f"[MUNet] Input shape: {x.shape}")
        
        # Linear Embedding
        x = self.linear_embedding(x)
        print(f"[MUNet] After Linear Embedding: {x.shape}")

        # Encoder
        skip_connections = []
        for i, block in enumerate(self.encoder_blocks):
            x = block(x)
            print(f"[MUNet] Encoder Block {i} output: {x.shape}")
            
            if i < len(self.patch_merging_layers):
                skip_connections.append(x)
                x = self.patch_merging_layers[i](x)
                print(f"[MUNet] After Patch Merging {i}: {x.shape}")

        # Bottleneck
        x = self.bottleneck(x)
        print(f"[MUNet] Bottleneck output: {x.shape}")

        # Decoder
        skip_connections = skip_connections[::-1]  # Reverse skip connections
        for i, block in enumerate(self.decoder_blocks):
            if i > 0:
                x = self.patch_expanding_layers[i - 1](x)
                print(f"[MUNet] Decoder Layer {i} - After Expanding: {x.shape}")
            
                skip_connection = skip_connections[i]
                
                # Check for shape mismatch before concatenation
                if x.shape != skip_connection.shape:
                    print(f"  [MUNet WARNING] Shape mismatch! x: {x.shape}, skip: {skip_connection.shape}. Interpolating...")
                    x = nn.functional.interpolate(x, size=skip_connection.shape[2:])
                
                # Concatenation
                x = torch.cat((x, skip_connection), dim=1)
                print(f"[MUNet] Decoder Layer {i} - After Concatenation: {x.shape}")
            
            # Pass through the actual decoder block
            x = block(x) 
            print(f"[MUNet] Decoder Layer {i} - After Block: {x.shape}")

        # Final Conv Layer
        x = self.final_conv(x)
        print(f"[MUNet] Final Output shape: {x.shape}")
        print(f">>> [MUNet] Forward Pass Complete\n")
        return x