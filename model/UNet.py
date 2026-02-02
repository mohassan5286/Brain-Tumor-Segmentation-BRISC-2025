import torch
import torch.nn as nn
import torch.nn.functional as F
from .transformer import PositionEmbeddingSine, Transformer
from torchvision import models
from torchvision.models.feature_extraction import create_feature_extractor

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, feature_dims, fine_tune=False):
        super(UNet, self).__init__()
        
        self.encoder1 = DoubleConv(in_channels, feature_dims)
        self.encoder2 = DoubleConv(feature_dims, feature_dims * 2)
        self.encoder3 = DoubleConv(feature_dims * 2, feature_dims * 4)
        self.encoder4 = DoubleConv(feature_dims * 4, feature_dims * 8)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.fc_4 = nn.Sequential(
            nn.Linear(feature_dims * 16, feature_dims * 8),  # Squeeze
            nn.ReLU(inplace=True))
        
        self.fc_3 = nn.Sequential(
            nn.Linear(feature_dims * 8, feature_dims * 4),  # Squeeze
            nn.ReLU(inplace=True))
        
        self.fc_2 = nn.Sequential(
            nn.Linear(feature_dims * 4, feature_dims * 2),  # Squeeze
            nn.ReLU(inplace=True))
        
        self.fc_1 = nn.Sequential(
            nn.Linear(feature_dims * 2, feature_dims * 1),  # Squeeze
            nn.ReLU(inplace=True))

        self.embed = nn.Embedding(768,1)
        self.pe_layer = PositionEmbeddingSine(768 // 2, normalize=True)
        self.transformer = Transformer(
            d_model=768,
            dropout=0.1,
            nhead=4,
            dim_feedforward=feature_dims * 4,
            num_encoder_layers=0,
            num_decoder_layers=1,
            normalize_before=False,
            return_intermediate_dec=False,
        )

        self.embed_4 = nn.Embedding(feature_dims * 8,1)
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

        self.embed_3 = nn.Embedding(feature_dims * 4,1)
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
        
        self.embed_2 = nn.Embedding(feature_dims * 2,1)
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
        
        self.embed_1 = nn.Embedding(feature_dims * 1,1)
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

        
        # self.bottleneck_conv = DoubleConv(feature_dims * 8, feature_dims * 16)

        full_swin = models.swin_t(weights='DEFAULT')
        
        if fine_tune:
            print('[INFO]: Fine-tuning all layers...')
            for params in full_swin.parameters():
                params.requires_grad = True
        if not fine_tune:
            print('[INFO]: Freezing hidden layers...')
            for params in full_swin.parameters():
                params.requires_grad = False
        
        self.swin_bottleneck = full_swin.features[7]

        self.before_swin_1x1_conv = nn.Conv2d(feature_dims * 8, 768, kernel_size=1, stride=1)
        self.after_swin_1x1_conv = nn.Conv2d(768, feature_dims * 16, kernel_size=1, stride=1)

        print(self.swin_bottleneck)
        
        del full_swin
        torch.cuda.empty_cache()
        
        self.upconv4 = nn.ConvTranspose2d(feature_dims * 16, feature_dims * 8, kernel_size=2, stride=2)
        self.decoder4 = DoubleConv(feature_dims * 16, feature_dims * 8)
        self.upconv3 = nn.ConvTranspose2d(feature_dims * 8, feature_dims * 4, kernel_size=2, stride=2)
        self.decoder3 = DoubleConv(feature_dims * 8, feature_dims * 4)
        self.upconv2 = nn.ConvTranspose2d(feature_dims * 4, feature_dims * 2, kernel_size=2, stride=2)
        self.decoder2 = DoubleConv(feature_dims * 4, feature_dims * 2)
        self.upconv1 = nn.ConvTranspose2d(feature_dims * 2, feature_dims, kernel_size=2, stride=2)
        self.decoder1 = DoubleConv(feature_dims * 2, feature_dims)
        
        self.final_conv = nn.Conv2d(feature_dims, out_channels, kernel_size=1)

    def forward(self, x, verbose=True):
        if verbose: print(f"Input: {x.shape}")

        # Encoder
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool(enc1))
        enc3 = self.encoder3(self.pool(enc2))
        enc4 = self.encoder4(self.pool(enc3))
        
        if verbose:
            print(f"Enc1: {enc1.shape} | Enc2: {enc2.shape} | Enc3: {enc3.shape} | Enc4: {enc4.shape}")

        # Bottleneck
        bottleneck = self.pool(enc4)
        if verbose: print(f"Bottleneck pre-transformer: {bottleneck.shape}")
        # bottleneck = self.bottleneck_conv(bottleneck)
        # if verbose: print(f"Bottleneck pre-transformer: {bottleneck.shape}")

        bottleneck = self.before_swin_1x1_conv(bottleneck) 
        if verbose: print(f"Before Bottleneck swin: {bottleneck.shape}")
        bottleneck = self.swin_bottleneck(bottleneck)
        if verbose: print(f"Bottleneck swin: {bottleneck.shape}")
        bottleneck = self.after_swin_1x1_conv(bottleneck)
        if verbose: print(f"After Bottleneck swin: {bottleneck.shape}")
        
        b, c, h, w = bottleneck.shape


        bottleneck_avg = bottleneck.mean(dim=(2, 3), keepdim=True).squeeze(-1)
        if verbose: print(f"Averaging output: {bottleneck_avg.shape}")
        bottleneck_pe = self.pe_layer(bottleneck)
        
        transformer_output = self.transformer(bottleneck, None, bottleneck_avg, self.embed.weight, bottleneck_pe) 
        if verbose: print(f"Transformer output: {transformer_output.shape}")

        transformer_output_expanded = transformer_output.unsqueeze(-1) 
        bottleneck = bottleneck * transformer_output_expanded
        
        # Decoder
        dec4 = self.upconv4(bottleneck)
        if verbose: print(f"Dec4 (after upconv): {dec4.shape}")

        enc4_avg = self.fc_4(transformer_output.squeeze(-1)).unsqueeze(2)
        enc4_pe = self.pe_layer_4(enc4)
        transformer_output_4 = self.transformer_4(enc4, None, enc4_avg, self.embed_4.weight, enc4_pe) 
        transformer_output_4_expanded = transformer_output_4.unsqueeze(-1) 
        enc4 = enc4 * transformer_output_4_expanded

        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        if verbose: print(f"Dec4 (after cat/conv): {dec4.shape}")

        dec3 = self.upconv3(dec4)
        enc3_avg = self.fc_3(transformer_output_4.squeeze(-1)).unsqueeze(2)
        enc3_pe = self.pe_layer_3(enc3)
        transformer_output_3 = self.transformer_3(enc3, None, enc3_avg, self.embed_3.weight, enc3_pe) 
        transformer_output_3_expanded = transformer_output_3.unsqueeze(-1) 
        enc3 = enc3 * transformer_output_3_expanded

        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        if verbose: print(f"Dec3: {dec3.shape}")

        dec2 = self.upconv2(dec3)
        enc2_avg = self.fc_2(transformer_output_3.squeeze(-1)).unsqueeze(2)
        enc2_pe = self.pe_layer_2(enc2)
        transformer_output_2 = self.transformer_2(enc2, None, enc2_avg, self.embed_2.weight, enc2_pe) 
        transformer_output_2_expanded = transformer_output_2.unsqueeze(-1) 
        enc2 = enc2 * transformer_output_2_expanded

        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        if verbose: print(f"Dec2: {dec2.shape}")

        dec1 = self.upconv1(dec2)
        enc1_avg = self.fc_1(transformer_output_2.squeeze(-1)).unsqueeze(2)
        enc1_pe = self.pe_layer_1(enc1)
        transformer_output_1 = self.transformer_1(enc1, None, enc1_avg, self.embed_1.weight, enc1_pe) 
        transformer_output_1_expanded = transformer_output_1.unsqueeze(-1) 
        enc1 = enc1 * transformer_output_1_expanded

        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        
        final_out = torch.sigmoid(self.final_conv(dec1))
        if verbose: print(f"Final Output: {final_out.shape}")
        
        return final_out