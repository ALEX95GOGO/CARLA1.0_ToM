
import os
import torch
from torchvision.models import resnet18
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class CNNBackbone(nn.Module):
    def __init__(self, task: str = 'pred', target_size=(224, 224)) -> None:
        super().__init__()
        base_model = resnet18(pretrained=True)

        # adjust the first convolution layer based on the task
        if task == 'pred':
            base_model.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        else:
            base_model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # extract the stages of ResNet
        self.stage1 = nn.Sequential(base_model.conv1, base_model.bn1, base_model.relu)  # [B, 64, 112, 112]
        self.stage2 = nn.Sequential(base_model.maxpool, base_model.layer1)              # [B, 64, 56, 56]
        self.stage3 = base_model.layer2                                                 # [B, 128, 28, 28]
        self.stage4 = base_model.layer3                                                 # [B, 256, 14, 14]
        self.stage5 = base_model.layer4                                                 # [B, 512, 7, 7]

        # 1x1 convolution layers to reduce feature maps to 1 channel
        self.reduce_f3 = nn.Conv2d(128, 1, kernel_size=1)
        self.reduce_f4 = nn.Conv2d(256, 1, kernel_size=1)
        self.reduce_f5 = nn.Conv2d(512, 1, kernel_size=1)

        # normalization layer
        self.sigmoid = nn.Sigmoid()
        self.target_size = target_size

    def forward(self, x: torch.Tensor) -> list:
        f1 = self.stage1(x)  # [B, 64, 112, 112]
        f2 = self.stage2(f1) # [B, 64, 56, 56]
        f3 = self.stage3(f2) # [B, 128, 28, 28]
        f4 = self.stage4(f3) # [B, 256, 14, 14]
        f5 = self.stage5(f4) # [B, 512, 7, 7]
        return [f1, f2, f3, f4, f5]

    def forward_with_attention(self, x: torch.Tensor) -> tuple:
        features = self.forward(x)

        # Generate Attention Map based on the last three feature maps
        attention_maps = []

        # process the last three feature maps
        for idx, (f, reduce_layer) in enumerate(zip(features[-3:], [self.reduce_f3, self.reduce_f4, self.reduce_f5])):
            if idx == 2:  
                upsampled = F.interpolate(f, size=self.target_size, mode='bilinear', align_corners=False)
                upsampled = reduce_layer(upsampled)
            else:
                reduced = reduce_layer(f)
                upsampled = F.interpolate(reduced, size=self.target_size, mode='bilinear', align_corners=False)

            normalized_map = self.sigmoid(upsampled)

            attention_maps.append(normalized_map)

        merged_attention = torch.mean(torch.stack(attention_maps, dim=0), dim=0)  # [B, 1, 224, 224]

        return features, merged_attention


class SequenceEncoder_human_sa(nn.Module):
    def __init__(self, feature_dim: int = 128, task: str = 'pred', gradient_detach=False) -> None:
        super().__init__()
        self.task = task
        self.gradient_detach = gradient_detach

        # CNN backbone to extract features
        self.cnn = CNNBackbone(task=self.task)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, feature_dim)

        # Temporal transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=feature_dim, nhead=4)
        self.temporal = nn.TransformerEncoder(encoder_layer, num_layers=2)
        temporal_out_dim = feature_dim

        # Final binary risk prediction head
        self.fc_risk = nn.Sequential(
            nn.Linear(temporal_out_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, list]:
        """
        Args:
            x: Tensor [B, T, C, H, W]
        Returns:
            risk_pred: Tensor [B, 1]
            attention_maps: list of attention features from CNN
        """
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        
        if self.gradient_detach:
            skip_feats_all, attention_maps = self.cnn.forward_with_attention(x)
            skip_feats_all_detached = [f.detach() for f in skip_feats_all]
            skip_feats = [feat.view(B, T, *feat.shape[1:]) for feat in skip_feats_all_detached]
        else:
            skip_feats_all, attention_maps = self.cnn.forward_with_attention(x)
            skip_feats = [feat.view(B, T, *feat.shape[1:]) for feat in skip_feats_all]


        pooled_feats = []
        for t in range(T):
            f5_t = skip_feats[-1][:, t]  # [B, 512, H, W]
            pooled = self.pool(f5_t).view(B, -1)
            feat = self.fc(pooled)
            pooled_feats.append(feat)

        feat_seq = torch.stack(pooled_feats, dim=1)  # [B, T, feature_dim]       
        feat_seq = feat_seq.permute(1, 0, 2)
        out = self.temporal(feat_seq)
        final_feat = out[-1]  # [B, feature_dim]

        risk_pred = self.fc_risk(final_feat)  # [B, 1]
        return risk_pred, attention_maps
    
    def load_model(self, weights_path, device='cpu'):
        """
        Load model weights from a file.
        
        Args:
            weights_path: Path to the model weights file.
            device: Optional device string, e.g., 'cuda:0' or 'cpu'.
        """
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Weights not found: {weights_path}")
        state = torch.load(weights_path, map_location=device)
        self.load_state_dict(state)
        self.to(device)
        self.eval()

    def save_model(self, save_path: str):
        """
        Save model weights to a file.
        
        Args:
            save_path: Path to save the model weights.
        """
        torch.save(self.state_dict(), save_path)
        print(f"Model saved to {save_path}")
