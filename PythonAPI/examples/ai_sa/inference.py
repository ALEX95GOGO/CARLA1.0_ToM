from typing import List, Tuple, Optional
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

@torch.no_grad()
def infer_sequence(model: torch.nn.Module,
                   frames: List[np.ndarray],
                   task: str = "pred",
                   device: Optional[str] = None) -> Tuple[float, List[np.ndarray]]:
    """
    Run inference on a sequence of 3 frames.

    Args:
        model: SequenceEncoder_human_sa instance (eval mode recommended).
        frames: list of length 3; each is numpy uint8 image [H, W, C] (C=3 for RGB).
        task: 'pred' expects 3-channel input; any other task means grayscale (1-channel).
        device: optional explicit device string, e.g., 'cuda:0' or 'cpu'.

    Returns:
        risk_value: float scalar.
        attn_maps: list of length 3; each is numpy float array [224, 224] in [0,1].
    """
    if len(frames) != 3:
        raise ValueError("Exactly 3 frames are required.")

    target_hw = (224, 224)
 
    images = torch.stack(frames, dim=0).unsqueeze(0).to(device)  # [1,3,3,H,W]

    risk_pred, merged_attention = model(images)
    risk_value = float(risk_pred.squeeze().item())

    # Reshape merged attention into per-frame maps
    B, T = 1, 3
    attn = merged_attention.view(B, T, 1, target_hw[0], target_hw[1])  # [1,3,1,224,224]
    attn = attn.squeeze(0).squeeze(1).detach().cpu().numpy()  # [3,224,224]
    attn = np.clip(attn, 0.0, 1.0)

    return risk_value, [attn[i] for i in range(T)]