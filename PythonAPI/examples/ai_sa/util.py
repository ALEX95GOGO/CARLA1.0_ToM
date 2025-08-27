import os
import subprocess
from typing import Optional, Tuple
import numpy as np
import torch
import torch.nn.functional as F
import cv2  


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

def get_avaliable_gpu():

    try:
        result = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,nounits,noheader'],
            encoding='utf-8'
        )
        memory_used = [int(x) for x in result.strip().split('\n') if x.strip().isdigit()]
        
        if not memory_used:
            return 'cpu'
        
        best_gpu = min(range(len(memory_used)), key=lambda i: memory_used[i])
        return f'cuda:{best_gpu}'
    
    except FileNotFoundError:
        return 'cpu'
    except subprocess.CalledProcessError:
        return 'cpu'
    except Exception as e:
        print(f"Failed to get GPU info, use CPU: {e}")
        return 'cpu'

def read_rgb_image_strict(path: str) -> np.ndarray:
        """
        Read an image from disk and return an RGB uint8 numpy array [H, W, 3].
        Raises FileNotFoundError or ValueError with absolute path details on failure.
        """
        abs_path = os.path.abspath(path)
        if not os.path.exists(abs_path):
            raise FileNotFoundError(f"Image not found: {abs_path}")
        img_bgr = cv2.imread(abs_path, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise ValueError(f"OpenCV cannot read the file (corrupted or unsupported): {abs_path}")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        if img_rgb.dtype != np.uint8:
            img_rgb = img_rgb.astype(np.uint8)
        return img_rgb

def _ensure_rgb_uint8(img: np.ndarray) -> np.ndarray:
    """
    Ensure the input is an RGB image in uint8 [H, W, 3].
    Raises ValueError if not satisfied.
    """
    if not isinstance(img, np.ndarray):
        raise ValueError("Input frame must be a numpy array.")
    if img.ndim != 3:
        raise ValueError("Input frame must have 3 dimensions (H, W, C).")
    if img.shape[2] not in [1, 3]:
        raise ValueError("Input frame channel must be 1 (grayscale) or 3 (RGB).")
    if img.dtype != np.uint8:
        raise ValueError("Input frame dtype must be uint8.")
    return img


def _to_tensor(img: np.ndarray) -> torch.Tensor:
    """
    Convert uint8 numpy image [H, W, C] to float tensor [C, H, W] in [0,1].
    """
    t = torch.from_numpy(img).float() / 255.0
    t = t.permute(2, 0, 1).contiguous()
    return t


def _resize_norm(t: torch.Tensor, size=(224, 224)) -> torch.Tensor:
    """
    Resize to target size and apply ImageNet normalization.
    Input: [C, H, W] in [0,1]; Output: [C, 224, 224]
    """
    t = F.interpolate(t.unsqueeze(0), size=size, mode="bilinear", align_corners=False).squeeze(0)
    mean = torch.tensor(IMAGENET_MEAN, dtype=t.dtype, device=t.device).view(-1, 1, 1)
    std  = torch.tensor(IMAGENET_STD,  dtype=t.dtype, device=t.device).view(-1, 1, 1)
    return (t - mean) / std




def _to_rgb_uint8(img) -> np.ndarray:
    """
    Convert input to RGB uint8 numpy array [H, W, 3].
    Accepts torch.Tensor or np.ndarray in [C,H,W] or [H,W,C], any numeric dtype.
    If float, values are min-max normalized to [0,1] before casting.
    """
    if isinstance(img, torch.Tensor):
        img = img.detach().cpu().numpy()
    if img.ndim == 3 and img.shape[0] in (1, 3) and img.shape[0] != img.shape[2]:
        img = np.transpose(img, (1, 2, 0))  # [C,H,W] -> [H,W,C]
    if img.ndim != 3 or img.shape[2] not in (1, 3):
        raise ValueError("image must have shape [H,W,1/3] or [C,H,W] with C in {1,3}")

    # Normalize to [0,1] float
    if img.dtype == np.uint8:
        img_f = img.astype(np.float32) / 255.0
        #print(img.shape)
    else:
        img_f = img.astype(np.float32)
        vmin, vmax = float(img_f.min()), float(img_f.max())
        img_f = (img_f - vmin) / (vmax - vmin) if vmax > vmin else np.zeros_like(img_f)

    # Ensure RGB
    if img_f.shape[2] == 1:
        img_f = np.repeat(img_f, 3, axis=2)

    return (img_f * 255.0).clip(0, 255).astype(np.uint8)


def _attn_to_heat_rgb(attn, out_size: Tuple[int, int]) -> np.ndarray:
    """
    Convert 2D attention map to a colored heatmap (RGB uint8) using 'jet', resized to out_size (W,H).
    """
    if isinstance(attn, torch.Tensor):
        attn = attn.detach().cpu().numpy()
    if attn.ndim == 3:
        attn = attn.squeeze()
    if attn.ndim != 2:
        raise ValueError("attention map must be 2D [H,W] or [1,H,W]")

    attn = attn.astype(np.float32)
    vmin, vmax = float(attn.min()), float(attn.max())
    attn = (attn - vmin) / (vmax - vmin) if vmax > vmin else np.zeros_like(attn, dtype=np.float32)

    W, H = out_size
    attn_resized = cv2.resize(attn, (W, H), interpolation=cv2.INTER_LINEAR)
    heat_bgr = cv2.applyColorMap((attn_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heat_rgb = cv2.cvtColor(heat_bgr, cv2.COLOR_BGR2RGB)
    return heat_rgb


def overlay_attention(
    image,
    attn_map,
    alpha: float = 0.5,
    save_path: Optional[str] = None
) -> np.ndarray:
    """
    Overlay a 'jet' attention heatmap on the original image.

    Args:
        image:    RGB image [H,W,3] or [3,H,W] (torch or numpy).
        attn_map: Attention map [H,W] (or [1,H,W]); any dtype/range (auto-normalized).
        alpha:    Blend factor for heatmap (0..1).
        save_path:Optional path to save the overlay (PNG recommended).

    Returns:
        overlay:  RGB uint8 image [H,W,3] with attention heatmap overlay.
    """
    img = _to_rgb_uint8(image)
    H, W = img.shape[:2]
    heat = _attn_to_heat_rgb(attn_map, out_size=(W, H))

    overlay = (img.astype(np.float32) * (1.0 - alpha) + heat.astype(np.float32) * alpha)
    overlay = overlay.clip(0, 255).astype(np.uint8)

    if save_path is not None:
        dir_name = os.path.dirname(save_path)
        if dir_name: 
            os.makedirs(dir_name, exist_ok=True)
        cv2.imwrite(save_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    return overlay

