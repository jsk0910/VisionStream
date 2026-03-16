"""
Super Resolution Adapter — Real-ESRGAN / SwinIR wrappers for BaseVisionModel.

Supported models:
  - real_esrgan: Real-ESRGAN via realesrgan pip package (xinntao/Real-ESRGAN)
  - swinir:      Swin2SR via HuggingFace transformers (caidas/swin2SR-*)

Both models fall back to bicubic interpolation if their dependencies are not installed.
"""
import torch
import numpy as np
from typing import Any, Optional
from modules.registry import BaseVisionModel, register_vision_model

# ── Optional dependency checks ───────────────────────────
try:
    from realesrgan import RealESRGANer
    from basicsr.archs.rrdbnet_arch import RRDBNet
    HAS_REALESRGAN = True
except ImportError:
    HAS_REALESRGAN = False

try:
    from transformers import AutoImageProcessor, Swin2SRForImageSuperResolution
    HAS_SWIN2SR = True
except ImportError:
    HAS_SWIN2SR = False


@register_vision_model("real_esrgan")
class RealESRGANModel(BaseVisionModel):
    """Wrapper for Real-ESRGAN (Super Resolution).

    Dependencies:
        pip install realesrgan basicsr

    The model weights are downloaded automatically on first use
    (~64 MB for x4plus, cached in ~/.cache).
    """

    def __init__(self, scale: int = 4, model_name: str = "RealESRGAN_x4plus",
                 device: str = "cuda:0", half: bool = False, **kwargs):
        self.device = device
        self.scale = scale
        self.model = None

        if HAS_REALESRGAN:
            # Build RRDBNet architecture matching the official pretrained weights
            net = RRDBNet(
                num_in_ch=3, num_out_ch=3, num_feat=64,
                num_block=23, num_grow_ch=32, scale=scale,
            )
            self.model = RealESRGANer(
                scale=scale,
                model_path=None,  # auto-download from GitHub Releases
                dni_weight=None,
                model=net,
                tile=0,
                tile_pad=10,
                pre_pad=0,
                half=half,
                device=device,
            )
            # Download weights if model_path was None
            try:
                import os
                from basicsr.utils.download_util import load_file_from_url
                url_map = {
                    "RealESRGAN_x4plus": (
                        "https://github.com/xinntao/Real-ESRGAN/releases/download/"
                        "v0.1.0/RealESRGAN_x4plus.pth"
                    ),
                    "RealESRGAN_x2plus": (
                        "https://github.com/xinntao/Real-ESRGAN/releases/download/"
                        "v0.2.1/RealESRGAN_x2plus.pth"
                    ),
                }
                if model_name in url_map:
                    model_dir = os.path.join(
                        os.path.expanduser("~"), ".cache", "realesrgan"
                    )
                    os.makedirs(model_dir, exist_ok=True)
                    model_path = load_file_from_url(
                        url_map[model_name],
                        model_dir=model_dir,
                    )
                    loadnet = torch.load(model_path, map_location=torch.device(device))
                    if "params_ema" in loadnet:
                        keyname = "params_ema"
                    elif "params" in loadnet:
                        keyname = "params"
                    else:
                        keyname = None
                    if keyname:
                        net.load_state_dict(loadnet[keyname], strict=True)
                    else:
                        net.load_state_dict(loadnet, strict=True)
                    net.eval()
                    net.to(device)
            except Exception as e:
                print(f"[RealESRGAN] Could not load weights: {e}")
                print("[RealESRGAN] Falling back to bicubic interpolation.")
                self.model = None

    def predict(self, x: torch.Tensor) -> Any:
        """
        Super-resolve input image(s).

        Args:
            x: [B, C, H, W] or [C, H, W] float tensor in [0, 1].

        Returns:
            [B, C, H*scale, W*scale] float tensor in [0, 1].
        """
        if x.ndim == 3:
            x = x.unsqueeze(0)

        if self.model is None:
            # Fallback: bicubic interpolation
            return torch.nn.functional.interpolate(
                x, scale_factor=self.scale, mode="bicubic", align_corners=False
            ).clamp(0, 1)

        # RealESRGANer expects BGR numpy uint8 [H, W, C]
        results = []
        for i in range(x.shape[0]):
            img_np = (x[i].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            img_bgr = img_np[:, :, ::-1]  # RGB -> BGR
            output_bgr, _ = self.model.enhance(img_bgr, outscale=self.scale)
            output_rgb = output_bgr[:, :, ::-1].copy()  # BGR -> RGB
            out_tensor = torch.from_numpy(output_rgb).float().permute(2, 0, 1) / 255.0
            results.append(out_tensor)

        return torch.stack(results).to(x.device)

    def get_task_type(self) -> str:
        return "super_resolution"


@register_vision_model("swinir")
class SwinIRModel(BaseVisionModel):
    """Wrapper for Swin2SR (Super Resolution) from HuggingFace.

    Dependencies:
        pip install transformers

    Uses 'caidas/swin2SR-classical-sr-x4-64' by default.
    Model weights are downloaded automatically from HuggingFace Hub.
    """

    def __init__(self, model_variant: str = "caidas/swin2SR-classical-sr-x4-64",
                 device: str = "cuda:0", **kwargs):
        self.device = device
        self.model = None
        self.processor = None

        if HAS_SWIN2SR:
            try:
                self.processor = AutoImageProcessor.from_pretrained(model_variant)
                self.model = Swin2SRForImageSuperResolution.from_pretrained(
                    model_variant
                ).to(device)
                self.model.eval()
            except Exception as e:
                print(f"[SwinIR] Could not load model: {e}")
                print("[SwinIR] Falling back to bicubic interpolation.")
                self.model = None
                self.processor = None

    def predict(self, x: torch.Tensor) -> Any:
        """
        Super-resolve input image(s).

        Args:
            x: [B, C, H, W] or [C, H, W] float tensor in [0, 1].

        Returns:
            [B, C, H*scale, W*scale] float tensor in [0, 1].
        """
        if x.ndim == 3:
            x = x.unsqueeze(0)

        if self.model is None or self.processor is None:
            return torch.nn.functional.interpolate(
                x, scale_factor=4, mode="bicubic", align_corners=False
            ).clamp(0, 1)

        results = []
        for i in range(x.shape[0]):
            # Processor expects PIL or numpy HWC uint8
            img_np = (x[i].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            inputs = self.processor(images=img_np, return_tensors="pt").to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                output = outputs.reconstruction

            # Clamp and move to [0, 1]
            output = output.squeeze(0).clamp(0, 255) / 255.0
            results.append(output.cpu())

        return torch.stack(results).to(x.device)

    def get_task_type(self) -> str:
        return "super_resolution"
