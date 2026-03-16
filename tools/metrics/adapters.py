"""
Metric Adapters — LPIPS, mAP (COCO), and FID with BaseMetric interface.

These metrics require optional dependencies:
  - LPIPS: pip install lpips
  - mAP:   pip install pycocotools
  - FID:   pip install pytorch-fid  (or torch + scipy)
"""
import torch
import numpy as np
from typing import Any, Optional
from modules.registry import BaseMetric, register_metric

# ── Optional dependency checks ───────────────────────────
try:
    import lpips as _lpips_module
    HAS_LPIPS = True
except ImportError:
    HAS_LPIPS = False

try:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    HAS_COCOTOOLS = True
except ImportError:
    HAS_COCOTOOLS = False

try:
    from scipy.linalg import sqrtm
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


# ═══════════════════════════════════════════════════════════════
#  LPIPS — Learned Perceptual Image Patch Similarity
# ═══════════════════════════════════════════════════════════════

@register_metric("lpips")
class LPIPSMetric(BaseMetric):
    """Learned Perceptual Image Patch Similarity (lower is better).

    Uses AlexNet features by default. Requires `pip install lpips`.
    """

    def __init__(self, net: str = "alex", device: str = "cuda:0", **kwargs):
        self.device = device
        self._fn = None

        if HAS_LPIPS:
            self._fn = _lpips_module.LPIPS(net=net).to(device)
            self._fn.eval()
        else:
            print("[LPIPSMetric] lpips not installed. Run: pip install lpips")

    def compute(self, original: torch.Tensor, reconstructed: torch.Tensor,
                **kwargs) -> float:
        """
        Compute LPIPS between original and reconstructed images.

        Args:
            original:      [B, C, H, W] float [0, 1]
            reconstructed: [B, C, H, W] float [0, 1]

        Returns:
            Average LPIPS score (scalar). Lower = more similar.
        """
        if self._fn is None:
            return 0.0

        if original.ndim == 3:
            original = original.unsqueeze(0)
        if reconstructed.ndim == 3:
            reconstructed = reconstructed.unsqueeze(0)

        # LPIPS expects [-1, 1] range
        orig_norm = original.to(self.device) * 2 - 1
        recon_norm = reconstructed.to(self.device) * 2 - 1

        with torch.no_grad():
            score = self._fn(orig_norm, recon_norm)

        return score.mean().item()

    def name(self) -> str:
        return "LPIPS"

    def higher_is_better(self) -> bool:
        return False


# ═══════════════════════════════════════════════════════════════
#  mAP — COCO-style Mean Average Precision
# ═══════════════════════════════════════════════════════════════

@register_metric("map")
class COCOmAPMetric(BaseMetric):
    """COCO-style Mean Average Precision for object detection.

    Requires `pip install pycocotools`.

    Usage:
        metric = COCOmAPMetric(ann_file="path/to/instances_val2017.json")
        score = metric.compute(original, reconstructed,
                               predictions=[{...}, ...],
                               image_ids=[...])
    """

    def __init__(self, ann_file: Optional[str] = None, **kwargs):
        self.ann_file = ann_file
        self._coco_gt = None

        if ann_file and HAS_COCOTOOLS:
            try:
                self._coco_gt = COCO(ann_file)
            except Exception as e:
                print(f"[COCOmAPMetric] Failed to load annotations: {e}")
        elif not HAS_COCOTOOLS:
            print("[COCOmAPMetric] pycocotools not installed. "
                  "Run: pip install pycocotools")

    def compute(self, original: torch.Tensor, reconstructed: torch.Tensor,
                predictions: Optional[list] = None,
                image_ids: Optional[list] = None,
                **kwargs) -> float:
        """
        Compute COCO mAP from detection predictions.

        Args:
            original: Not used directly (kept for interface compatibility).
            reconstructed: Not used directly.
            predictions: List of dicts in COCO result format:
                [{"image_id": int, "category_id": int,
                  "bbox": [x, y, w, h], "score": float}, ...]
            image_ids: Optional list of image IDs to evaluate.

        Returns:
            mAP@[.5:.95] score. Returns 0.0 if dependencies or data are missing.
        """
        if self._coco_gt is None or predictions is None:
            return 0.0

        if not predictions:
            return 0.0

        try:
            # Create COCO results object
            coco_dt = self._coco_gt.loadRes(predictions)

            # Run evaluation
            coco_eval = COCOeval(self._coco_gt, coco_dt, "bbox")
            if image_ids:
                coco_eval.params.imgIds = image_ids
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()

            # Return mAP@[.5:.95] (first stat)
            return float(coco_eval.stats[0])

        except Exception as e:
            print(f"[COCOmAPMetric] Evaluation error: {e}")
            return 0.0

    def name(self) -> str:
        return "mAP@[.5:.95]"

    def higher_is_better(self) -> bool:
        return True


# ═══════════════════════════════════════════════════════════════
#  FID — Fréchet Inception Distance
# ═══════════════════════════════════════════════════════════════

@register_metric("fid")
class FIDMetric(BaseMetric):
    """Fréchet Inception Distance (lower is better).

    Computes FID between two sets of images using InceptionV3 features.
    Requires `pip install scipy` (and `torchvision` for InceptionV3).

    For large-scale FID computation, consider using `pytorch-fid` CLI directly:
        python -m pytorch_fid path/to/real path/to/generated
    """

    def __init__(self, device: str = "cuda:0", dims: int = 2048, **kwargs):
        self.device = device
        self.dims = dims
        self._model = None

        try:
            from torchvision.models import inception_v3
            self._model = inception_v3(
                pretrained=True, transform_input=False
            ).to(device)
            self._model.eval()
            # Remove final FC layer to get features
            self._model.fc = torch.nn.Identity()
        except Exception as e:
            print(f"[FIDMetric] Could not load InceptionV3: {e}")

    def _get_features(self, images: torch.Tensor) -> np.ndarray:
        """Extract InceptionV3 features from a batch of images."""
        # Resize to 299x299 (InceptionV3 input size)
        resized = torch.nn.functional.interpolate(
            images, size=(299, 299), mode="bilinear", align_corners=False
        )
        # Normalize to [-1, 1]
        resized = resized * 2 - 1

        with torch.no_grad():
            features = self._model(resized.to(self.device))

        return features.cpu().numpy()

    def compute(self, original: torch.Tensor, reconstructed: torch.Tensor,
                **kwargs) -> float:
        """
        Compute FID between original and reconstructed image batches.

        Args:
            original:      [B, C, H, W] float [0, 1]
            reconstructed: [B, C, H, W] float [0, 1]

        Returns:
            FID score (scalar). Lower = more similar distributions.
            Note: FID is most meaningful with large batches (>= 50 images).
        """
        if self._model is None or not HAS_SCIPY:
            if not HAS_SCIPY:
                print("[FIDMetric] scipy not installed. Run: pip install scipy")
            return 0.0

        if original.ndim == 3:
            original = original.unsqueeze(0)
        if reconstructed.ndim == 3:
            reconstructed = reconstructed.unsqueeze(0)

        # Get features
        feat_real = self._get_features(original)
        feat_fake = self._get_features(reconstructed)

        # Compute statistics
        mu_real = np.mean(feat_real, axis=0)
        mu_fake = np.mean(feat_fake, axis=0)
        sigma_real = np.cov(feat_real, rowvar=False)
        sigma_fake = np.cov(feat_fake, rowvar=False)

        # Handle single-image case (cov returns scalar)
        if sigma_real.ndim == 0:
            sigma_real = np.array([[sigma_real]])
        if sigma_fake.ndim == 0:
            sigma_fake = np.array([[sigma_fake]])

        # Compute FID
        diff = mu_real - mu_fake
        covmean = sqrtm(sigma_real @ sigma_fake)

        # Numerical stability: handle complex results
        if np.iscomplexobj(covmean):
            covmean = covmean.real

        fid = float(
            diff @ diff + np.trace(sigma_real + sigma_fake - 2 * covmean)
        )
        return max(fid, 0.0)  # FID should be non-negative

    def name(self) -> str:
        return "FID"

    def higher_is_better(self) -> bool:
        return False


# ═══════════════════════════════════════════════════════════════
#  VMAF — Video Multimethod Assessment Fusion
# ═══════════════════════════════════════════════════════════════

@register_metric("vmaf")
class VMAFMetric(BaseMetric):
    """Video Multimethod Assessment Fusion (higher is better).

    Requires `ffmpeg` with `libvmaf` support, and `ffmpeg-python`.
    """

    def __init__(self, **kwargs):
        pass

    def compute(self, original_path: str, reconstructed_path: str, **kwargs) -> float:
        """
        Compute VMAF between an original and a reconstructed video file.
        Note: This metric is file-based rather than tensor-based because
        VMAF is computed over the entire video sequence using ffmpeg.

        Args:
            original_path: path to the reference video.
            reconstructed_path: path to the distorted/reconstructed video.

        Returns:
            VMAF score (scalar).
        """
        try:
            import ffmpeg
            import json
            import tempfile
            import os
            
            with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp_file:
                log_path = tmp_file.name
                
            # ffmpeg -i distorted.mp4 -i reference.mp4 -lavfi libvmaf="log_path=vmaf.json:log_fmt=json" -f null -
            main = ffmpeg.input(reconstructed_path)
            ref = ffmpeg.input(original_path)
            
            # Note: order is distorted, then reference
            out = ffmpeg.filter([main, ref], 'libvmaf', log_path=log_path, log_fmt='json')
            
            out = ffmpeg.output(out, 'pipe:', format='null')
            out.run(capture_stdout=True, capture_stderr=True)
            
            with open(log_path, 'r') as f:
                vmaf_log = json.load(f)
                
            os.remove(log_path)
            
            if "VMAF score" in vmaf_log.get("pooled_metrics", {}).get("vmaf", {}):
                return float(vmaf_log["pooled_metrics"]["vmaf"]["mean"])
                
            return 0.0
        except Exception as e:
            print(f"[VMAFMetric] Failed to compute VMAF: {e}")
            return 0.0

    def name(self) -> str:
        return "VMAF"

    def higher_is_better(self) -> bool:
        return True
