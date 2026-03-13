"""
Classification Adapter — timm wrapper for BaseVisionModel.
"""
import torch
import torch.nn.functional as F
from typing import Any, Dict, Optional
from modules.registry import BaseVisionModel, register_vision_model

try:
    import timm
    HAS_TIMM = True
except ImportError:
    HAS_TIMM = False


@register_vision_model("timm_classifier")
class TimmClassifier(BaseVisionModel):
    """Wrapper for any timm model for classification tasks."""

    def __init__(self, model_name: str = "resnet50", pretrained: bool = True, 
                 device: str = "cuda:0", **kwargs):
        if not HAS_TIMM:
            raise ImportError("Please install timm: pip install timm")
        
        self.device = device
        self.model_name = model_name
        self.model = timm.create_model(model_name, pretrained=pretrained).to(device)
        self.model.eval()
        
        # Get data config for preprocessing
        self.data_config = timm.data.resolve_data_config(self.model.pretrained_cfg)
        self.transform = timm.data.create_transform(**self.data_config, is_training=False)

    def predict(self, x: torch.Tensor) -> Any:
        """
        x: Input tensor (B, C, H, W) or (C, H, W)
        """
        if x.ndim == 3:
            x = x.unsqueeze(0)
        
        x = x.to(self.device)
        
        # Note: timm's create_transform expects PIL or numpy usually,
        # but if we have a tensor, we might need to handle normalization manually
        # as per data_config. For now, assuming x is already [0, 1] float tensor.
        
        with torch.no_grad():
            output = self.model(x)
            probabilities = F.softmax(output, dim=1)
            top5_prob, top5_catid = torch.topk(probabilities, 5)
            
        return {
            "probabilities": probabilities,
            "top5_catid": top5_catid,
            "top5_prob": top5_prob
        }

    def get_task_type(self) -> str:
        return "classification"


@register_vision_model("resnet50")
class ResNet50(TimmClassifier):
    def __init__(self, **kwargs):
        super().__init__(model_name="resnet50", **kwargs)


@register_vision_model("vit_base_patch16_224")
class ViTBase(TimmClassifier):
    def __init__(self, **kwargs):
        super().__init__(model_name="vit_base_patch16_224", **kwargs)
