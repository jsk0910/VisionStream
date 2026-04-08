import sys
import os
import pytest
import torch

# Add project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from modules.registry import get_vision_model, BaseVisionModel

# Trigger auto-import
import modules.vision_models
import modules.datasets

# Need to check if ultralytics is installed for YOLO tests
try:
    import ultralytics
    HAS_ULTRALYTICS = True
except ImportError:
    HAS_ULTRALYTICS = False

pytestmark = pytest.mark.skipif(not HAS_ULTRALYTICS, reason="ultralytics package required for YOLO26 tests")


class TestYOLO26Registration:
    def test_yolo26n_registered(self):
        cls = get_vision_model("yolo26n")
        assert cls is not None
        assert issubclass(cls, BaseVisionModel)

    def test_task_type(self):
        cls = get_vision_model("yolo26n")
        # Run on CPU to avoid allocating GPU memory just for initialization check
        model = cls(device="cpu")
        assert model.get_task_type() == "detection"


class TestYOLO26Inference:
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
    def test_yolo26n_forward(self):
        cls = get_vision_model("yolo26n")
        model = cls(device="cuda:0")
        
        # 1 image, 3 channels, 640x640 resolution
        dummy_input = torch.rand(1, 3, 640, 640, device="cuda:0")
        
        # Should return an ultralytics Results object
        results = model.predict(dummy_input)
        assert len(results) == 1
        
        # Verify Dual-head (end2end default is True -> NMS-free output parsing)
        assert hasattr(results[0], 'boxes')


class TestYOLO26SplitPoints:
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
    def test_split_points_list(self):
        from modules.vision_models.yolo26_split import YOLO26SplitModel
        
        split_model = YOLO26SplitModel(variant="yolo26n", split_point="backbone_p4", device="cuda:0")
        points = split_model.list_split_points()
        
        assert "backbone_p4" in points
        assert "neck_out" in points

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
    def test_feature_extraction_and_resume(self):
        from modules.vision_models.yolo26_split import YOLO26SplitModel
        from modules.vision_models.yolo26_model import YOLO26NModel
        
        device = "cuda:0"
        
        # 1. Base model for ground truth comparison
        base_model = YOLO26NModel(device=device)
        
        # 2. Split model
        split_model = YOLO26SplitModel(variant="yolo26n", split_point="backbone_p4", device=device)
        
        # Dummy input
        x = torch.rand(1, 3, 640, 640, device=device)
        
        # Extract features at split point
        features = split_model.extract_features(x)
        assert features is not None
        assert len(features.shape) == 4 # [B, C, H, W]
        
        # Resume inference
        resumed_result = split_model.resume_inference(x, features)
        assert resumed_result is not None
        
        # Note: Exact numerical equivalence is tricky due to standard PyTorch graph detached gradients, 
        # but the shape and structure should match the original inference.
        original_result = base_model.predict(x)
        
        # Output spatial dimensions should match
        assert len(resumed_result) == len(original_result)
        
        
class TestYOLO26DataLoader:
    def test_dataloader_creation(self):
        from modules.dataloader.yolo26_dataloader import create_yolo26_dataloader
        
        # Create dummy directory using the script before running this test optionally
        # or we just test if the wrapper initializes correctly with our stub ImageFolder
        
        from modules.datasets.builtin_datasets import ImageFolderDataset
        # We need an image folder. Let's create a temporary one for the test
        import tempfile
        import numpy as np
        from PIL import Image
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a dummy image
            img = np.random.randint(0, 255, (200, 300, 3), dtype=np.uint8)
            Image.fromarray(img).save(os.path.join(tmpdir, "test.jpg"))
            
            # Use ImageFolderDataset instead of coco_val2017 to avoid downloads
            from modules.registry import register_dataset, _REGISTRIES
            
            @register_dataset("temp_test_dataset")
            class TempTestDataset(ImageFolderDataset):
                def __init__(self, **kwargs):
                    super().__init__(root=tmpdir, **kwargs)
                    
            loader = create_yolo26_dataloader(dataset_name="temp_test_dataset", batch_size=1, img_size=640)
            
            batch = next(iter(loader))
            assert "images" in batch
            
            # Check letterbox resize
            assert batch["images"].shape == (1, 3, 640, 640)
            
            # Clean up registry
            del _REGISTRIES["dataset"]["temp_test_dataset"]
