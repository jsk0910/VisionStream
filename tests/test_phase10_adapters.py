"""
Phase 10 Adapter Verification Tests — validates interface compliance
for all modified adapters (super_resolution, hub_datasets, metrics).
"""
import sys
import os
import pytest
import torch

# Add project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../build')))

from modules.registry import (
    BaseVisionModel, BaseDataset, BaseMetric,
    get_vision_model, get_dataset, get_metric,
)

# Trigger auto-import
import modules.vision_models
import modules.datasets
import tools.metrics.builtin_metrics
import tools.metrics.adapters


# ═══════════════════════════════════════════════════════════════
#  Super Resolution Adapter Tests
# ═══════════════════════════════════════════════════════════════

class TestSuperResolution:
    def test_real_esrgan_registered(self):
        """RealESRGAN should be registered in the vision model registry."""
        cls = get_vision_model("real_esrgan")
        assert cls is not None

    def test_swinir_registered(self):
        """SwinIR should be registered in the vision model registry."""
        cls = get_vision_model("swinir")
        assert cls is not None

    def test_real_esrgan_is_base_vision_model(self):
        """RealESRGAN class should inherit from BaseVisionModel."""
        cls = get_vision_model("real_esrgan")
        assert issubclass(cls, BaseVisionModel)

    def test_swinir_is_base_vision_model(self):
        """SwinIR class should inherit from BaseVisionModel."""
        cls = get_vision_model("swinir")
        assert issubclass(cls, BaseVisionModel)

    def test_real_esrgan_fallback_shape(self):
        """RealESRGAN should produce correct output shape (even with bicubic fallback)."""
        cls = get_vision_model("real_esrgan")
        # Instantiate without actual model (will use bicubic fallback)
        model = cls(scale=4, device="cpu")
        x = torch.rand(1, 3, 64, 64)
        out = model.predict(x)
        assert out.shape == (1, 3, 256, 256), f"Expected (1,3,256,256), got {out.shape}"

    def test_swinir_fallback_shape(self):
        """SwinIR should produce correct output shape (even with bicubic fallback)."""
        cls = get_vision_model("swinir")
        model = cls(device="cpu")
        x = torch.rand(1, 3, 64, 64)
        out = model.predict(x)
        assert out.shape == (1, 3, 256, 256), f"Expected (1,3,256,256), got {out.shape}"

    def test_task_type(self):
        """Both SR models should report task type as 'super_resolution'."""
        for name in ["real_esrgan", "swinir"]:
            cls = get_vision_model(name)
            model = cls(device="cpu")
            assert model.get_task_type() == "super_resolution"


# ═══════════════════════════════════════════════════════════════
#  Hub Datasets Tests
# ═══════════════════════════════════════════════════════════════

class TestHubDatasets:
    def test_coco_registered(self):
        """COCO val2017 should be registered."""
        cls = get_dataset("coco_val2017")
        assert cls is not None

    def test_div2k_registered(self):
        """DIV2K val should be registered."""
        cls = get_dataset("div2k_val")
        assert cls is not None

    def test_coco_is_base_dataset(self):
        """COCOValDataset should inherit from BaseDataset."""
        cls = get_dataset("coco_val2017")
        assert issubclass(cls, BaseDataset)

    def test_div2k_is_base_dataset(self):
        """DIV2KValDataset should inherit from BaseDataset."""
        cls = get_dataset("div2k_val")
        assert issubclass(cls, BaseDataset)

    def test_coco_has_interface_methods(self):
        """COCOValDataset should have __getitem__ and __len__."""
        cls = get_dataset("coco_val2017")
        assert hasattr(cls, '__getitem__')
        assert hasattr(cls, '__len__')

    def test_div2k_has_interface_methods(self):
        """DIV2KValDataset should have __getitem__ and __len__."""
        cls = get_dataset("div2k_val")
        assert hasattr(cls, '__getitem__')
        assert hasattr(cls, '__len__')


# ═══════════════════════════════════════════════════════════════
#  Metric Adapter Tests
# ═══════════════════════════════════════════════════════════════

class TestMetricAdapters:
    def test_lpips_registered(self):
        """LPIPS should be registered."""
        cls = get_metric("lpips")
        assert cls is not None

    def test_map_registered(self):
        """mAP should be registered."""
        cls = get_metric("map")
        assert cls is not None

    def test_fid_registered(self):
        """FID should be registered."""
        cls = get_metric("fid")
        assert cls is not None

    def test_lpips_is_base_metric(self):
        """LPIPS should inherit from BaseMetric."""
        cls = get_metric("lpips")
        assert issubclass(cls, BaseMetric)

    def test_map_is_base_metric(self):
        """mAP should inherit from BaseMetric."""
        cls = get_metric("map")
        assert issubclass(cls, BaseMetric)

    def test_fid_is_base_metric(self):
        """FID should inherit from BaseMetric."""
        cls = get_metric("fid")
        assert issubclass(cls, BaseMetric)

    def test_all_metrics_have_name(self):
        """All adapter metrics must implement name() returning a string."""
        for metric_name in ["lpips", "map", "fid"]:
            cls = get_metric(metric_name)
            instance = cls.__new__(cls)
            # Call name directly without __init__ (avoid GPU deps)
            assert isinstance(cls.name(instance), str), \
                f"{metric_name}.name() should return str"

    def test_all_metrics_have_higher_is_better(self):
        """All adapter metrics must implement higher_is_better()."""
        for metric_name in ["lpips", "map", "fid"]:
            cls = get_metric(metric_name)
            instance = cls.__new__(cls)
            result = cls.higher_is_better(instance)
            assert isinstance(result, bool), \
                f"{metric_name}.higher_is_better() should return bool"

    def test_lpips_lower_is_better(self):
        """LPIPS: lower is better."""
        cls = get_metric("lpips")
        instance = cls.__new__(cls)
        assert cls.higher_is_better(instance) is False

    def test_map_higher_is_better(self):
        """mAP: higher is better."""
        cls = get_metric("map")
        instance = cls.__new__(cls)
        assert cls.higher_is_better(instance) is True

    def test_fid_lower_is_better(self):
        """FID: lower is better."""
        cls = get_metric("fid")
        instance = cls.__new__(cls)
        assert cls.higher_is_better(instance) is False


# ═══════════════════════════════════════════════════════════════
#  Environment File Existence Tests
# ═══════════════════════════════════════════════════════════════

class TestEnvironmentFiles:
    PROJECT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

    @pytest.mark.parametrize("cuda_ver", ["118", "121", "128"])
    def test_conda_env_exists(self, cuda_ver):
        path = os.path.join(self.PROJECT, f"envs/conda/environment_cuda{cuda_ver}.yml")
        assert os.path.isfile(path), f"Missing: {path}"

    @pytest.mark.parametrize("cuda_ver", ["118", "121", "128"])
    def test_pip_requirements_exists(self, cuda_ver):
        path = os.path.join(self.PROJECT, f"envs/requirements/requirements_cuda{cuda_ver}.txt")
        assert os.path.isfile(path), f"Missing: {path}"

    @pytest.mark.parametrize("cuda_ver", ["118", "121", "128"])
    def test_dockerfile_exists(self, cuda_ver):
        path = os.path.join(self.PROJECT, f"envs/docker/Dockerfile.cuda{cuda_ver}")
        assert os.path.isfile(path), f"Missing: {path}"

    @pytest.mark.parametrize("script", ["setup_conda.sh", "setup_pip.sh", "launch_docker.sh"])
    def test_script_exists_and_executable(self, script):
        path = os.path.join(self.PROJECT, f"scripts/{script}")
        assert os.path.isfile(path), f"Missing: {path}"
        assert os.access(path, os.X_OK), f"Not executable: {path}"
