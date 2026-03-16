"""
MLOps Monitoring Adapters.
Provides an abstraction over WandB and TensorBoard for tracking experiment metrics.
"""
from typing import Dict, Any, Optional

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False


class BaseLogger:
    """Abstract Logger Interface."""
    def log_metrics(self, metrics: Dict[str, float], step: int):
        pass
        
    def finish(self):
        pass


class WandbLogger(BaseLogger):
    def __init__(self, project: str, run_name: str, config: Optional[Dict[str, Any]] = None):
        if not HAS_WANDB:
            raise ImportError("WandB is not installed. Run: pip install wandb")
        
        # Determine if we should be silent
        import os
        os.environ["WANDB_SILENT"] = "true"
        
        self.run = wandb.init(project=project, name=run_name, config=config, reinit=True)
        
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        if step is not None:
            self.run.log(metrics, step=step)
        else:
            self.run.log(metrics)
            
    def finish(self):
        self.run.finish()


class TensorBoardLogger(BaseLogger):
    def __init__(self, log_dir: str):
        if not HAS_TENSORBOARD:
            raise ImportError("TensorBoard is not installed (likely missing future module).")
            
        self.writer = SummaryWriter(log_dir=log_dir)
        
    def log_metrics(self, metrics: Dict[str, float], step: int):
        for k, v in metrics.items():
            self.writer.add_scalar(k, v, step)
            
    def finish(self):
        self.writer.close()


def get_logger(logger_type: str, run_name: str, log_dir: str, config: Dict[str, Any] = None) -> BaseLogger:
    """Factory to spin up requested logger."""
    if logger_type.lower() == "wandb":
        return WandbLogger(project="VisionStream", run_name=run_name, config=config)
    elif logger_type.lower() == "tensorboard":
        return TensorBoardLogger(log_dir=log_dir)
    else:
        # Fallback dummy logger
        return BaseLogger()
