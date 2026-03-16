"""
Experiment Versioning Utility
Tracks experiment configurations alongside code state (via Git hash) to ensure reproducibility.
"""
import os
import json
import subprocess
from datetime import datetime
from typing import Dict, Any

def get_git_hash() -> str:
    """Retrieve the current Git commit hash."""
    try:
        commit_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD'], stderr=subprocess.STDOUT)
        return commit_hash.decode('utf-8').strip()
    except Exception:
        return "unknown"

class ExperimentTracker:
    """Records experiment parameters and auto-generates run names."""
    
    def __init__(self, exp_name: str, base_dir: str = "user_workspace/experiments"):
        self.exp_name = exp_name
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_id = f"{exp_name}_{self.timestamp}"
        self.git_hash = get_git_hash()
        
        self.run_dir = os.path.join(base_dir, self.run_id)
        os.makedirs(self.run_dir, exist_ok=True)
        
    def save_config(self, config: Dict[str, Any]):
        """Save the experiment configuration along with metadata."""
        meta_config = {
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "git_hash": self.git_hash,
            "config": config
        }
        
        path = os.path.join(self.run_dir, "config.json")
        with open(path, "w") as f:
            json.dump(meta_config, f, indent=4)
            
        return path
        
    def get_run_dir(self) -> str:
        return self.run_dir
